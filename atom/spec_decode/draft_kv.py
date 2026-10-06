import logging
from functools import partial

import torch

from atom.config import KVCacheTensor
from atom.model_ops.attentions.pool_layout.pool_rows import PoolRowsMixin
from atom.model_ops.attentions.pool_layout.sub_pool_spec import SubPoolSpec, page_pool

logger = logging.getLogger("atom")

# This pool's one row space. Named, not a geometry: `make_kv_pool` already
# settled the draft's geometry, and there is only ever the one.
DRAFT_KV_ROWS = "draft_kv"


def needs_non_causal_kv(draft_hf) -> bool:
    """Whether this draft reads its own KV pool bidirectionally.

    A BLOCK drafter proposes all its tokens in one pass and lets every position
    in the block attend every other, so its read is non-causal -- unlike a
    serial MTP/EAGLE draft, which emits one token per pass and is plainly causal.

    Asked of the config rather than the model because this runs from the
    proposer's `__init__`, before the draft model exists. `dspark_with_draft` is
    the marker `_normalize_draft_dspark_config` synthesizes for exactly the
    standalone block drafters; `kv_lora_rank` then separates the two backbones,
    and only the non-MLA one reads through a pool of its own (an MLA draft binds
    into the target's latent rows and gets its non-causal read from the MLA
    path, which already takes `attn_metadata.causal`).
    """
    return bool(getattr(draft_hf, "dspark_with_draft", False)) and not getattr(
        draft_hf, "kv_lora_rank", None
    )


def draft_kv_builder(model_runner, draft_hf):
    """The KV builder a draft needs of its own, or None if it needs none.

    A draft is not an attention flavor; it is a model that has one. So the
    flavor comes from the draft's own config through the same selector the
    target uses, and that backend says whether a draft of its flavor caches
    into a pool of its own (`DRAFT_OWNS_KV_POOL`). Where it does not, the
    runner never sees a draft builder at all.

    Which is why speculative decoding imports no attention anywhere: adding a
    flavor is a flag and a `make_kv_pool` on that backend.
    """
    from aiter import dtypes

    from atom.utils.selector import attn_family, get_attn_backend

    backend = get_attn_backend(attn_family(draft_hf))
    if not backend.DRAFT_OWNS_KV_POOL:
        return None

    non_causal = needs_non_causal_kv(draft_hf)
    kv_cache_dtype = model_runner.config.kv_cache_dtype
    if non_causal and kv_cache_dtype != "bf16":
        # A block drafter attends its own block bidirectionally, and the only
        # aiter kernel that reads a paged GQA cache non-causally at qlen > 1 is
        # `mha_v3.flash_attn_with_kvcache`. That kernel takes fp8 KV only when q
        # is ALSO fp8, and its descales are strictly 2-D (batch, kv_head) --
        # per-tensor. This pool carries per-(kv head, token) scales, which it
        # rejects by assert. Worse, omitting descales under fp8 does not raise:
        # it warns and substitutes 1.0, i.e. silently wrong numerics.
        #
        # The draft owns this pool, so it can pick its own dtype. bf16 costs 2x
        # on the DRAFT only (the target keeps its fp8 pool) and is small: at
        # 16 kv heads / head_dim 64 / 5 layers it is 2720 B per token per rank.
        logger.info(
            "Draft KV pool forced to bf16 (target is %s): a non-causal block "
            "drafter reads through mha_v3, which cannot express this pool's "
            "per-token fp8 scales",
            kv_cache_dtype,
        )
        kv_cache_dtype = "bf16"

    make_pool = partial(
        backend.make_kv_pool,
        draft_hf,
        world_size=model_runner.world_size,
        kv_dtype=dtypes.d_dtypes[kv_cache_dtype],
    )
    # A way to build it, not the pool: how many rows it has is the walk's to
    # say, and this runs from the proposer's `__init__` -- `build_drafter`
    # still running, `runner.drafter` not yet set, nothing to walk.
    return DraftKvBuilder(model_runner, make_pool, non_causal=non_causal)


class DraftKvBuilder(PoolRowsMixin):
    """A draft model's own KV pool, riding the target model's block ids.

    Implements the subset of `AttentionMetadataBuilder` hooks ModelRunner
    consults for sizing and per-module binding, so a draft that cannot share
    the target's pool fits the builder protocol without leaking into the
    target's builder. It does NOT drive prepare_decode/prepare_prefill; it
    piggybacks on the target builder's metadata flow during propose.

    It knows nothing about attention: the caller that resolved the draft's
    flavor hands over a way to build the pool. What is left is the part that is
    genuinely about being a draft -- its blocks are the target's blocks,
    re-paged at its own block size.
    """

    def __init__(self, model_runner, make_pool, non_causal: bool = False):
        self.model_runner = model_runner
        self._make_pool = make_pool
        # Decides the LAYOUT this pool is bound in, not its size. See
        # `needs_non_causal_kv` and `MhaKvPool.fa_views`.
        self._non_causal = non_causal
        self._kv_pool = None
        self.num_blocks = 0  # set in allocate_kv_cache_tensors

    @property
    def kv_pool(self):
        """The pool, built at the row count the walk found.

        On demand because the draft model does not exist at `__init__`; every
        reader is a sizing or binding step, which all run after it does. The
        count is the walk's and not `num_hidden_layers`, or the pool is sized
        off one number and addressed by another.

        Indexed and not `.get(..., 0)`: this builder exists because the flavor
        owns a pool, so no rows is a contradiction, and a pool of none prices
        at zero and binds nothing -- a draft running on no KV at all. The
        `KeyError` is the one `pool_rows` promises.

        On demand is also what makes the block reachable, and it is the target
        builder's: `propose` hands the draft the target's block tables, and the
        draft's kernels take the block off the cache they were bound to, so a
        block of its own would address one page with an id that counted
        another. There is nowhere in a draft to convert between the two -- it
        has no metadata of its own. The builder is constructed after
        `build_drafter`, which is why this is read here and not bound earlier.
        """
        if self._kv_pool is None:
            # The SCHEDULER's block, not the target builder's. The two agree for
            # a paged target, but an MLA target's builder pages at
            # ATOM_MLA_PAGE_SIZE -- 1 by default -- which is a kernel page, not
            # a block id. Sizing the draft at 1 contradicts the invariant
            # `allocate_kv_cache_tensors` enforces below (one entry per
            # scheduler block, indexed by the target's block tables, so all
            # three are one number), and an MHA draft on an MLA target cannot
            # allocate at all: MhaKvPool's SHUFFLE packing needs 16 to divide
            # the block size.
            self._kv_pool = self._make_pool(
                layers=self.row_counts()[DRAFT_KV_ROWS],
                target_block_size=self.model_runner.block_size,
            )
        return self._kv_pool

    def invalidate_pool_rows(self) -> None:
        """Drop the pool with the walk that sized it, so the two cannot part."""
        super().invalidate_pool_rows()
        self._kv_pool = None

    def sub_pool_specs(self) -> list[SubPoolSpec]:
        """`page_pool` puts the draft in the target's entry class, so the two
        contributions sum into one per-block cost instead of a second pool."""
        return [page_pool(self.kv_pool.entry_bytes)]

    def paged_pool_bytes(self, blocks: int) -> int:
        """The draft's share of the runner's one paged allocation.

        The same `entry_bytes` `sub_pool_specs` adds to the target's, so the
        draft's region is exactly what the block budget already charged for
        it -- which is what "riding the target's block ids" costs.
        """
        return self.kv_pool.pool_bytes(blocks)

    def allocate_kv_cache_tensors(self, *, blocks: int, buf) -> dict:
        """Back the draft's pool from its region. Nothing for the runner to
        setattr: the pool is this builder's, and its hooks below are the only
        readers.

        One entry per scheduler block, the count the target was built at --
        that is what riding the target's block ids means.
        """
        runner = self.model_runner
        pool = self.kv_pool
        # `blocks` counts SCHEDULER blocks -- it is the budget `paged_pool_bytes`
        # charged per -- and one entry answers one of them, so the block the
        # pool took has to be that one too. It took the target builder's, which
        # makes this the check that the target's and the scheduler's agree.
        # Raised and not asserted because `python -O` drops what it does not
        # run, and a short pool reads the next request's page rather than fault.
        if pool.block_size != runner.block_size:
            raise ValueError(
                f"the draft's blocks are {pool.block_size} tokens and the "
                f"scheduler's {runner.block_size}; a draft is allocated one "
                "entry per scheduler block and indexed by the target's block "
                "tables, so all three are one number"
            )
        self.num_blocks = blocks
        pool.allocate(blocks, runner.device, buf=buf)
        logger.info(
            "Allocated draft KV pool: %d blocks, %d B of the paged allocation",
            blocks,
            pool.pool_bytes(blocks),
        )
        return {}

    def release_kv_pools(self) -> None:
        # The field, not the property: releasing one nobody built would build it.
        if self._kv_pool is not None:
            self._kv_pool.release()

    def _pooled_models(self) -> list:
        """The draft's, and only the draft's -- this pool exists precisely
        because that model could not share the target's."""
        return [self.model_runner.drafter.model]

    def _module_kinds(self, module) -> tuple:
        """One row space, the draft's own KV rows."""
        is_paged = (
            hasattr(module, "base_attention")
            and hasattr(module, "use_mla")
            and not module.use_mla
        )
        return (DRAFT_KV_ROWS,) if is_paged else ()

    def build_kv_cache_tensor(self, module):
        """Bind one of the draft's attention modules to its own pool.

        Returns None for anything this pool does not hold, so ModelRunner falls
        through to the target builder.
        """
        if DRAFT_KV_ROWS not in self._module_kinds(module):
            return None
        runner = self.model_runner
        idx = self.pool_rows[DRAFT_KV_ROWS][module]
        # The layout the DRAFT's reader needs, which is not always the target's.
        # A non-causal block drafter reads through mha_v3, and that kernel hard
        # unpacks four axes -- SHUFFLE's x-pack splits head_dim across two of
        # them, so it cannot be offered at all. Same bytes either way.
        k_cache, v_cache = (
            self.kv_pool.fa_views(idx)
            if self._non_causal
            else self.kv_pool.kv_views(idx)
        )
        module.max_model_len = runner.config.max_model_len
        # The POOL's dtype, not the runner's: a draft may hold bf16 beside an
        # fp8 target (see `needs_non_causal_kv`), and asking a bf16 pool for
        # scale views is the KeyError `scale_views` promises.
        if self.kv_pool.kv_dtype is not torch.bfloat16:
            module.k_scale, module.v_scale = self.kv_pool.scale_views(idx)
        module.k_cache = k_cache
        module.v_cache = v_cache
        return KVCacheTensor(
            layer_num=module.layer_num,
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale=getattr(module, "k_scale", None),
            v_scale=getattr(module, "v_scale", None),
        )

    def get_kv_transfer_tensors(self):
        from atom.kv_transfer.disaggregation.page_region import page_region
        from atom.kv_transfer.disaggregation.types import KVTransferTensors

        return KVTransferTensors(
            # The draft's rows sit in the same block ids as the target's, so
            # its regions need a name that says which stack they are.
            pages=[
                page_region(t, semantic_role=f"draft.{role}")
                for role, t in self.kv_pool.region_tensors()
            ],
            # A paged MHA/GQA draft is a per-rank shard. Appending it to an
            # otherwise replicated target must disable whole-object collapse.
            tp_replication_factor=1,
        )
