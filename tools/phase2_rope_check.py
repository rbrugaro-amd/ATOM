"""Compare the rope ATOM builds for the DFlash draft against the reference's cos/sin."""

import math

import torch
from aiter.rotary_embedding import get_rope

# Exactly what KimiK3DFlashAttention.__init__ does for this checkpoint.
rope_params = {
    "factor": 16.0,
    "original_max_position_embeddings": 65536,
    "rope_theta": 10000.0,
    "rope_type": "yarn",
}
rope_theta = rope_params.get("rope_theta") or 10000.0
rope_scaling = dict(rope_params)
rope_scaling.setdefault(
    "original_max_position_embeddings", 1048576
)  # no-op: key present
emb = get_rope(
    64,
    rotary_dim=64,
    max_position=1048576,
    base=rope_theta,
    rope_scaling=rope_scaling,
    is_neox_style=True,
)
print("class:", type(emb).__name__)
for a in ("mscale", "scaling_factor", "attn_factor", "base", "rotary_dim"):
    if hasattr(emb, a):
        print(f"  {a} = {getattr(emb, a)}")
expected = 0.1 * math.log(16.0) + 1.0
print("\nreference attention_scaling = 1.2772588722239782")
print(f"0.1*ln(16)+1                = {expected!r}")
got = getattr(emb, "mscale", None)
print(f"ATOM rope mscale            = {got!r}")
if got is not None:
    print("MSCALE MATCH:", abs(got - 1.2772588722239782) < 1e-9)

ref = torch.load(
    "tools/ref_dump/ref_dflash_eager.pt", map_location="cpu", weights_only=False
)
meta, T = ref["meta"], ref["tensors"]
print("\nref meta rope_attention_scaling:", meta.get("rope_attention_scaling"))
rcos, rsin = T["r.cos"].float()[0], T["r.sin"].float()[0]  # [23, 64]
S = rcos.shape[0]

cache = emb.cos_sin_cache if hasattr(emb, "cos_sin_cache") else None
print("cos_sin_cache:", None if cache is None else tuple(cache.shape))
if cache is not None:
    c = cache[:S].float()
    half = c.shape[-1] // 2
    acos, asin = c[..., :half], c[..., half:]
    print("ATOM cos half-width:", acos.shape, " ref cos width:", rcos.shape)
    # reference duplicates the 32 freqs to 64; ATOM stores the 32 unique ones
    rc32 = rcos[:, :half] if rcos.shape[-1] == 2 * half else rcos
    rs32 = rsin[:, :half] if rsin.shape[-1] == 2 * half else rsin
    print("\nmax|ATOM cos - ref cos| =", (acos - rc32).abs().max().item())
    print("max|ATOM sin - ref sin| =", (asin - rs32).abs().max().item())
    ratio = (rc32.abs().mean() / acos.abs().mean()).item()
    print(
        "mean|ref cos| / mean|ATOM cos| =",
        round(ratio, 6),
        "  (1.0 = same convention; ~1.2773 = ATOM is MISSING the mscale)",
    )
