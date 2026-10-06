"""Does the fixed DCP guard refuse the right drafts? Uses the REAL normalized
draft config (SpeculativeConfig), not raw config.json -- `needs_non_causal_kv`
keys on `dspark_with_draft`, which only normalization synthesizes."""
import glob
from atom.config import SpeculativeConfig
from atom.utils.selector import attn_family, get_attn_backend
from atom.spec_decode.draft_kv import needs_non_causal_kv

HF = "/data/workloads-inference/hf_hub_cache"
def snap(pat):
    g = glob.glob(f"{HF}/models--{pat}/snapshots/*/")
    return g[0] if g else None

K3 = snap("moonshotai--Kimi-K3")
CASES = [
    ("MTP: draft IS the target (GLM-5.2 / K3 shape)", K3, "mtp", False),
    ("DSpark MLA draft (Inferact)", snap("Inferact--Kimi-K3-DSpark"), "dspark", False),
    ("DFlash GQA draft (RadixArk)", snap("RadixArk--Kimi-K3-DSpark"), "dspark", False),
    ("Plain GQA draft (Eagle3-llama shape)", snap("amd--Llama-3.1*"), "eagle3", True),
]
print(f"{'case':46s} {'family':10s} {'owns_pool':9s} {'dflash':6s} {'REFUSED':7s} {'expect':6s} verdict")
for name, path, method, expect in CASES:
    if not path:
        print(f"  {name[:44]:44s} (no local checkpoint)"); continue
    try:
        sc = SpeculativeConfig(model=path, method=method, num_speculative_tokens=5)
        d = sc.draft_model_hf_config
        fam = attn_family(d); owns = bool(get_attn_backend(fam).DRAFT_OWNS_KV_POOL)
        df = bool(needs_non_causal_kv(d)); refused = owns and not df
        print(f"  {name[:44]:44s} {fam.value:10s} {str(owns):9s} {str(df):6s} "
              f"{str(refused):7s} {str(expect):6s} {'OK' if refused == expect else '** MISMATCH **'}")
    except Exception as e:
        print(f"  {name[:44]:44s} ERROR {type(e).__name__}: {str(e)[:60]}")
