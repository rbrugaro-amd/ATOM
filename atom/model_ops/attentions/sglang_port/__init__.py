# SPDX-License-Identifier: MIT
"""Kernels vendored verbatim from SGLang, for the DFlash GQA draft read.

Source: lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260916,
/sgl-workspace/sglang/python/sglang/kernels/ops/

  verify_mla.py   attention/verify_mla.py
  kv_indices.py   kvcache/kv_indices.py

Copied rather than reimplemented because this is the exact code path AMD-MLPerf
run 36680198889 validated at TP8 / DCP8 / DSpark k=5 against THIS draft
checkpoint (RadixArk/Kimi-K3-DSpark @ 3c5bac30), including
`--speculative-draft-kv-cache-dtype fp8_e4m3`. Reproducing the math from the
paper would forfeit that provenance.

The only edit is in verify_mla.py: its single cross-file import
(`_AMD_LAUNCH_KWARGS` from verify_splitkv.py) is inlined as the two-key literal
that file defines for HIP. Both modules are otherwise byte-identical and import
nothing but torch/triton, so neither drags in an `sglang` dependency.

Despite the name, verify_mla is NOT MLA-only: NOPE_DIM/PE_DIM generalize it, and
at PE_DIM == 0 it is a plain GQA attention over separate K and V buffers. Its
own tuning table names our case -- `_BLOCK_CONFIG[64] = (4, 256, 4)  # K3 GQA
(dspark draft attention)`.
"""
