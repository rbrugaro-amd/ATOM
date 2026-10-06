#!/usr/bin/env bash
# A/B harness for the Kimi-K3 DSpark draft arms.
#
#   ./tools/dflash_ab.sh <arm> <workload>
#     arm      : radixark | inferact | nospec
#     workload : smoke | lossless | gsm8k | gsm8k_card | accept | swebench
#                | perf | trace
#
# gsm8k is the M2 exit criterion (FEWSHOT=5 default, LIMIT=N for a quick pass).
# gsm8k_card is the same dataset measured the way the MODEL CARDS measure it
# (chat template, per-question acceptance averaged equally over questions,
# concurrency 1), so our acc_len is comparable to the published figure. It
# reports the card-style AND the global estimator on the same run.
#
# Bring-up shape is deliberately NOT the C32 production recipe: TP8 / DCP1, no
# LMCache, no prefix caching, so the draft is the only variable. DCP8 is a later
# milestone -- DCP is broken for a non-MLA draft in vLLM and SGLang too
# (vllm#59184, #57992; sglang#34920), so it is isolated on purpose.
#
# --spec-decode-acceptance-length is NEVER passed. It forces a fixed mean and
# makes the rejection sampler ignore real draft/target agreement, i.e. it would
# fabricate the exact number this harness exists to measure.
set -euo pipefail

ARM="${1:?arm: radixark|inferact|nospec}"
WORKLOAD="${2:-smoke}"

HF=/data/workloads-inference/hf_hub_cache
MODEL=$HF/models--moonshotai--Kimi-K3/snapshots/f831ab66814297da540d832a5235f8e904f29d06
RADIXARK=$HF/models--RadixArk--Kimi-K3-DSpark/snapshots/3c5bac301d9cf392706189d82ed947feca6c2f0f
INFERACT=$HF/models--Inferact--Kimi-K3-DSpark/snapshots/cf6b8244620e7ea4b0651d214f28e89eac75bed6

IMAGE=rocm/atom-dev:nightly_202610051453
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PORT=30000
K=${K:-5}
RUNDIR=${RUNDIR:-/tmp/dflash_ab/$ARM}
mkdir -p "$RUNDIR"

case "$ARM" in
  radixark) SPEC_ARGS="--method dspark --draft-model $RADIXARK --num-speculative-tokens $K" ;;
  inferact) SPEC_ARGS="--method dspark --draft-model $INFERACT --num-speculative-tokens $K" ;;
  nospec)   SPEC_ARGS="" ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac

CNAME="k3-ab-$ARM"

KEEP_ON_FAIL=${KEEP_ON_FAIL:-1}

stop_server() {
  docker rm -f "$CNAME" >/dev/null 2>&1 || true
}

on_exit() {
  local rc=$?
  # Keep the container on failure: `docker rm -f` destroys the logs, which are
  # the only evidence of why bring-up failed.
  if (( rc != 0 )) && [[ "$KEEP_ON_FAIL" == "1" ]]; then
    echo "FAILED (rc=$rc); container '$CNAME' left for inspection:" >&2
    echo "  docker logs --tail 200 $CNAME" >&2
  else
    stop_server
  fi
}
trap on_exit EXIT

# Profiling follows the capture-trace skill; see it for the rationale behind
# each knob. Both the flag and the ATOM_PROFILER_MORE env go on the SERVER --
# the profiler runs inside the model-runner worker processes, so setting them
# on the client does nothing (skill, pre-flight #3).
# M3 (production shape) knob. Default OFF = the bring-up shape every result so
# far was measured in, so nothing already banked shifts underneath a re-run.
#
# This is the one M3 delta with a concrete hazard worth naming: the draft's
# sibling KV pool rides the TARGET's block ids (`DraftKvBuilder.sub_pool_specs`
# joins the target's PAGE entry class). If the target serves a cached prefix
# block WITHOUT recomputing it, `compute_draft_kv` never writes the draft's
# context rows for those positions, and the block pass then attends whatever
# those pool rows last held. That is invisible except as lost acceptance --
# exactly the signature this project already chased once.
# `--cudagraph-mode FULL` and `--max-num-seqs 64` are ALREADY in effect, so
# prefix caching and LMCache are the only real M3 deltas.
PREFIX_CACHE_ARG="--no-enable_prefix_caching"
if [[ "${PREFIX_CACHING:-0}" == "1" ]]; then
  PREFIX_CACHE_ARG="--enable_prefix_caching"
fi

# ---- LMCACHE=1 : CPU KV offload, ported from the C32 production recipe -------
# (AMD-MLPerf/mlperf-endpoints configs/kimi-k3/mi355x/atom/server_C32.env)
#
# Offload implies prefix caching: the point of pushing KV to host RAM is to get
# it back on a later hit. The recipe enables both, so LMCACHE=1 turns prefix
# caching on too unless PREFIX_CACHING is set explicitly. This is also the one
# plausible explanation for the 0.0% prefix-cache hit rate measured earlier
# (RESUME: "ESCALATED") -- that run had caching on with NO offload backend.
#
# THE HAZARD, named before running: the draft's sibling KV pool registers its
# own transfer regions as `draft.{role}` at `tp_replication_factor=1`
# ("a paged MHA/GQA draft is a per-rank shard", draft_kv.py:258-271). Offloading
# and restoring a PER-RANK-SHARDED region alongside the target's replicated ones
# is a path nobody has exercised. If draft rows are offloaded and restored
# wrongly -- or silently not restored -- the block pass attends stale KV, which
# shows up ONLY as lost acceptance. Watch acceptance, not just "it ran".
#
# NUMA: the recipe pins ATOM_NUMA_NODE=0,0,0,0,1,1,1,1 (8 ranks -> 2 nodes).
# THIS BOX HAS 8 NUMA NODES, so that mapping would crowd every rank onto nodes
# 0-1. Left on auto deliberately; do not copy the recipe's pinning here.
#
# Sizing: LMCACHE_MAX_LOCAL_CPU_SIZE is PER RANK, so TP8 pins 8 x 128 = 1024 GB
# of host RAM. Verified 2938 GB available before enabling.
LMCACHE_ARG=""
if [[ "${LMCACHE:-0}" == "1" ]]; then
  LMCACHE_ARG='--kv-transfer-config {"kv_connector":"lmcache_offload","kv_role":"offload"}'
  [[ "${PREFIX_CACHING:-1}" != "0" ]] && PREFIX_CACHE_ARG="--enable_prefix_caching"
fi

# ---- DCP=<n> : decode context parallel ---------------------------------------
# DCP reuses the TP group's GPUs, so DCP=8 at -tp 8 is the same 8 devices with
# the KV sharded by token. The draft then replicates its KV heads across that
# group (tp_eff = tp/dcp) and merges the per-rank partial softmaxes by LSE.
# DCP=1 (default) emits NO flag at all, so the DCP1 command line is unchanged.
DCP_ARG=""
if [[ "${DCP:-1}" != "1" ]]; then
  DCP_ARG="--decode-context-parallel-size ${DCP}"
fi

TRACE_DIR="${TRACE_DIR:-/tmp/dflash_ab/trace/$ARM}"
EXTRA_ENVS=()
# NOTE: every `EXTRA_ENVS+=` MUST come after this line. It used to be below the
# LMCACHE block, which silently discarded the offload envs and turned LMCACHE=1
# into a no-op that still looked like a successful run.
if [[ "${LMCACHE:-0}" == "1" ]]; then
  EXTRA_ENVS+=(
    -e "LMCACHE_LOCAL_CPU=True"
    -e "LMCACHE_CHUNK_SIZE=${LMCACHE_CHUNK_SIZE:-1024}"
    -e "LMCACHE_NUMA_MODE=auto"
    -e "LMCACHE_MAX_LOCAL_CPU_SIZE=${LMCACHE_MAX_LOCAL_CPU_SIZE:-128}"
    -e "OFFLOAD_KV_FOR_HYBRID=1"
    -e "OFFLOAD_GPU_STAGING_CHUNKS=32"
  )
fi
# MEASUREMENT-ONLY: pass the K3 aux-feature probe through when set on the host.
if [[ "${ATOM_K3_AUX_PROBE:-0}" == "1" ]]; then
  EXTRA_ENVS+=(-e "ATOM_K3_AUX_PROBE=1")
fi
# BEHAVIOURAL (falsifier arm): feed the draft the K3 aggregate stream instead
# of the raw block prefix. See RESUME.md "rung 0 step 2".
if [[ "${ATOM_K3_AUX_AGGREGATE:-0}" == "1" ]]; then
  EXTRA_ENVS+=(-e "ATOM_K3_AUX_AGGREGATE=1")
fi
PROFILE_ARGS=""
if [[ "$WORKLOAD" == "trace" ]]; then
  # A stale dir gives a confusing mixed per-rank layout; the server does not
  # clear it (skill, pre-flight #4).
  rm -rf "$TRACE_DIR"; mkdir -p "$TRACE_DIR"
  PROFILE_ARGS="--torch-profiler-dir $TRACE_DIR"
  # 0 = kernel names only (~50-80 MB/rank). 1 adds shapes+stack+memory and
  # risks OOMing the exporter (skill, parameter table).
  EXTRA_ENVS+=(-e "ATOM_PROFILER_MORE=${ATOM_PROFILER_MORE:-0}")
fi

wait_trace_export() {
  # MANDATORY. Kineto exports LAZILY on the worker side: /stop_profile returns
  # immediately while the per-rank .json write + gzip takes 10-60 s. Tearing the
  # container down here -- which is exactly what the success path does -- would
  # destroy every trace. (capture-trace skill, step 3.)
  #
  # Completion is ".gz present AND no orphan .json": the exporter writes the raw
  # .json, then gzips and unlinks it, so a surviving .json means it died
  # mid-export. Size of the .gz alone is NOT a valid signal.
  local tp=8
  echo "waiting for the kineto exporter (async; 10-60s is normal)..."
  for _ in $(seq 1 60); do
    local gz json
    gz=$(find "$TRACE_DIR" -name "*.pt.trace.json.gz" 2>/dev/null | wc -l)
    json=$(find "$TRACE_DIR" -name "*.pt.trace.json" ! -name "*.gz" 2>/dev/null | wc -l)
    echo "  gz=$gz/$tp orphan_json=$json"
    if (( gz >= tp && json == 0 )); then echo "export complete"; return 0; fi
    sleep 10
  done
  echo "WARN: exporter did not finish; trace may be truncated" >&2
  return 0
}

start_server() {
  stop_server
  # Stale compile cache causes silent failures after code changes (CLAUDE.md).
  rm -rf "$HOME/.cache/atom"/* 2>/dev/null || true
  docker run -d --name "$CNAME" \
    --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined \
    --group-add video --ipc=host --shm-size 64G --network host \
    -v "$REPO":/work -w /work -v "$HF":"$HF":ro -v /tmp:/tmp \
    -e AITER_LOG_LEVEL=WARNING -e PYTHONNOUSERSITE=1 -e PYTHONHASHSEED=0 \
    "${EXTRA_ENVS[@]}" \
    "$IMAGE" \
    python3 -m atom.entrypoints.openai_server \
      --model "$MODEL" \
      --served-model-name moonshotai/Kimi-K3 \
      --host 0.0.0.0 --server-port $PORT \
      -tp 8 \
      $DCP_ARG \
      --kv_cache_dtype fp8 \
      --block-size 128 \
      --max-num-seqs 64 \
      --max-num-batched-tokens 8192 \
      --gpu-memory-utilization 0.9 \
      $PREFIX_CACHE_ARG \
      $LMCACHE_ARG \
      --trust-remote-code \
      --tool-call-parser kimi_k3 \
      $SPEC_ARGS \
      $PROFILE_ARGS \
      >/dev/null
  # `docker run -d` prints the container id, not the server output. Stream the
  # real logs out of the container instead.
  ( docker logs -f "$CNAME" > "$RUNDIR/server.log" 2>&1 & )
  echo "started $CNAME (arm=$ARM k=$K) -> $RUNDIR/server.log"
}

wait_ready() {
  # VRAM, not curl /health: /health can return OK with no model loaded (CLAUDE.md).
  local deadline=$((SECONDS + ${READY_TIMEOUT:-2400}))
  while (( SECONDS < deadline )); do
    if ! docker ps --format '{{.Names}}' | grep -qx "$CNAME"; then
      echo "FAIL: container exited early; tail of log:" >&2
      tail -40 "$RUNDIR/server.log" >&2; return 1
    fi
    # `|| true`: rocm-smi exits nonzero on this box (it prints "Not supported on
    # the given system" for some fields) and pipefail would kill the script.
    # Last field, not $2: the line is "GPU[0] : GPU Memory Allocated (VRAM%): 72",
    # so splitting on ": " puts the LABEL in $2 and the number in $3.
    local vram=""
    vram=$(rocm-smi --showmemuse 2>/dev/null | awk -F': ' '/VRAM%/{print $NF+0; exit}' || true)
    if [[ -n "$vram" && "$vram" -gt 0 ]] && curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "ready: VRAM%=$vram"; return 0
    fi
    sleep 15
  done
  echo "FAIL: not ready within timeout" >&2; tail -40 "$RUNDIR/server.log" >&2; return 1
}

run_gsm8k() {
  # The M2 exit criterion. Exact-text losslessness is NOT achievable on this
  # stack (RESUME.md: the target does not reproduce itself run to run, and the
  # spec arm samples at qlen=k+1 while no-spec samples at qlen=1), so the real
  # correctness gate is an AGGREGATE accuracy metric, which individual near-tie
  # flips cannot move.
  #
  # lm_eval lives in the image, not on the host, so it runs via `docker exec`
  # inside the server's own container -- which also gives it the HF cache mount
  # and --network host for free. Offline: the dataset is already in the cache,
  # and letting it reach the network would make the run non-reproducible.
  local shots="${FEWSHOT:-5}"
  local limit_arg=""
  [[ -n "${LIMIT:-}" ]] && limit_arg="--limit ${LIMIT}"
  # Two non-obvious arguments, both of which failed the run outright:
  #   tokenizer=  : lm_eval builds a tokenizer from `model=`, and
  #                 "moonshotai/Kimi-K3" is a bare repo id that is NOT in the
  #                 hub cache (only the snapshot dir is), so offline it tries
  #                 the network and dies. `model=` must stay the SERVED name
  #                 because it goes into the request body; point the tokenizer
  #                 at the local snapshot instead.
  #   HF_DATASETS_CACHE : the HF cache is bind-mounted :ro, and `datasets`
  #                 takes a FileLock next to the cached dataset before reading
  #                 it -- read-only is fatal even for a pure read. gsm8k is
  #                 4.5 MB, so it is staged to a writable copy once.
  local ds_cache=/tmp/dflash_ab/hf_datasets
  if [[ ! -d "$ds_cache/openai___gsm8k" ]]; then
    mkdir -p "$ds_cache"
    cp -r "$HF/datasets/openai___gsm8k" "$ds_cache/"
    chmod -R u+w "$ds_cache"
  fi
  docker exec \
    -e HF_HOME="$HF" -e HF_DATASETS_CACHE="$ds_cache" \
    -e HF_DATASETS_OFFLINE=1 -e HF_HUB_OFFLINE=1 \
    -e AITER_LOG_LEVEL=WARNING \
    "$CNAME" lm_eval \
      --model local-completions \
      --model_args "model=moonshotai/Kimi-K3,tokenizer=$MODEL,base_url=http://127.0.0.1:$PORT/v1/completions,num_concurrent=32,max_retries=3,tokenized_requests=False,trust_remote_code=True" \
      --tasks gsm8k \
      --num_fewshot "$shots" \
      $limit_arg \
    2>&1 | tee "$RUNDIR/${ARM}_gsm8k.log"
  echo
  echo "== $ARM / gsm8k (${shots}-shot) =="
  grep -E "flexible-extract|strict-match" "$RUNDIR/${ARM}_gsm8k.log" | head -2
}

case "$WORKLOAD" in
  trace)
    start_server
    wait_ready
    python3 "$REPO/tools/dflash_ab_client.py" \
      --arm "$ARM" --workload trace --port "$PORT" --outdir "$RUNDIR"
    wait_trace_export
    echo "trace -> $TRACE_DIR"
    find "$TRACE_DIR" -type f -name "*.gz" -exec ls -la {} \; 2>/dev/null | head
    echo
    echo "Analyse with the in-repo tool (extracts prefill/decode/draft step"
    echo "annotations and compares draft steps across iterations):"
    echo "  python3 tools/analyze_trace_summary.py $TRACE_DIR/rank_0/*.pt.trace.json.gz"
    ;;
  smoke|lossless|accept|swebench|perf)
    start_server
    wait_ready
    python3 "$REPO/tools/dflash_ab_client.py" \
      --arm "$ARM" --workload "$WORKLOAD" --port "$PORT" --outdir "$RUNDIR"
    ;;
  gsm8k_card)
    # Card-methodology GSM8K acceptance: chat template + per-question estimator,
    # concurrency 1. See the gsm8k_card section in tools/dflash_ab_client.py for
    # why both of those are forced. Runs on the HOST (datasets 5.0.1 is there
    # and the client stages its own writable dataset cache), unlike `gsm8k`,
    # which needs lm_eval from inside the image.
    start_server
    wait_ready
    python3 "$REPO/tools/dflash_ab_client.py" \
      --arm "$ARM" --workload gsm8k_card --port "$PORT" --outdir "$RUNDIR" \
      --limit "${LIMIT:-200}" --fewshot "${FEWSHOT:-5}" \
      --max-tokens "${MAX_TOKENS:-512}" \
      2>&1 | tee "$RUNDIR/${ARM}_gsm8k_card.log"
    ;;
  gsm8k)
    start_server
    wait_ready
    run_gsm8k
    ;;
  *) echo "unknown workload: $WORKLOAD" >&2; exit 2 ;;
esac
