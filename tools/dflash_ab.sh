#!/usr/bin/env bash
# A/B harness for the Kimi-K3 DSpark draft arms.
#
#   ./tools/dflash_ab.sh <arm> <workload>
#     arm      : radixark | inferact | nospec
#     workload : smoke | lossless | accept | swebench | perf
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

start_server() {
  stop_server
  # Stale compile cache causes silent failures after code changes (CLAUDE.md).
  rm -rf "$HOME/.cache/atom"/* 2>/dev/null || true
  docker run -d --name "$CNAME" \
    --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined \
    --group-add video --ipc=host --shm-size 64G --network host \
    -v "$REPO":/work -w /work -v "$HF":"$HF":ro -v /tmp:/tmp \
    -e AITER_LOG_LEVEL=WARNING -e PYTHONNOUSERSITE=1 -e PYTHONHASHSEED=0 \
    "$IMAGE" \
    python3 -m atom.entrypoints.openai_server \
      --model "$MODEL" \
      --served-model-name moonshotai/Kimi-K3 \
      --host 0.0.0.0 --server-port $PORT \
      -tp 8 \
      --kv_cache_dtype fp8 \
      --block-size 128 \
      --max-num-seqs 64 \
      --max-num-batched-tokens 8192 \
      --gpu-memory-utilization 0.9 \
      --no-enable_prefix_caching \
      --trust-remote-code \
      --tool-call-parser kimi_k3 \
      $SPEC_ARGS \
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

case "$WORKLOAD" in
  smoke|lossless|accept|swebench|perf)
    start_server
    wait_ready
    python3 "$REPO/tools/dflash_ab_client.py" \
      --arm "$ARM" --workload "$WORKLOAD" --port "$PORT" --outdir "$RUNDIR"
    ;;
  *) echo "unknown workload: $WORKLOAD" >&2; exit 2 ;;
esac
