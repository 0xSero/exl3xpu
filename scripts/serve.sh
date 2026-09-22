#!/usr/bin/env bash
# Launch vLLM (XPU) serving the EXL3 checkpoint on one B70.
# Usage: GPU=0 PORT=8100 scripts/serve.sh [extra vllm args...]
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -euo pipefail
MODEL="${MODEL:-/models/turboderp-Qwen3.8-27B-exl3-4.00bpw}"
GPU="${GPU:-0}"
PORT="${PORT:-8100}"
MAX_SEQS="${MAX_SEQS:-64}"
export ZE_AFFINITY_MASK="$GPU"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_XPU_ENABLE_XPU_GRAPH="${XPU_GRAPH:-1}"
CAPTURE="${CAPTURE:-[1,2,4,8,16,24,32,40,48,56,64]}"
exec vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b-exl3 \
  --language-model-only \
  --dtype float16 \
  --port "$PORT" --host 0.0.0.0 \
  --max-model-len "${MAX_LEN:-32768}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --max-num-seqs "$MAX_SEQS" \
  --max-num-batched-tokens "${MAX_BATCHED:-8192}" \
  --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":$CAPTURE}" \
  --trust-remote-code \
  "$@"
