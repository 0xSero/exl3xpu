#!/usr/bin/env bash
# Launch vLLM (XPU) serving the EXL3 checkpoint on one or more B70s.
# Usage: GPU=0 PORT=8100 scripts/serve.sh [extra vllm args...]
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -euo pipefail
MODEL="${MODEL:-/models/turboderp-Qwen3.8-27B-exl3-4.00bpw}"
GPU="${GPU:-0}"
PORT="${PORT:-8100}"
export ZE_AFFINITY_MASK="$GPU"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_LLM_ALLREDUCE="${TORCH_LLM_ALLREDUCE:-1}"
exec vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b-exl3 \
  --language-model-only \
  --dtype float16 \
  --port "$PORT" --host 0.0.0.0 \
  --max-model-len "${MAX_LEN:-32768}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --max-num-seqs "${MAX_SEQS:-64}" \
  --max-num-batched-tokens "${MAX_BATCHED:-8192}" \
  --trust-remote-code \
  "$@"
