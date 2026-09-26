#!/usr/bin/env bash
# Run INSIDE the sglb70-* container: EXL3 in SGLang on the one mapped B70.
# Usage: sglb70_serve.sh RUN_NAME [extra sglang args...]   (env: MODEL, PORT, SGLB70_ENV="K=V K=V")
set -uo pipefail
RUN=$1; shift
MODEL=${MODEL:-/models/turboderp-Qwen3.8-27B-exl3-4.00bpw}
PORT=${PORT:-8210}
D=/w/runs/$RUN; mkdir -p "$D"
for kv in ${SGLB70_ENV:-}; do export "$kv"; done
export HF_HUB_OFFLINE=1 EXL3_MODEL_PATH=$MODEL
ARGS=(--model-path "$MODEL" --served-model-name qwen3.8-27b-exl3 --quantization exl3 --dtype float16
      --device xpu --host 127.0.0.1 --port "$PORT" --trust-remote-code
      --reasoning-parser qwen3 --tool-call-parser qwen3_coder "$@")
{
  echo "date: $(date -u +%FT%TZ)"; echo "argv: python3 -m sglang.launch_server ${ARGS[*]}"
  echo "sglang: $(python3 -c 'import sglang;print(sglang.__version__)')  torch: $(python3 -c 'import torch;print(torch.__version__)')"
  echo "image: ${SGLB70_IMAGE:-lmsysorg/sglang:v0.5.20-xpu}"; echo "device: $(ls /dev/dri/by-path)"
  echo "exl3xpu: $(cd /w/exl3xpu && md5sum exl3xpu/_C.so exl3xpu/sglang_plugin.py | tr '\n' ' ')"
  echo "env:"; env | grep -E '^(EXL3_|SGLANG_|SYCL_|ZE_|UR_|ONEAPI_|IGC_)' | sort
} > "$D/serve_cmd.txt"
exec python3 -m sglang.launch_server "${ARGS[@]}" > "$D/server.log" 2>&1
