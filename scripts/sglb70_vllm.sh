#!/usr/bin/env bash
# vLLM exl3xpu baseline on ONE B70 (default 84:00.0), same argv as the Omarchy engine, host port 8110.
# Usage: sglb70_vllm.sh [image]   env: SGLB70_PCI, VPORT, VENV="K=V ..." (extra -e)
set -euo pipefail
IMG=${1:-ghcr.io/0xsero/exl3xpu@sha256:21412bdd7535e9c653eeb3d099dce3bc79a83d440def2cd9556111c79c870fa8}
PCI=${SGLB70_PCI:-0000:84:00.0}; PORT=${VPORT:-8110}
R=$(basename "$(readlink -f /dev/dri/by-path/pci-$PCI-render)"); C=$(basename "$(readlink -f /dev/dri/by-path/pci-$PCI-card)")
BYP=$HOME/sglb70/byp-vllm; rm -rf "$BYP"; mkdir -p "$BYP"; ln -s ../$R "$BYP/pci-$PCI-render"; ln -s ../$C "$BYP/pci-$PCI-card"
docker rm -f sglb70-vllm >/dev/null 2>&1 || true
EX=(); for kv in ${VENV:-}; do EX+=(-e "$kv"); done
docker run -d --name sglb70-vllm --device /dev/dri/$R -v "$BYP":/dev/dri/by-path:ro --shm-size 32g --network host \
  -e HF_HUB_OFFLINE=1 "${EX[@]}" -v "$HOME/models/turboderp-Qwen3.8-27B-exl3-4.00bpw":/models:ro "$IMG" \
  models/qwen3.8-27b-exl3-4.00bpw --gpu 0 --port "$PORT" --model-path /models \
  -- --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 ${VEXTRA:-}
echo "vllm baseline on $PCI ($R) port $PORT"
