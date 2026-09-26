#!/usr/bin/env bash
# Host side: run bench/sweep.py inside the container against the sglb70 server. Usage: sglb70_sweep.sh RUN LABEL [sweep args]
RUN=$1; LABEL=$2; shift 2
CARD=${CARD:-$(basename $(readlink -f /dev/dri/by-path/pci-0000:84:00.0-card))}
docker exec -e EXL3_GPU_CARDS=$CARD -e SGL_SERVER_LOG=/w/runs/$RUN/server.log -e TOKENIZER=/models/turboderp-Qwen3.8-27B-exl3-4.00bpw sglb70-dev \
  bash -c "cd /w/exl3xpu && python3 bench/sweep.py --base http://127.0.0.1:${PORT:-8210} --max-tokens 0 --label $LABEL --out /w/runs/$RUN/sweep.jsonl $(printf '%q ' "$@")" 2>&1 | grep -vE "^\{" | tail -40
