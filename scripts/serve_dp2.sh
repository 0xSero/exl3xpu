#!/usr/bin/env bash
# Data-parallel: one vLLM replica per B70 (ports 8100/8101) + least-outstanding proxy on :8000.
cd "$(dirname "$0")/.."
mkdir -p logs
GPU=0 PORT=8100 scripts/serve.sh "$@" > logs/serve0.log 2>&1 &
GPU=1 PORT=8101 scripts/serve.sh "$@" > logs/serve1.log 2>&1 &
python3 scripts/lb.py --port 8000 --backends http://localhost:8100,http://localhost:8101 > logs/lb.log 2>&1 &
wait
