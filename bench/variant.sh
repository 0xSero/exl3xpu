#!/usr/bin/env bash
# Run one serving variant end to end: restart vLLM with overrides, wait, sweep, log.
# Usage: bench/variant.sh LABEL "SWEEP ARGS" [--set k=v ...]
# e.g.   bench/variant.sh mtp5-dv512 "--decode-c 1 --classes prose,code --skip-prefill" --set vllm.speculative_config.num_speculative_tokens=5
cd "$(dirname "$0")/.."
LABEL=$1; SWEEP=$2; shift 2
MODEL_PATH=${MODEL_PATH:-/models/turboderp-Qwen3.8-27B-exl3-4.00bpw}
CFG=${CFG:-models/qwen3.8-27b-exl3-4.00bpw}
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
scripts/stop.sh >/dev/null
: > logs/serve${GPU:-0}.log
python3 scripts/serve.py $CFG --gpu ${GPU:-0} --port ${PORT:-8100} --model-path $MODEL_PATH "$@" > logs/serve${GPU:-0}.log 2>&1 &
for i in $(seq 1 180); do
  grep -qE "startup complete" logs/serve${GPU:-0}.log && break
  grep -qE "initialization failed|error: |Traceback" logs/serve${GPU:-0}.log && { echo "[$LABEL] LAUNCH FAILED"; grep -E "Error|error" logs/serve${GPU:-0}.log | tail -3; exit 1; }
  sleep 10
done
python3 bench/sweep.py --base http://localhost:${PORT:-8100} $SWEEP --label "$LABEL" --out bench/results.jsonl 2>&1 | tail -12
echo "[$LABEL] $(grep 'SpecDecoding metrics' logs/serve${GPU:-0}.log | tail -1 | grep -oE 'Mean acceptance length: [0-9.]+|Per-position acceptance rate: [0-9., ]+')"
grep -E "draft head uses" logs/serve${GPU:-0}.log | tail -1 | cut -c1-200
