#!/usr/bin/env bash
# Run on the HOST (omarchy). Same workload against the live service (gateway :12434, API key file) and our optimal
# server (:8101), one after the other; benches run inside exl3dev, live draft acceptance is read from the engine.
set -u
ENGINE=omarchy-local-ai-qwen38-27b-exl3-4bpw-arcb70-vllm-exl3xpu-tp1-engine
X() { docker exec -w /w "$@"; }
LIVE=http://127.0.0.1:12434; OURS=http://localhost:8101; OUT=bench/results.jsonl
DEC="--decode-c 1,2,4,8,16 --classes prose,code --skip-prefill --thinking --corpus real --temperature 0.7 --max-tokens 16384 --warm 30 --window 120"
PRE="--skip-decode --prefill-c 1 --prefill-ctx 4096,32768,131072 --corpus real --thinking"
IL="--streams 4 --interval 20 --sizes 1024,8192,32768 --warm 30 --window 150"
M() { docker exec $ENGINE curl -s -m 5 localhost:8000/metrics 2>/dev/null | grep -E "^vllm:spec_decode_num_(accepted|draft)_tokens_total|^vllm:num_requests_running" ; }
echo "== live engine before"; M
echo "== live decode"
X -e EXL3_API_KEY_FILE=/tmp/gw.key exl3dev python3 bench/sweep.py --base $LIVE $DEC --label live-t07 --out $OUT --dump-text bench/texts-live.jsonl 2>&1 | tail -12
echo "== live engine after decode"; M
for side in live ours; do
  if [ $side = live ]; then B=$LIVE; E="-e EXL3_API_KEY_FILE=/tmp/gw.key"; else B=$OURS; E=""; fi
  echo "== $side prefill"
  X $E exl3dev python3 bench/sweep.py --base $B $PRE --label $side-prefill --out $OUT 2>&1 | tail -5
  echo "== $side interleave"
  X $E exl3dev python3 bench/interleave.py --base $B $IL --label $side-il --out $OUT 2>&1 | tail -8
done
echo "LIVE COMPARE DONE"
