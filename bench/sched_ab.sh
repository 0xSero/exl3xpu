#!/usr/bin/env bash
# HOST script: restart our server in exl3dev on B70 #1 (--gpu 1) with overrides, then cold prefill + interleave.
# usage: sched_ab.sh LABEL [--set k=v ...]
LABEL=$1; shift
docker exec exl3dev bash -c 'for p in $(pgrep -f "^python3 scripts/serv[e]") $(pgrep -f "^/opt/venv/bin/python3 /opt/venv/bin/vll[m] serve") $(pgrep -f "^VLLM::"); do kill $p; done'; sleep 15
docker exec exl3dev truncate -s 0 /w/logs/serve1.log
docker exec -d -w /w exl3dev bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; python3 scripts/serve.py models/qwen3.8-27b-exl3-4.00bpw --gpu 1 --port 8101 --model-path /models/turboderp-Qwen3.8-27B-exl3-4.00bpw $* > logs/serve1.log 2>&1"
for i in $(seq 1 120); do grep -q "startup complete" ~/intel-arc-exl3/logs/serve1.log && break; grep -qE "Traceback|initialization failed" ~/intel-arc-exl3/logs/serve1.log && { echo "[$LABEL] LAUNCH FAILED"; exit 1; }; sleep 10; done
echo "[$LABEL] up: $(grep -m1 -oE "max_num_batched_tokens': [0-9]+|long_prefill_token_threshold': [0-9]+" ~/intel-arc-exl3/logs/serve1.log | tr '\n' ' ') $(grep -m1 -oE 'enable_prefix_caching=[A-Za-z]+' ~/intel-arc-exl3/logs/serve1.log)"
docker exec -w /w exl3dev python3 bench/sweep.py --base http://localhost:8101 --skip-decode --prefill-c 1 --prefill-ctx 4096,32768,131072 --corpus real --thinking --label $LABEL-prefill --out bench/results.jsonl 2>&1 | grep -E "^prefill"
docker exec -w /w exl3dev python3 bench/interleave.py --base http://localhost:8101 --streams 4 --interval 20 --sizes 1024,8192,32768 --warm 30 --window 150 --label $LABEL-il --out bench/results.jsonl 2>&1 | grep -E "background|arrivals  |largest" | cut -c1-200
echo "[$LABEL] done"
