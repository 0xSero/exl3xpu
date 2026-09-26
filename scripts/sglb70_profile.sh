#!/usr/bin/env bash
# Decode-stage profile at concurrency C on the running server: RUN C NAME
RUN=$1; C=$2; N=$3
tmux kill-session -t sglb70-bg 2>/dev/null
tmux new -d -s sglb70-bg "PORT=8210 bash ~/sglb70/exl3xpu/scripts/sglb70_sweep.sh $RUN prof$C --decode-c $C --classes prose --corpus real --temperature 0.7 --skip-prefill --warm 10 --window 100 > ~/sglb70/logs/bg_$N.log 2>&1"
sleep 50
curl -s -m 60 -X POST http://127.0.0.1:8210/start_profile -H "Content-Type: application/json" \
  -d "{\"output_dir\":\"/w/prof/$N\",\"num_steps\":15,\"activities\":[\"CPU\",\"XPU\"],\"profile_by_stage\":true,\"profile_stages\":[\"decode\"]}"; echo
for i in $(seq 60); do docker exec sglb70-dev bash -c "ls /w/prof/$N/*.gz 2>/dev/null | head -1" | grep -q gz && break; sleep 3; done
sleep 5
docker exec sglb70-dev bash -c "cd /w/prof/$N && ls; f=\$(ls -S *.gz | head -1); python3 /w/exl3xpu/bench/trace_kernels.py \$f 15 | head -32"
