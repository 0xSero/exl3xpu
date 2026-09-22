#!/usr/bin/env bash
# Kill every vLLM process in the dev container (API server, engine cores, resource trackers).
pkill -9 -f "vllm serve" 2>/dev/null
pkill -9 -f "VLLM::" 2>/dev/null
pkill -9 -f "multiprocessing.resource_tracker" 2>/dev/null
sleep 3
ps -eo pid,stat,cmd | grep -E "vllm|VLLM" | grep -v -E "grep|defunct" || echo "no vllm processes"
