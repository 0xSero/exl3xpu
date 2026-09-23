#!/usr/bin/env bash
# Stop running variant/sweep scripts and vLLM without matching this script's own command line.
self=$$
for pid in $(pgrep -f "bench/variant.sh|bench/sweep.py"); do
  [ "$pid" != "$self" ] && [ "$pid" != "$PPID" ] && kill "$pid" 2>/dev/null
done
"$(dirname "$0")/../scripts/stop.sh"
