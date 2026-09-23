#!/usr/bin/env bash
# Stop the vLLM server listening on a given port (never matches this script's own command line).
port=$1
for p in $(pgrep -f "vllm serve"); do
  [ "$p" = "$$" ] || [ "$p" = "$PPID" ] && continue
  tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -qE -- "--port $port( |$)" && kill "$p" && echo "stopped $p"
done
