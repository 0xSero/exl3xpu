#!/usr/bin/env bash
# Wait until run RUN is serving (prints READY) or died (prints DIED + log tail). Bounded by $2 seconds (default 900).
RUN=$1; T=${2:-900}; L=~/sglb70/runs/$RUN/server.log; end=$((SECONDS+T))
while [ $SECONDS -lt $end ]; do
  if grep -q "fired up" $L 2>/dev/null; then echo READY; exit 0; fi
  if [ -f ~/sglb70/logs/$RUN.exit ]; then echo "DIED exit=$(cat ~/sglb70/logs/$RUN.exit)"; grep -vE "Warning|warn" $L | tail -30 | cut -c1-300; exit 1; fi
  sleep 5
done
echo TIMEOUT; tail -5 $L | cut -c1-300; exit 2
