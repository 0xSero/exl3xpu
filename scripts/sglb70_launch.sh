#!/usr/bin/env bash
# Host side: start one SGLang server in tmux session sglb70-srv inside the sglb70-dev container.
# Usage: sglb70_launch.sh RUN [sglang args...]   (env passthrough: SGLB70_ENV="K=V ...")
RUN=$1; shift
tmux kill-session -t sglb70-srv 2>/dev/null
# the card re-enumerates (new renderD/card numbers) after every PCIe link drop: recreate the container if stale
want=$(basename "$(readlink -f /dev/dri/by-path/pci-${SGLB70_PCI:-0000:84:00.0}-render)")
have=$(docker exec sglb70-dev ls /dev/dri 2>/dev/null | grep renderD)
if [ "$want" != "$have" ]; then
  echo "card node changed ($have -> $want): recreating sglb70-dev"
  docker rm -f sglb70-dev >/dev/null 2>&1; bash ~/sglb70/exl3xpu/scripts/sglb70_container.sh >/dev/null
  docker exec sglb70-dev bash -c "cd /w/exl3xpu && pip install --no-deps --no-build-isolation -e . >/dev/null 2>&1; pip install -q --no-deps onednn-devel==2026.0.0 onednn==2026.0.0 >/dev/null 2>&1"
fi
# the server runs inside the container: killing the tmux client does not stop it
docker exec sglb70-dev bash -c 'pkill -f "[s]glang.launch_server" ; for i in $(seq 90); do pgrep -f "[s]glang.launch_server" >/dev/null || exit 0; sleep 1; done; pkill -9 -f "[s]glang"' 2>/dev/null
mkdir -p ~/sglb70/logs
rm -f ~/sglb70/logs/$RUN.exit
printf '%q ' "$@" > ~/sglb70/logs/$RUN.args
tmux new -d -s sglb70-srv "docker exec -e SGLB70_ENV=\"${SGLB70_ENV:-}\" sglb70-dev bash /w/exl3xpu/scripts/sglb70_serve.sh $RUN $(printf '%q ' "$@"); echo \$? > ~/sglb70/logs/$RUN.exit"
echo "launched $RUN"
