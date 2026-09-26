#!/usr/bin/env bash
# Dev container for EXL3-in-SGLang on ONE B70 (c3:00.0 = renderD132/card5 on omarchy). Maps only that card.
# Usage: scripts/sglb70_container.sh [name] [image]
set -euo pipefail
NAME=${1:-sglb70-dev}
IMG=${2:-lmsysorg/sglang:v0.5.20-xpu}
PCI=${SGLB70_PCI:-0000:84:00.0}
R=$(basename "$(readlink -f /dev/dri/by-path/pci-$PCI-render)")
C=$(basename "$(readlink -f /dev/dri/by-path/pci-$PCI-card)")
BYP=$HOME/sglb70/byp
rm -rf "$BYP"; mkdir -p "$BYP"
ln -s ../$R "$BYP/pci-$PCI-render"; ln -s ../$C "$BYP/pci-$PCI-card"
echo "card $PCI -> $R $C"
docker run -d --name "$NAME" --device /dev/dri/$R --device /dev/dri/$C \
  --group-add "$(getent group render | cut -d: -f3)" --group-add "$(getent group video | cut -d: -f3)" \
  -v "$BYP":/dev/dri/by-path:ro -v "$HOME/sglb70":/w -v "$HOME/models":/models:ro \
  --shm-size 32g --network host --ipc host -e HF_HUB_OFFLINE=1 \
  --entrypoint sleep "$IMG" infinity
