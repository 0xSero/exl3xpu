#!/usr/bin/env bash
# Launch SGLang with a named config (scripts/cfg/<cfg>.args + .env) as run <run>; wait until ready. Host side.
CFG=$1; RUN=$2; S=~/sglb70/exl3xpu/scripts
SGLB70_ENV="$(cat $S/cfg/$CFG.env)" bash $S/sglb70_launch.sh $RUN $(cat $S/cfg/$CFG.args) && bash $S/sglb70_wait.sh $RUN 1200
