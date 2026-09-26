#!/usr/bin/env bash
# Host side: gate panel against the running sglb70 server. Usage: sglb70_panel.sh RUN [LABEL]
# Realistic corpus (Gutenberg / HumanEval), t=0.7 top-p 0.95, no output cap; thinking on (C1,2,4,8,16) and off (C1,2,8);
# cold prefill 4K/32K/128K (2 waves, first discarded). Rows -> ~/sglb70/runs/RUN/sweep.jsonl, log -> panel.log
RUN=$1; L=${2:-$1}; S=~/sglb70/exl3xpu/scripts/sglb70_sweep.sh
{
echo "== panel $RUN $(date -u +%FT%TZ)"
bash $S $RUN $L-think --decode-c 1,2,4,8,16 --classes prose,code --thinking --corpus real --temperature 0.7 --skip-prefill --warm 15 --window 45
bash $S $RUN $L-nothink --decode-c 1,2,8 --classes prose,code --corpus real --temperature 0.7 --skip-prefill --warm 15 --window 45
bash $S $RUN $L-prefill --skip-decode --prefill-c 1 --prefill-ctx 4096,32768,131072 --prefill-waves 2
echo "== done $(date -u +%FT%TZ)"
} 2>&1 | tee -a ~/sglb70/logs/$RUN.panel.log
