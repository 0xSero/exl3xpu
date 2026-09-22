# Progress log (see GOAL.md)

All numbers: one B70, bench/sweep.py, cold unique prose prompts, thinking off, temp 0, raw rows in bench/results.jsonl.

## 2026-09-22

| step | decode C1 | C8 agg | C16 agg | C32 agg | C64 agg | prefill 4K | kept |
|---|---|---|---|---|---|---|---|
| Triton v1, eager | 7.3 | 56 | 102 | – | – | 1008 | baseline |
| ESIMD GEMV (dp4a), eager | 14.4 | 100 | – | – | – | – | yes |
| + XPU decode graphs (1-8) | 28.4 | 102 | – | – | – | – | yes |
| + DPAS kernel M<=64, blocked xh, graphs 1-64 | 28.5 | 152 | 239 | 391 | 411 | 1019 | yes |

Gate A1 (bit-exact weights): quick run PASS on all shape classes incl. 6-bit lm_head (reconstruct, vector,
DPAS). Exact fp16 codebook costs ~4% at M=1 vs the unrounded variant; kept (bit-exactness is a gate).

llama.cpp baseline: pending (pinned GGUF + MTP draft downloading; local GGUF failed the image's SHA pin).

## Next
- llama.cpp baseline sweep (1 GPU; then 2 GPU split + MTP)
- prefill: ESIMD reconstruct is in; fuse Hadamards into reconstruct (original-basis W) and bigger chunks
- DP=2 across both B70s
- speculative decoding (DSpark draft), lm_head speed, fp8 KV
- Gate A3: logits vs exllamav3 on the 3090s (KL + top-1)
