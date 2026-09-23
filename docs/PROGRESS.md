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

## 2026-09-23: goal v2 (one B70, MTP, 256K, vision)

Config: models/qwen3.8-27b-exl3-4.00bpw/model.yaml (MTP k=3, fp8 KV, max_model_len 262144, vision on).
Findings: the EXL3 repo already ships the MTP head (EXL3 4-bit, missing from quantization_config.json's
tensor_storage) and the vision tower (bf16). No grafting needed; plugin detects EXL3 modules from `.trellis`
entries in the weight index.

Harness bug fixed: with MTP, vLLM streams several tokens per SSE chunk; sweep.py counted chunks. Now counts
tokens (vLLM continuous_usage_stats; tokenizer fallback for llama.cpp). The earlier llama.cpp TP2+MTP cells
were undercounted the same way and must be re-measured.

| step | C1 prose | C1 code | notes | kept |
|---|---|---|---|---|
| MTP k=3, full-vocab draft head | 37.9 | 57.5 | C2 70.5 / C8 221.5 prose aggregate; accept len 3.72 | base |
| + pruned draft head (512 of 1940 vocab blocks, 98.6% corpus coverage) | **48.2** | 65.7 | accept len 3.71 (unchanged) | yes |
| k=4 (+ pruned head) | 44.4 | 72.8 | accept len 4.32; prose is the headline | no |
| k=5 | – | – | 256K KV does not fit (9.7 GiB needed, 9.5 free) | no |

Kernel steps (graph-captured linear budget, all 257 fused linears):
| step | M=1 | M=4 | kept |
|---|---|---|---|
| baseline | 33.2 ms | 41.0 ms | |
| split-K target 1024/2048/8192/16384 | 32.7 best | 40.7 | 1024 kept (marginal) |
| dp4a 0x6400 trick, loaded prev-words, fp16 partial dots (ISA 378 -> 300 instr/row) | 32.5 | 45.1 | yes for M=1 (bit-exact) |
| software-pipelined loads | 32.2 | 70.0 (spills) | no |
| DPAS with zero-padded A (M=2..8 flat ~36 ms) | – | 36.3 | yes; vector for M<=2 |
| K=6 tile pairing (32-lane decode), NT per case | lm_head 3.85 -> 3.46 ms | | yes |
Diagnosis: M=1 GEMV is co-limited by issue (~22 ms) and DRAM (~23.5 ms, 531 GB/s ceiling measured with decode
stripped); overhead outside the GEMV (Hadamards, gaps under graphs) is only 1.2 ms/step.

Prefill (linears of one 8192-token chunk):
| step | total | ceiling | kept |
|---|---|---|---|
| Triton Hadamards | 7.71 s | 1063 tok/s | |
| ESIMD row-major Hadamards | 4.31 s | 1902 tok/s | yes (GEMM ~140 TFLOPS, had 11%) |
