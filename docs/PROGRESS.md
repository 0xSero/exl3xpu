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

### C1 decode investigation (2026-09-23, one B70, MTP k=3, deterministic prompt panel from here on)
The sweep now uses a fixed prompt sequence (`--seed panel-v1`) and records per-cell MTP acceptance
(vLLM /metrics deltas). Per-step time = acceptance_len / tok_s is the engine metric; prose acceptance on
this panel is ~2.4 (random prompts earlier gave 3.2-3.7, which made cross-run C1 comparisons noisy).

| step | C1 prose | ms/step | kept |
|---|---|---|---|
| baseline (pruned head, new kernels) | 47.4 @3.23* | – | |
| async scheduling | 45.8 | – | no (already default-on) |
| FULL_AND_PIECEWISE graphs (drafter captured), sizes<=32 | – | – | no: UR OUT_OF_RESOURCES |
| same, sizes [1,2,4,8] | 46.0 | – | no gain |
| C++ single-op linear (host 74 -> 34 us/call) | 47.4 | – | yes (cleaner, no loss) |
| GDN metadata sync patch (A/B on fixed panel) | 47.1 @2.41 vs 46.3 @2.38 | 51.1 vs 51.4 | yes (~1%) |
| DPAS K=4 contiguous-VNNI B build | M=4 linears 35.1 vs 36.2 ms | | yes (bit-exact) |
| DPAS natural-order decode / strided fp16 mad | 37.0 / 49.2 ms | | no |
| fused single-kernel linear (had_in + split-K last-arriver + had_out) | M=1 34.7 vs 32.5, M=4 45.2 vs 35.1 | | no (serialized tail; hads cost only ~1.2 ms/step under graphs) |
| lm_head verify on vector path (M=4) | 4.41 vs 3.57 ms DPAS | | no |
(*random-prompt acceptance)

Eager per-step GPU budget (C1, k=3): 50 ms = EXL3 GEMV/DPAS 40.8 + Hadamards 3.1 (eager launches; ~1.2 under
graphs) + GDN 1.6 + RMSNorm 1.5 + rest ~3. Same-GPU floors: vector M=1 32.5 ms full / 25.1 ms memory-only;
DPAS M=4 35.1 ms full / 33.5 ms without decode (structural B-build + dpas chain ~8 ms over the memory floor).

### Prefill, calibrated (2026-09-23)
Prompt generator fixed: random-word documents are ~3.5 tokens/word (not 1.9), so earlier "32K"/"128K" cells were
~58K/232K tokens. Prompts are now sized with the tokenizer (within 0.1%).

| ctx | prefill tok/s | TTFT | notes |
|---|---|---|---|
| 4K | 1665 | 2.5 s | |
| 32K | 1363 | 24 s | |
| 128K | 760 | 168 s | attention-bound (XPU FA2, head_dim 256) |
| 254K | 508 | 500 s | linears alone would allow ~1900 tok/s |

| step | result | kept |
|---|---|---|
| TRITON_ATTN backend | 32K: 43 tok/s (vs 1363 FA2) | no |
| DSpark block-7 draft (Qwen3DSparkModel arch fix) | C1 prose 30.3 @1.83, code 73.6 @4.48 (vs MTP 47.6 / 68.0) | no (prose headline) |
| draft vocab 256/384/512 blocks | C1 prose 47.7 / 47.2 / 47.6 | 512 kept (flat) |
| DPAS prev<<2 hoist | 35.23 vs 35.12 ms | no change |
GPU is PL2-throttled during decode: act 2600 MHz of 2800 (power1_cap 230 W, profile base); needs root to change.

### fp8-KV prefill attention at fp16 speed (2026-09-23)
Finding: vllm_xpu_kernels FA2 (head_dim 256, GQA 24:4) runs 69-74 TFLOPS on fp16 K/V but 39 TFLOPS on fp8 K/V
(bench/attn_bench.py). Server A/B at 128K: fp16 KV 1016 tok/s vs fp8 KV 760 tok/s. 256K only fits one B70 with fp8 KV.
Fix (exl3xpu/fp8kv_prefill.py + vllm_patches.patch_fp8kv_prefill): for single-sequence prefill chunks (>=64
queries, >=4K context, not under graph capture) gather the cached K/V in 32K-key blocks, dequantize to fp16,
run the fp16 kernel per block (causal only for the chunk's own keys), merge with log-sum-exp.
Correctness: tests/test_fp8kv_prefill.py (rel <= 4e-4 vs one full fp16 pass); needle recall at 128K: 3/3 depths.
Harness fix: prefill prompts are built (tokenizer-sized) before the clock starts.

| ctx | before (fp8 KV) | after | target |
|---|---|---|---|
| 4K | 1665 | 1612 | ok |
| 32K | 1363 | 1531 | ok |
| 128K | 760 | 1059 | ok |
| 254K | 508 | 763 | open: needs an attention kernel ~1.6x faster than XPU FA2 at head_dim 256 |

### Gate status (2026-09-23, one B70, MTP k=3 + pruned draft head, fp8 KV 256K, vision)
| gate | result |
|---|---|
| T1 C1 prose >= 50 tok/s | 47.6 (panel acceptance 2.4); code 68.0 |
| T2 prefill >= 1000 | 4K 1612, 32K 1531, 128K 1059 pass; 254K 763 |
| T3 256K on one card | pass (KV 265,888 tokens, 1.02x) + needle 3/3 at 128K |
| T4 C2 / C8 | prose 85 / 239, code 120 / 349 aggregate |
| T5 vision | pass |
| T6 MTP identity | 5/8 exact; divergence at a 0.016-nat top-2 tie (numerics, not a bug) |
| A1 bit-exact weights | pass |
| A3 logits vs exllamav3 (3090) | pass: top-1 99.63%, KL 9.8e-5 nats over 16,320 positions |

Rejected this round: TRITON_ATTN (43 tok/s at 32K); 256-GRF DPAS (NT=4 44.4 / NT=8 53.7 vs 35.1 ms);
work-group 4/16/32 and split-K 2048 (all worse than local 8 / 1024).
Bounds: XPU FA2 = 69 TFLOPS (fp16) at head_dim 256, the only FA version shipped. 254K prefill at 69 TFLOPS
attention + ~130 TFLOPS GEMMs caps near 800 tok/s; even a 100 TFLOPS attention kernel gives ~960.
GPU runs 2600 MHz of 2800 under decode (~47 W card power; 230 W cap; power profile 'base').

### Custom ESIMD flash-attention (user-requested, 2026-09-23)
csrc/fa_esimd.h: fp16, head_dim 256, GQA, causal (bottom-right), O + LSE; 2D block loads (K^T transposed = VNNI,
V VNNI-transformed), 8 query rows/thread, 256-GRF, online softmax base 2. tests/test_fa_esimd.py.
| version | 8K causal | 8K x 32K | vs FA2 | kept |
|---|---|---|---|---|
| v1 exact rescale | 48.2 TF | 37.7 TF | FA2 61.3 / 75.2 TF | kept as experiment, not wired |
| v2 lazy rescale (TAU=8) | 45.3 TF | 36.7 TF | no gain, rel err 1e-5 -> 4e-4 | reverted |
Diagnosis: at head_dim 256 an 8-row fp32 O accumulator is 8 KB of the 16 KB GRF, so a thread cannot hold more
query rows to reuse K/V tiles; ~8 FLOP/byte of L1 traffic per thread (x6 GQA heads re-reading the same tiles)
puts ~100+ TFLOPS beyond Xe2 L1 bandwidth. 1000 tok/s at 256K needs ~117 TFLOPS attention: not credible on one B70.

### Full sweep (2026-09-23, B70 #1 — B70 #0 was running the user's llama.cpp service)
Config: models/qwen3.8-27b-exl3-4.00bpw/model.yaml (MTP k=3, pruned draft head, fp8 KV 267,761 tokens, 256K
max len, max_num_seqs 16, image+video). Deterministic prompts (panel-v1), greedy. Aggregate tok/s (per-stream).

| C | prose off | code off | prose think | code think |
|---|---|---|---|---|
| 1 | 47.2 (47.0) | 66.7 (66.3) | 61.4 (56.6) | 48.5 (47.7) |
| 2 | 83.2 (42.7) | 119.8 (60.0) | 110.8 (54.1) | 82.7 (41.6) |
| 4 | 138.6 (35.5) | 200.4 (50.0) | 175.6 (42.5) | 144.1 (35.0) |
| 8 | 236.3 (30.5) | 345.3 (43.5) | 299.6 (37.9) | 234.1 (29.0) |
| 16 | 180.7 (18.2) | 264.9 (22.9) | 240.5 (23.2) | 184.4 (18.1) |
Prefill (cold): 4K 1589, 32K 1497, 128K 1049 tok/s. Vision + video tests pass. Audio: not supported by the model.

C16 < C8 diagnosed: a C16 verify is M=64 tokens; the MB=64 DPAS kernel (1 tile/thread) took 124 ms for all
linears vs 50.6 ms at M=32. Fixes measured (all linears, M=64): 2x MB=32 blocks 100 ms; MB=64 NT=1 256-GRF 105 ms;
MB=64 NT=2 256-GRF 87.3 ms (kept, bit-exact); MB=32 NT=4 256-GRF regressed M=32 (60.4 vs 50.7, not kept).

After the MB64 fix, C8/C16 re-measured on B70 #1: think off prose C8 238.4 / C16 233.1, code C8 344.1 / C16 332.5;
think on prose C8 318.1 / C16 294.7, code C8 239.8 / C16 REQ_FAIL. The C16 failure was a UR DEVICE_LOST in
graph replay at 12:09: the kernel log shows a job timeout on c3:00.0 at the same second another session's
llama.cpp engine was started on that card (our engine was at 91.5% VRAM). Repro of the cell on the idle B70 #0:
new MB64 (NT2, 256-GRF) 240.2 agg (23.3/stream, accept 2.27) with no errors; old MB64 197.3 (19.3/stream),
also clean. Not a kernel bug; MB64 fix confirmed +22% on the cell. C16 still == C8 on think-on code: next is
profiling the non-linear per-step cost at C16.

### Split-K sizing retune (2026-09-23), kept
C16 eager profile (25 verify steps): DPAS MB64 56 ms/step, GDN spec kernel 13.3, HadOut 9.2 (33 us/call vs
HadIn 5 us), lm_head 4.2, FA2 2.8, draft ~3.8; ~97 ms GPU vs ~150 ms wall per step (rest is host).
HadOut was slow because split-K was sized for 4096 threads: o_proj/out_proj/down at M=64 split K 26 ways
(~34 MB of fp32 partials per call). All linears, graph-captured (bench/linear_budget.py), 4096 -> 1024 threads:
M=1 36.1->32.5, M=2 38.5->33.5, M=4 41.4->35.7, M=16 49.3->39.1, M=32 54.4->50.6 ms (repeated, stable);
M=64 best at 2048: 87.7->77.0 ms. New defaults 1024 / 2048 (MB=64); env EXL3_TARGET_THREADS(_MB64).
Gate A1 PASS (401 tensors). Served, thinking on (agg tok/s, before -> after): C1 prose 61.4->68.6, code
48.5->56.9; C8 prose 318.1->326.0, code 239.8->262.5; C16 prose 294.7->305.5, code 240.2->249.6.
T1 now passes on both classes (code 56.9 >= 50).

### load_words without the p-1 reload (2026-09-23), kept; DPAS word prefetch, rejected
All linears (graph-captured, mean of 2 alternating reps), base -> prev-from-registers: M=1 32.5 -> 29.2,
M=4 35.4 -> 35.2, M=16 39.0 -> 37.8, M=64 76.8 -> 67.7 ms. Gate A1 PASS. Double-buffered trellis words in
DpasKernel (EXL3_DPAS_PREFETCH): M=16 39.0 -> 86.4 ms (register spill), M=64 76.8 -> 79.2: rejected.
Served (B70 #1, think off, agg tok/s, full-v2 -> prev-from-regs): C1 prose 53.9 -> 53.7, code 75.3 -> 75.9;
C8 prose 253.8 -> 265.4, code 371.4 -> 390.0; C16 prose 241.5 -> 262.7, code 349.0 -> 374.5. Kept.

### DPAS MB<=16 split-K target 1408 (2026-09-23)
Target-thread sweep for DPAS MB=8/16 (all linears, graph-captured, B70 #1): 1024: M=4 35.1, M=16 38.7 ms;
1280-1536 plateau: M=4 32.1-32.3, M=16 36.0-36.2; 1664+: 36.4 / 41.3 (cliff); 2048: 38.6 / 44.4. MB=32 stays
at 1024 (47.6 vs 49.6 at 1536); MB=64 flat 1408-2048 (67.9-68.6), stays 2048. Vector kernel at M=3/4
(VECMAX=4): 45.7 / 44.6 ms vs DPAS 35: rejected. New default MB<=16 1408 (env EXL3_TARGET_THREADS_MB16).
Gate A1 PASS. After: M=1 29.4, M=4 33.4, M=16 36.2, M=32 47.9, M=64 69.3 ms.
Served (B70 #1, agg tok/s, full-v2 -> MB<=16 1408 + prev-from-regs): think off C1 prose 53.9 -> 56.6, code
75.3 -> 79.2; C2 97.3 -> 104.2 / 137.6 -> 147.7; C4 170.7 -> 184.3 / 240.1 -> 254.7. Think on C1 prose
64.5 -> 70.3, code 54.3 -> 58.3; C2 122.5 -> 122.7 / 97.8 -> 106.2; C4 223.0 -> 237.5 / 173.0 -> 187.0. Kept.
Decode-cost probe (EXL3_DEBUG_NODECODE / NOSTATE builds): inconclusive, codegen changes dominate (M=1 29.2 ->
94.9 ms with decode removed); not a usable bound.
DPAS MB=8 tiles/thread (EXL3_DPAS8_NT), all linears, 2 reps: NT=4 (current) M=4 31.7-31.8, M=8 33.1-33.2 ms;
NT=2 41.9-42.1 / 42.7-43.1; NT=8 49.9-50.0 / 51.4-52.3. Rejected (NT=4 stays).
Async scheduling at C8/C16 (think off, confirmed enabled in the engine log): C8 prose 268.3 -> 268.9, code
389.1 -> 389.9; C16 prose 265.9 -> 261.1, code 378.0 -> 372.1. No gain: rejected. The ~48 ms/step gap vs the
eager-mode GPU sum is not overlappable host work; next C16 step is a graph-mode profile to re-derive it.
FULL_AND_PIECEWISE (drafter graphs), sizes [1,2,4,8,16,32,48,64]: UR OUT_OF_RESOURCES at capture_begin of the
2nd PIECEWISE size (Level Zero graph-resource limit; piecewise splits 64 layers into many subgraphs). Rejected.
DPAS MB=4 (repeat count 4, opt-in EXL3_DPAS_MB4): outputs bitwise identical to MB=8 (24 layer/M cases), all
linears 2 reps: M=3 31.9-32.0 -> 31.5-32.6, M=4 32.6-32.8 -> 32.0-32.1 ms (NT=4; NT=8 no better). Within noise:
M=4 is decode/memory-bound, not DPAS-bound. Rejected (left opt-in).
mul1 codebook as two 16x16 multiplies (EXL3_MUL16): Gate A1 PASS but slower everywhere (2 reps): M=1 29.4 ->
34.4, M=4 32.9 -> 39.4, M=16 36-38 -> 43.4, M=64 69.3 -> 78.0 ms. Rejected (left opt-in).

### 32K-context decode cells (2026-09-23, B70 #1, v3 build)
C1 x 32K: per-stream decode prose 50.3, code 71.8 tok/s (0-context: 56.6 / 79.2); aggregate 26.4 / 34.1 because
the 21-35 s prefill falls in the 60 s window. C4 x 32K (150 s warm-up): aggregate 3.4 prose / 1.6 code, TTFT
39-59 s: prefill-bound, every stream re-prefills a fresh 32K prompt per few hundred output tokens.
max_num_batched_tokens 2048 / 4096 vs 8192: prefill 4K 1568 / 1666 vs 1589, 32K 1407 / 1474 vs 1497; C4 x 32K
unchanged (2.8 / 13.2 and 2.7 / 1.5). Kept 8192. Cells recorded with PREFILL_IN_WINDOW / PREFILL_BOUND flags.

### C16 thinking-on concurrency (2026-09-23)
Engine log at C16 think-on: Running 10 / Waiting 6 at 91.5% KV; 8 running = 73%; 1 running = 9.2%. One stream
of ~2.1K tokens holds ~12 of ~131 blocks: the hybrid allocator uses one page size for all KV-cache groups; the
GDN state (3.1 MiB/layer, padded 4 MiB) forces 2048-token attention blocks under fp8 KV, and each group
allocates whole blocks. mamba_cache_mode default (drop align): C8/C16 prose 354.0/332.8, code 268.2/274.5
(vs 355.2/327.0, 275.5/270.9): noise, same 10 running. enable_prefix_caching=false: 346.6/336.9, 268.9/267.7,
same 10 running. Both rejected. Lever left: GDN state dtype (page size), needs a quality check (Gate A3).
GDN state dtype: vLLM "auto" already stores the temporal state in the model dtype (fp16; config.json asks fp32),
so the 3.1 MiB page is the fp16 size; lower would be fp8 state (quality risk). fp16 attention KV halves the block
but also halves pool tokens (256K no longer fits). Conclusion: C16 thinking-on stays allocator-bound (KV_BOUND);
C16 prose 327-337 vs C8 346-355.

### MTP draft length, thinking on (2026-09-23, v3 build, B70 #1)
agg tok/s C1 prose / C1 code / C4 prose / C4 code: k=2 61.9 / 55.3 / 219.5 / 187.6; k=3 70.3 / 58.3 / 237.5 /
187.0; k=4 75.8 / 59.5 / 215.0 / 159.4. k=4 wins C1 (+8% prose), k=3 wins C4 (k=4 -9%/-15%). Next: vLLM dynamic
SD schedule (num_speculative_tokens_per_batch_size) to use k=4 at batch 1 and k=3 above.
Dynamic SD schedule [[1,1,4],[2,64,3]]: forces PIECEWISE mixed capture -> UR OUT_OF_RESOURCES (12 sizes: fails
at 5/12; sizes [5,8,16,32,64]: also fails). Rejected on this driver. Fixed k stays 3 (k=4 is the C1-only option).
DSpark block-7 draft, thinking on (v3 build, max_model_len 32K as before): C1 prose 62.6 / code 22.0, C4 prose
106.9 / code 88.0 (MTP k=3: 70.3 / 58.3 / 237.5 / 187.0), accept 2.44. Rejected for thinking-on.

### K=6 planar trellis words (lm_head) (2026-09-23), kept
K=6 has D=3 words per period, so the per-value word gathers used stride-3 register regions (Xe regions only
stride by powers of two -> per-element moves, 2 per unrolled value). Words are now de-interleaved once per tile
into D contiguous planes (GEMV: contiguous selects; DPAS B-build: <P,1,8,P> instead of <P,3,8,6>). K=4 (D=1)
unchanged. Gate A1 PASS; outputs bitwise identical (24 layer/M cases). lm_head (1.27 GB, 6-bit): M=1 4.05 ->
1.72 ms (235 -> 555 GB/s), M=4 3.82 -> 2.00, M=16 4.03 -> 2.20, M=64 5.38 -> 4.13. All linears: M=1 29.4 ->
27.3, M=4 33.2 -> 31.7, M=16 36.2 -> 34.0-34.9 ms. (Opt-out: EXL3_NO_PLANAR.)
Fused single-kernel linear (exl3_set_fused 1) re-tested on the current build: M=1 27.2 -> 28.4, M=4 31.4 -> 45.9 ms
(small layers worse too: out_proj 366 -> 295 GB/s). Still rejected.
Per-layer split-K target at M=1 (TARGET 512/1024/2048/4096): 1024 is best for every layer type (out_proj 344/367/
340/284 GB/s, o_proj 339/360/337/279, down_proj 391/446/350/371). The small projections' ~360 GB/s is fixed
per-call overhead (Hadamard kernels + gaps, ~5-8 us on a ~26 us GEMV), worth ~1 ms/step in total.
Full (unpruned) MTP draft head after the K=6 planar lm_head (thinking on, agg): C1 prose 76.9 -> 73.8, code
61.4 -> 59.6; C4 245.7 -> 231.4, 199.4 -> 183.3; acceptance ~unchanged. Pruned 512-block head stays.
Draft vocab blocks after the planar lm_head (thinking on, agg C1 prose / C1 code / C4 prose / C4 code):
256: 73.5 / 58.9 / 242.2 / 187.1 (accept 2.75/2.29); 384: 76.7 / 59.2 / 243.6 / 197.2; 512: 76.9 / 61.4 / 245.7 /
199.4; full: 73.8 / 59.6 / 231.4 / 183.3. 512 stays.

### C1 thinking-on graph-mode profile (planar build, 145 steps, ~41 ms/step)
Outside the captured target graph, per step: target lm_head (M=4) 1.78 ms, draft lm_head 3 x 0.49 ms, draft-layer
linears ~1.3 ms, FA2/argmax/reduce ~0.8 ms: ~5.7 ms GPU. The graph (target linears ~30 ms + attention/GDN) is the
rest; ~146 small aten::copy_ per step cost ~6 ms CPU but overlap with GPU work. C1 is GPU-bound on the M=4
target linears (31.7 ms, 415 GB/s; decode-bound: MB=4 and vector-kernel variants measured no better).
LSC L1/L2 prefetch of trellis words EXL3_PF_DIST rows ahead (GEMV + DPAS, opt-in EXL3_L1_PF), all linears 2 reps:
dist 1: M=1 27.1-27.2, M=4 30.6-31.7, M=16 34.2-35.1; dist 2: 27.6-27.7 / 30.6-31.4 / 34.6-35.2; dist 4: 28.6 /
31.2-31.9 / 35.2-36.4; current: 27.2 / 31.3-31.5 / 34.6-34.7 ms. Noise or worse: rejected (left opt-in).

### ISA read of the K=4 DPAS loop; Gate A1 coverage fix (2026-09-23)
IGC dump of DpasKernel<4,2,8,4>: ~300 instructions per K-row for 32 decode groups of 32 values, i.e. ~9 SIMD32
per group: shl(:w) / shr(:ud) / or(:w) / mul (single native d x uw) / dp4a / mov (fp16 view copy) / mad(:hf), plus
~2 SIMD32-equivalent moves duplicating the 16-lane word region into both 32-lane halves. Decode is already
near the integer-ALU roofline (~25 ms of ALU work per step vs 27-31 ms measured for M<=4).
- EXL3_DECODE_INLINE (strided fp16 view straight into the mad): no change (27.2 / 31.4 ms), IGC still copies.
- EXL3_K4_HALVES (two 16-lane halves reading the region directly): timed -4% at M<=16, but its OUTPUTS WERE
  WRONG (abs err up to 120 at M=3..8). Rejected.
- Gate A1 only exercised vector M=4 and DPAS M=64, so the production MB=8/16/32 DPAS and vector M=1/2 paths were
  untested at the weight level. tests/test_bitexact_xpu.py now also checks vector M=1/2 and DPAS M=3/8/16/32 on a
  128-row sample per tensor; it catches the halves bug, and the production kernel passes (GATE_A1_PASS, 401 tensors).

### MB=64 DPAS kernel analysis (2026-09-23)
ISA (256-GRF, no GRF spills; 102 predicate-flag spill loads in the epilogue store chain): IGC hoists part of the next
tile's decode ahead of each 8-dpas burst. Per 16x16 tile per XVE: decode ~144 cycles (72 SIMD32 ALU), DPAS ~128
(8 x dpas.8x8), serial sum ~272; measured ~423 (67 ms, 52 TFLOPS). Every layer type is equally slow at M=64
(gate_up 199 GB/s with P=1, lm_head 232), so it is not split-K partials. Largest remaining term: each thread
re-loads the 64-row activation block (2 KB per K-row, 1 KB per tile) from L1 at ~8 B/clk/XVE (~128 cycles/tile).
Cheap fixes already measured and lost (MB=32 NT=4 halves A traffic but doubles decode: 100 ms at M=64).
Next large-M lever is a redesign: decode each weight tile once into SLM and share it across threads that own
different row blocks (decode once, A traffic per thread / MB), not a tuning change.
