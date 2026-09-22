# Standing goal: EXL3 on 2x Intel Arc Pro B70, bit-exact, faster than llama.cpp

Model: turboderp/Qwen3.8-27B-exl3 @ 4.00bpw (omarchy:~/models/turboderp-Qwen3.8-27B-exl3-4.00bpw).
Engine: vLLM 0.26.1 (intel/llm-scaler-vllm:0.26.0-b2) + exl3xpu plugin (this repo).

## Gate A: bit-exact EXL3
A1. Dequantized weights bit-identical to exllamav3 `reconstruct()` (CUDA, 3090) for every quantized
    tensor in the checkpoint, on every kernel path (vector M<=4, DPAS, reconstruct/prefill).
    Test: tests/oracle_cuda.py (ref vs CUDA) + tests/test_bitexact_xpu.py (every XPU path vs ref, all tensors).
A2. Layer outputs: XPU linear vs exllamav3 LinearEXL3 on identical fp16 input, rel err <= the CUDA
    kernel's own error vs an fp64 reference (accumulation-order noise only).
A3. End to end vs exllamav3 on a 3090 (same prompts, greedy): top-1 token agreement >= 99% over a
    sealed 64-prompt panel of 256 teacher-forced positions each, mean KL(exllamav3 || ours) <= 0.005 nats.
    (Bitwise-equal logits across different GPUs/accumulation orders is not physically achievable;
    this is the strongest meaningful end-to-end criterion.)

## Gate B: beat llama.cpp on the same B70s
Baseline: llama.cpp SYCL (image ghcr.io/0xsero/qwen38-b70-attested, Qwen3.8-27B GGUF Q4_K_M), best
config we can find for it (1 and 2 GPUs), measured with bench/sweep.py (cold unique prompts).
Win condition, on every cell of the panel:
B1. decode C=1 per-stream tok/s  > llama.cpp
B2. decode aggregate tok/s at C = 8, 16, 32, 64 > llama.cpp's best at any concurrency
B3. cold prefill tok/s at 4K and 32K > llama.cpp
All cells: no REQ_FAIL / GPU_IDLE / KV_FULL / CACHE_HIT flags, correct output (Gate A3 holds for the
exact config measured).

## Work loop (repeat until both gates hold, then keep optimising the weakest cell)
1. Check both B70s are doing useful work; nothing stale holding VRAM (scripts/stop.sh).
2. Pick the weakest cell vs llama.cpp -> profile -> change one thing -> re-measure (paired, same boot).
3. Re-run Gate A tests after any kernel change. Never keep a faster kernel that breaks A1.
4. Log every kept/rejected step in PROGRESS.md with numbers; commit.

## Levers not yet pulled
- DP=2 (both B70s), TP=2
- Speculative decoding (Qwen3.8-27B-DSpark draft in ~/models; exl3 repo has no MTP head)
- fp8 KV cache, prefill chunk size, graph capture sizes, memory fraction
- ESIMD reconstruct + fused Hadamard for prefill; lm_head (6-bit) kernel; fusing had_in into GEMM
