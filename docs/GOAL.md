# Standing goal (v2, set 2026-09-22): one B70, MTP, 256K context, vision

Model: turboderp/Qwen3.8-27B-exl3 @ 4.00bpw, plus the BF16 MTP head and vision tower from
Qwen/Qwen3.8-27B @1d4bf0f2 (the EXL3 repo ships neither). Engine: vLLM 0.26.1 XPU + exl3xpu.
One Intel Arc Pro B70 (32 GB). Loop does not stop until every gate below holds.

## Gate T (targets, one card, one config)
T1. decode C=1 >= 50 tok/s per stream (prose, thinking off, temp 0, MTP on, cold unique prompt, 0 context)
T2. cold prefill >= 1000 tok/s at every served context: 4K, 32K, 128K, 256K (unique prompts, TTFT-based)
T3. max_model_len 262144 served on the one card, KV_FULL never fires at C=1 with a 256K prompt
T4. C=2 and C=8 aggregate decode reported (and must not regress below the no-MTP DPAS numbers: 152 at C8)
T5. vision: an image request is answered correctly (describe a synthetic test image with known content)
T6. MTP on and verified: acceptance length reported, greedy output identical with and without MTP

## Gate A (still holds)
A1. EXL3 weights bit-exact on every kernel path (tests/test_bitexact_xpu.py) after any kernel change.
A3. logits vs exllamav3 on 3090 (tests/gateA3).

## Budget notes (per card)
weights 14.9 GiB EXL3 + ~0.85 GiB MTP bf16 + ~0.9 GiB vision bf16; KV for 256K: 16 attn layers x 2 x 4 kv
heads x 256 x 256K = 16.8 GiB fp16 -> needs fp8 KV (8.4 GiB); GDN state ~150 MiB per sequence.

## Levers
- M=1..4 GEMV bandwidth (now 415-630 GB/s per layer, ~33 ms/token of linears) -> ~580 GB/s everywhere
- lm_head (6-bit, 0.95 GB, ~3.6 ms per call; the MTP draft pays it per draft token)
- MTP draft length, draft precision (bf16 vs EXL3-quantized head), draft lm_head cost
- fewer kernels per linear (fuse had_in into GEMM, fuse gate/up + activation), XPU graphs coverage
- prefill: fused-Hadamard reconstruct (original-basis W), oneDNN GEMM efficiency, chunk size, attention kernel at long context
- fp8 KV cache, memory fraction
