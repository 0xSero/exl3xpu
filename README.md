# exl3xpu

![Qwen3.8-27B EXL3 on one Intel Arc Pro B70](docs/banner/banner.png)

EXL3 ([exllamav3](https://github.com/turboderp-org/exllamav3) trellis quantization) inference on Intel Arc
Battlemage GPUs, as a vLLM plugin. Native ESIMD kernels decode the trellis bit-exactly and run the GEMMs
on the Xe2 vector and XMX units; vLLM supplies scheduling, paged KV cache, the GDN/attention kernels and the
OpenAI API.

Tested on Intel Arc Pro B70 (BMG-G31, 32 GB), vLLM 0.26.1 XPU
(`intel/llm-scaler-vllm:0.26.0-b2@sha256:52218ad85513ab6686d4c090c83c2bd8c5b02423c63aa4dabd41837fe641fe3b`),
host kernel 7.1.8 (xe), compute-runtime 26.31.39395.13, IGC 2.40.13, Level Zero loader 1.32.0.

## Results: Qwen3.8-27B EXL3 4.00bpw, one B70

Config [`models/qwen3.8-27b-exl3-4.00bpw/model.yaml`](models/qwen3.8-27b-exl3-4.00bpw/model.yaml):
MTP speculative decoding k=3 (the EXL3 MTP head shipped in the checkpoint, draft lm_head pruned to 512 vocab
blocks), fp8 KV cache (272,570 tokens in 1600-token blocks), max context 262,144, 16 sequences, image (4/prompt) and
video (1/prompt) input. Model revision `113cf7ab958054860e43fb7f3063b1af19171095` (branch `4.00bpw`).

Decode with thinking on, aggregate tok/s. C16 is KV-bound (about 10 of 16 long-reasoning streams fit the fp8 pool). Cold unique tokenizer-sized prompts, greedy, no output
cap, 60-90 s sustained windows (`bench/sweep.py`); raw rows in `bench/results/2026-09-23.jsonl`.

| C | prose (thinking on) | code (thinking on) |
|---|---|---|
| 1 | 76.6 | 63.3 |
| 2 | 136.2 | 115.3 |
| 4 | 248.7 | 196.3 |
| 8 | 362.6 | 294.3 |
| 16 | 409.9 | 344.2 |

Cold prefill, one request: 4K **1589**, 32K **1497**, 128K **1049**, 254K **763** tok/s.

Same card, tuned llama.cpp SYCL Q4_K_M (`qwen38-q4km-arcb70-llamacpp-tp1`): C1 25.0, C8 56.8, C16 56.0
aggregate; prefill 4K 999, 32K 629 tok/s.

## Correctness

- Dequantized weights are **bit-identical** to exllamav3's `reconstruct()` for all 401 EXL3 tensors of the
  checkpoint (25.6G weights) on every kernel path: `tests/test_bitexact_xpu.py`. The pure-PyTorch
  reference (`exl3xpu/ref.py`) is validated bit-exact against exllamav3's CUDA kernels on an RTX 3090
  (`tests/oracle_cuda.py`).
- End-to-end logits vs exllamav3 on a 3090 (teacher-forced, sealed 64x256 panel): `tests/gateA3/`.

## Layout

```
exl3xpu/          core, model-agnostic
  vllm_plugin.py    `exl3` quantization config + linear method (fused shards, lm_head), vLLM entry point
  ops.py            torch custom op: dispatch M<=128 fused GEMM, else reconstruct + oneDNN GEMM
  ref.py            bit-exact PyTorch reference decoder (the spec)
  triton_kernels.py portable fallback kernels
csrc/             ESIMD kernels (Hadamard in/out, dp4a GEMV, DPAS GEMM, reconstruct) + torch bindings
models/<id>/      one directory per served model: model.yaml (serving config), recipe.json (measured)
scripts/          build_ext.sh, serve.py (config -> vllm serve), lb.py (data-parallel proxy), stop.sh
bench/sweep.py    saturation sweep (decode per-stream/aggregate, cold prefill, GPU busy flags)
tests/            oracle vs CUDA, bit-exactness on XPU, kernel timing, Gate A3 logits panel
docs/             DESIGN.md (format + kernels), GOAL.md, PROGRESS.md (tuning log)
```

## Run

```bash
docker build -f docker/Dockerfile -t exl3xpu .
hf download turboderp/Qwen3.8-27B-exl3 --revision 4.00bpw --local-dir $MODELS/turboderp-Qwen3.8-27B-exl3-4.00bpw

# one replica on GPU 0 (port 8100)
docker run --rm --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro --group-add render \
  --shm-size 32g --network host -v $MODELS:/models exl3xpu models/qwen3.8-27b-exl3-4.00bpw --gpu 0 \
  --model-path /models/turboderp-Qwen3.8-27B-exl3-4.00bpw

# both GPUs, one replica each, least-outstanding proxy on :8000
docker run ... exl3xpu models/qwen3.8-27b-exl3-4.00bpw --dp --model-path /models/turboderp-Qwen3.8-27B-exl3-4.00bpw
```

`/dev/dri/by-path` must be mounted: oneCCL enumerates devices through it and the engine fails to start
without it.

Without Docker (inside any vLLM XPU environment with oneAPI 2025.3): `scripts/build_ext.sh && pip install -e .`,
then `python3 scripts/serve.py models/qwen3.8-27b-exl3-4.00bpw --gpu 0`. Add `--print` to see the exact
`vllm serve` command.

## Adding a model

Create `models/<id>/model.yaml` (copy the Qwen one): source repo + revision, vLLM args, env, parallel
layout. The core needs no changes as long as the checkpoint is EXL3 with the `mul1` codebook at 4 or 6 bpw
(build with `EXL3_FLAGS=-DEXL3_ALL_CODEBOOKS` for 2/3/5 bpw and the mcg/3INST codebooks), all linear dims
are multiples of 128, and vLLM already implements the architecture. Tensor parallelism is not supported yet;
use data parallel.

## Status

Candidate recipe (`models/qwen3.8-27b-exl3-4.00bpw/recipe.json`). Weights bit-exact vs exllamav3 on every
kernel path; logits vs exllamav3 on a 3090: top-1 99.63%, KL 9.8e-5. Vision, video and a 128K needle test
pass. Open items: C16 does not scale past C8, 256K prefill is bound by fp8 attention, repeated waves. Log in
`docs/PROGRESS.md`.
