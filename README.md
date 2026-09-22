# exl3xpu

EXL3 ([exllamav3](https://github.com/turboderp-org/exllamav3) trellis quantization) inference on Intel Arc
Battlemage GPUs, as a vLLM plugin. Native ESIMD kernels decode the trellis bit-exactly and run the GEMMs
on the Xe2 vector and XMX units; vLLM supplies scheduling, paged KV cache, the GDN/attention kernels and the
OpenAI API.

Tested on 2x Intel Arc Pro B70 (BMG-G31, 32 GB), vLLM 0.26.1 XPU (`intel/llm-scaler-vllm:0.26.0-b2`).

## Results: Qwen3.8-27B EXL3 4.00bpw, one B70

Cold unique prompts, sustained 40 s windows, no output cap (`bench/sweep.py`). Full rows:
[`models/qwen3.8-27b-exl3-4.00bpw/recipe.json`](models/qwen3.8-27b-exl3-4.00bpw/recipe.json).

| cell | exl3xpu (EXL3 4.0bpw) | llama.cpp SYCL (Q4_K_M, tuned image) |
|---|---|---|
| decode C=1, tok/s | **28.6** | 25.0 |
| decode C=8, aggregate tok/s | **152** | 57 |
| decode C=16, aggregate | **239** | 56 |
| decode C=32, aggregate | **391** | – (64 slots: device lost) |
| decode C=64, aggregate | **411** (24.7K tok/min) | – |
| prefill 4K cold, tok/s | **1019** | 999 |

The table is one card. `--dp` runs a second replica on the second card (aggregate sweep pending).

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
docker run --rm --device /dev/dri --group-add render --shm-size 32g --network host \
  -v $MODELS:/models exl3xpu models/qwen3.8-27b-exl3-4.00bpw --gpu 0 \
  --model-path /models/turboderp-Qwen3.8-27B-exl3-4.00bpw

# both GPUs, one replica each, least-outstanding proxy on :8000
docker run ... exl3xpu models/qwen3.8-27b-exl3-4.00bpw --dp --model-path /models/turboderp-Qwen3.8-27B-exl3-4.00bpw
```

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

Candidate. Pending: speculative decoding, DP=2 sweep, prefill kernels, long-context cells, repeated waves,
Gate A3. See `docs/PROGRESS.md`.
