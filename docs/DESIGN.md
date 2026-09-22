# Design

## EXL3 format (as consumed here)

A linear `y = x @ W`, `W` is `(k, n)`:

| tensor | dtype / shape | meaning |
|---|---|---|
| `trellis` | int16 `[k/16, n/16, 16*K]` | one 16x16 tile per `(i, j)`, `256*K` bits per tile |
| `suh` | fp16 `[k]` | input sign/scale vector |
| `svh` | fp16 `[n]` | output sign/scale vector |
| `mul1` / `mcg` | int32 scalar | presence selects the codebook (neither: 3INST) |

Tile bitstream: uint32 words (little-endian int16 pairs), bits MSB-first. Value `t` (0..255) is decoded from
the 16-bit window ending at bit `(t+1)*K`, circular within the tile, and lands at

```
row = 8*((t>>1)&1) + 2*((t>>3)&3) + (t&1)        (k)
col = (t>>5) + 8*((t>>2)&1)                        (n)
```

`mul1` codebook: `x = state * 0x83DCD12D`, `v = fp16(1024 + bytesum(x)) * 0x1EEE(fp16) + 0xC931(fp16)` (one
fp16 FMA). `W = diag(suh) H W_inner H diag(svh)` with `H` the normalized Sylvester Hadamard in 128 blocks, so
`y = had(had(x*suh) @ W_inner) * svh`.

## Kernels (csrc/exl3_esimd.h)

**Bit periodicity.** With `g = gcd(K, 32)`, every `D = K/g` words hold `V = 32/g` whole values, so value
`V*grp + u` always sits at the same bit offset relative to word `D*grp`. Kernels vectorise over `grp` and fully
unroll `u`: each state is two strided register selects, a funnel shift and a mask. K=4: 8 values per word.

**Codebook in 3 instructions.** `dp4a(1024, x, 0x01010101)` gives `1024 + bytesum` in one instruction, then
u32->fp16 and one fp16 `mad` reproduce CUDA's `__hfma` rounding exactly.

**Vector GEMV (M <= 4).** Accumulators `acc[h][grp]` per output half; the x value for `(grp, u)` is a
compile-time strided replicate of the 16-row x block, so there is no gather. 415-630 GB/s weight streaming.

**DPAS GEMM (M 5..128).** A 16x16 EXL3 tile is exactly one Xe2 DPAS B operand (K16 x N16, VNNI). Reading the
trellis words in transposed lane order makes each decoded vector land in VNNI with strided region moves.
One decode feeds `MB/8` DPAS ops. Activations are stored blocked (`[k/16][M][16]`) so a tile-row's A
operand is one contiguous load.

**Split-K + Hadamard epilogue.** Threads = column strips x K splits; partial sums go to an fp32 buffer; one
kernel sums the splits, applies the output Hadamard (in-register butterflies) and `svh`.

**Prefill (M > 128).** Reconstruct fp16 `W_inner` slices (bit-exact ESIMD kernel) and run oneDNN GEMM.

## vLLM integration

`exl3` is registered with `register_quantization_config`, loaded in every vLLM process through the
`vllm.general_plugins` entry point. vLLM fuses sibling projections (qkv, gate/up, GDN in_proj_qkvz); each
EXL3 constituent keeps its own `suh`, so the input Hadamard produces one activation copy per shard group and
the GEMM picks the copy by output block (`shard_of_nb`). Unquantized tensors (norms, `in_proj_a/b`) go through
vLLM's normal path. The op is an opaque `torch.library` custom op, so torch.compile and XPU graph capture
treat it as one node.

## Build notes

- JIT (`spir64`) only: AOT through the system `ocloc` rejects the GEMM kernels ("more than one module with an
  entry point"), and llm-scaler's builder hardcodes AOT targets, so `scripts/build_ext.sh` calls icpx directly.
- `setvars.sh` must be sourced before `set -u`.
- XPU graphs: `FULL_DECODE_ONLY` with explicit capture sizes; piecewise capture of large prefill sizes fails
  with `UR_RESULT_ERROR_OUT_OF_RESOURCES`.
