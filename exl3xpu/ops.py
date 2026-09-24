"""
EXL3 linear forward on XPU, registered as an opaque torch custom op so torch.compile / vLLM
graph capture treat it as a single node.
"""
from __future__ import annotations
import os
import torch

from . import triton_kernels as tk

# Rows at or below this use the fused decode-in-GEMM kernel; above it we reconstruct fp16
# weight slices and use the oneDNN GEMM (compute bound regime).
SMALL_M_MAX = int(os.environ.get("EXL3_SMALL_M_MAX", "128"))
# Prefill (M > SMALL_M_MAX) GEMMs in int8 XMX (2x fp16 rate): per-token activation scale (Hadamard-rotated
# activations have no outliers), one static weight scale (the mul1 codebook bound). Opt-in. The C++ op fuses the
# quantization into had_in / reconstruct / had_out; the Python fallback below is the unfused prototype.
INT8_PREFILL = os.environ.get("EXL3_INT8_PREFILL", "0") == "1"
RECON_SLICE_N = int(os.environ.get("EXL3_RECON_SLICE_N", "16384"))

_backend = os.environ.get("EXL3_BACKEND", "auto")
_esimd = None


def _get_esimd():
    global _esimd
    if _esimd is None:
        try:
            torch.ops.load_library(os.environ.get("EXL3_LIB") or os.path.join(os.path.dirname(__file__), "_C.so"))
            _esimd = torch.ops.exl3xpu_C
            if hasattr(_esimd, "linear"):
                # symbolic-shape-aware fake impl (a C++ Meta kernel would specialise the token dim)
                @torch.library.register_fake("exl3xpu_C::linear")
                def _linear_fake(x, trellis, suh, svh, shard_of_nb, group_bounds, K, cb, small_m_max, slice_n):
                    return x.new_empty((*x.shape[:-1], svh.shape[0]))
            # split-K sizing overrides (defaults in csrc/exl3_ops.sycl are tuned on B70)
            if os.environ.get("EXL3_TARGET_THREADS"):
                _esimd.exl3_set_target_threads(int(os.environ["EXL3_TARGET_THREADS"]))
            if os.environ.get("EXL3_TARGET_THREADS_MB16") and hasattr(_esimd, "exl3_set_target_threads_mb16"):
                _esimd.exl3_set_target_threads_mb16(int(os.environ["EXL3_TARGET_THREADS_MB16"]))
            if os.environ.get("EXL3_TARGET_THREADS_MB64") and hasattr(_esimd, "exl3_set_target_threads_mb64"):
                _esimd.exl3_set_target_threads_mb64(int(os.environ["EXL3_TARGET_THREADS_MB64"]))
            if INT8_PREFILL and hasattr(_esimd, "exl3_set_int8"):
                _esimd.exl3_set_int8(1)          # fused W8A8 prefill inside the C++ linear
        except Exception as e:  # noqa
            if _backend != "triton":
                # never degrade silently to the ~5x slower Triton path; opt in with EXL3_BACKEND=triton
                raise RuntimeError(f"exl3xpu: failed to load the ESIMD op library: {e}") from e
            _esimd = False
    return _esimd


_wbuf: dict = {}


def _weight_buffer(device, numel):
    buf = _wbuf.get(device)
    if buf is None or buf.numel() < numel:
        buf = torch.empty(numel, dtype=torch.float16, device=device)
        _wbuf[device] = buf
    return buf[:numel]


def exl3_linear_impl(x: torch.Tensor, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor,
                     shard_of_nb: torch.Tensor, group_bounds: list[int], K: int, cb: int) -> torch.Tensor:
    shape = x.shape
    k = shape[-1]
    n = svh.shape[0]
    x2 = x.reshape(-1, k)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    M = x2.shape[0]
    out = torch.empty((M, n), dtype=x.dtype, device=x.device)
    if M == 0:
        return out.view(*shape[:-1], n)

    esimd = _get_esimd() if _backend != "triton" else False

    if M <= SMALL_M_MAX:
        if esimd and esimd.exl3_supported(K, cb):
            esimd.exl3_gemm_small(x2, trellis, suh, svh, shard_of_nb, out, K, cb)
        else:
            xh = tk.had_in(x2, suh)
            y32 = tk.gemm_small(xh, trellis, K, cb, shard_of_nb)
            tk.had_out(y32, svh, out)
        return out.view(*shape[:-1], n)

    # Large M: reconstruct Hadamard-domain fp16 weights slice by slice, oneDNN GEMM, output Hadamard
    fast = esimd and hasattr(esimd, "exl3_had_in_rm")
    if fast:
        xh = torch.empty((suh.shape[0], M, k), dtype=torch.float16, device=x.device)
        esimd.exl3_had_in_rm(x2, suh, xh)
    else:
        xh = tk.had_in(x2, suh)                          # [G, M, k] fp16
    y = torch.empty((M, n), dtype=torch.float16, device=x.device)
    for g in range(len(group_bounds) - 1):
        g0, g1 = group_bounds[g], group_bounds[g + 1]
        for n0 in range(g0, g1, RECON_SLICE_N):
            n1 = min(n0 + RECON_SLICE_N, g1)
            w = _weight_buffer(x.device, k * (n1 - n0)).view(k, n1 - n0)
            if esimd and hasattr(esimd, "exl3_reconstruct"):
                esimd.exl3_reconstruct(trellis, w, n0, K, cb)
            else:
                tk.reconstruct(trellis, K, cb, n0, n1 - n0, out=w)
            if INT8_PREFILL:
                sx = xh[g].abs().amax(1, keepdim=True).float().clamp_min(1e-8) / 127
                sw = w.abs().amax(0, keepdim=True).float().clamp_min(1e-8) / 127
                acc = torch._int_mm(torch.round(xh[g].float() / sx).to(torch.int8),
                                    torch.round(w.float() / sw).to(torch.int8))
                y[:, n0:n1] = (acc.float() * sx * sw).to(torch.float16)
            else:
                torch.matmul(xh[g], w, out=y[:, n0:n1])
    if fast:
        esimd.exl3_had_out_h(y, svh, out)
    else:
        tk.had_out(y, svh, out)
    return out.view(*shape[:-1], n)


@torch.library.custom_op("exl3xpu::linear", mutates_args=())
def _exl3_linear_py(x: torch.Tensor, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor,
                    shard_of_nb: torch.Tensor, group_bounds: list[int], K: int, cb: int) -> torch.Tensor:
    return exl3_linear_impl(x, trellis, suh, svh, shard_of_nb, group_bounds, K, cb)


@_exl3_linear_py.register_fake
def _(x, trellis, suh, svh, shard_of_nb, group_bounds, K, cb):
    return x.new_empty((*x.shape[:-1], svh.shape[0]))


def exl3_linear(x, trellis, suh, svh, shard_of_nb, group_bounds, K, cb):
    """EXL3 linear. Uses the all-C++ op (no Python per call) when the ESIMD library supports K/cb."""
    esimd = _get_esimd() if _backend != "triton" else False
    if INT8_PREFILL and x.numel() // x.shape[-1] > SMALL_M_MAX and not (esimd and hasattr(esimd, "exl3_set_int8")):
        return _exl3_linear_py(x, trellis, suh, svh, shard_of_nb, group_bounds, K, cb)
    if esimd and hasattr(esimd, "linear") and esimd.exl3_supported(K, cb):
        return esimd.linear(x, trellis, suh, svh, shard_of_nb, group_bounds, K, cb, SMALL_M_MAX, RECON_SLICE_N)
    return _exl3_linear_py(x, trellis, suh, svh, shard_of_nb, group_bounds, K, cb)
