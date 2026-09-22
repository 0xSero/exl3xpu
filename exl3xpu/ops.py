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
SMALL_M_MAX = int(os.environ.get("EXL3_SMALL_M_MAX", "16"))
RECON_SLICE_N = int(os.environ.get("EXL3_RECON_SLICE_N", "16384"))

_backend = os.environ.get("EXL3_BACKEND", "auto")
_esimd = None


def _get_esimd():
    global _esimd
    if _esimd is None:
        try:
            torch.ops.load_library(os.path.join(os.path.dirname(__file__), "_C.so"))
            _esimd = torch.ops.exl3xpu_C
        except Exception as e:  # noqa
            import logging
            logging.getLogger(__name__).warning("exl3xpu: ESIMD ops unavailable (%s), using Triton", e)
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

    esimd = _get_esimd() if _backend in ("auto", "esimd") else False

    if M <= SMALL_M_MAX:
        if esimd and M <= 8 and esimd.exl3_supported(K, cb):
            esimd.exl3_gemm_small(x2, trellis, suh, svh, shard_of_nb, out, K, cb)
        else:
            xh = tk.had_in(x2, suh)
            y32 = tk.gemm_small(xh, trellis, K, cb, shard_of_nb)
            tk.had_out(y32, svh, out)
        return out.view(*shape[:-1], n)

    # Large M: reconstruct Hadamard-domain fp16 weights slice by slice, oneDNN GEMM, output Hadamard
    xh = tk.had_in(x2, suh)                              # [G, M, k] fp16
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
            torch.matmul(xh[g], w, out=y[:, n0:n1])
    tk.had_out(y, svh, out)
    return out.view(*shape[:-1], n)


@torch.library.custom_op("exl3xpu::linear", mutates_args=())
def exl3_linear(x: torch.Tensor, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor,
                shard_of_nb: torch.Tensor, group_bounds: list[int], K: int, cb: int) -> torch.Tensor:
    return exl3_linear_impl(x, trellis, suh, svh, shard_of_nb, group_bounds, K, cb)


@exl3_linear.register_fake
def _(x, trellis, suh, svh, shard_of_nb, group_bounds, K, cb):
    return x.new_empty((*x.shape[:-1], svh.shape[0]))
