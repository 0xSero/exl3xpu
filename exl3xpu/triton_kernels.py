"""
Triton (XPU) kernels for EXL3 trellis-quantized linears.

Weight storage (see ref.py): trellis int16 [k/16, n/16, 16*K] viewed here as int32 [k/16, n/16, 8*K].
For output element (row r, col c) of a 16x16 tile, the trellis stream index is
    t = 32*(c & 7) + 8*((r >> 1) & 3) + 4*(c >> 3) + 2*(r >> 3) + (r & 1)
and its 16-bit state is the window ending at bit (t+1)*K (MSB-first, circular within the tile).
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl


@triton.jit
def _decode_block(tr_ptr, tile_k0, tile_n0, tiles_n,
                  K: tl.constexpr, CB: tl.constexpr,
                  BK: tl.constexpr, BN: tl.constexpr):
    """Decode W_inner[tile_k0*16 : +BK, tile_n0*16 : +BN] -> fp16 [BK, BN]."""
    WORDS: tl.constexpr = 8 * K
    NBITS: tl.constexpr = 256 * K
    kk = tl.arange(0, BK)[:, None]
    nn = tl.arange(0, BN)[None, :]
    r = kk & 15
    c = nn & 15
    t = 32 * (c & 7) + 8 * ((r >> 1) & 3) + 4 * (c >> 3) + 2 * (r >> 3) + (r & 1)
    e = (t + 1) * K
    b0 = (e - 16 + NBITS) % NBITS
    i0 = b0 // 32
    i1 = (e - 1) // 32
    s = (i1 + 1) * 32 - e
    tile = (tile_k0 + (kk >> 4)) * tiles_n + (tile_n0 + (nn >> 4))
    base = tile.to(tl.int64) * WORDS
    w0 = tl.load(tr_ptr + base + i0).to(tl.uint32).to(tl.uint64)
    w1 = tl.load(tr_ptr + base + i1).to(tl.uint32).to(tl.uint64)
    st = (((w0 << 32) | w1) >> s.to(tl.uint64)) & 0xFFFF
    st = st.to(tl.uint32)
    if CB == 2:
        x = st * 0x83DCD12D
        bs = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24)
        v = (1024.0 + bs.to(tl.float32)) * 0.00676727294921875 + (-10.3828125)
        return v.to(tl.float16)
    else:
        if CB == 1:
            x = st * 0xCBAC1FED
        else:
            x = st * 89226354 + 64248484
        x = (x & 0x8FFF8FFF) ^ 0x3B603B60
        lo = (x & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
        hi = (x >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
        return (lo.to(tl.float32) + hi.to(tl.float32)).to(tl.float16)


@triton.jit
def _reconstruct_kernel(tr_ptr, w_ptr, tiles_n, n_total, n_offset_tiles, n_out,
                        K: tl.constexpr, CB: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    w = _decode_block(tr_ptr, pid_k * (BK // 16), n_offset_tiles + pid_n * (BN // 16), tiles_n, K, CB, BK, BN)
    rows = pid_k * BK + tl.arange(0, BK)[:, None]
    cols = pid_n * BN + tl.arange(0, BN)[None, :]
    tl.store(w_ptr + rows.to(tl.int64) * n_out + cols, w, mask=cols < n_out)


def reconstruct(trellis: torch.Tensor, K: int, cb: int, n_offset: int = 0, n: int | None = None,
                out: torch.Tensor | None = None) -> torch.Tensor:
    """W_inner[:, n_offset:n_offset+n] as fp16 [k, n]."""
    tk, tn, _ = trellis.shape
    k = tk * 16
    if n is None:
        n = tn * 16 - n_offset
    assert n_offset % 16 == 0 and n % 16 == 0
    if out is None:
        out = torch.empty((k, n), dtype=torch.float16, device=trellis.device)
    tr32 = trellis.view(torch.int32)
    BK, BN = 16, 128
    grid = (k // BK, triton.cdiv(n, BN))
    _reconstruct_kernel[grid](tr32, out, tn, tn * 16, n_offset // 16, n, K=K, CB=cb, BK=BK, BN=BN)
    return out


# ---------------------------------------------------------------------------------------------
# Hadamard-128 transforms (as dense +-1 matmul on the XMX units, scaled in fp32)

@triton.jit
def _had_in_kernel(x_ptr, suh_ptr, xh_ptr, h_ptr, M, Kdim, stride_xm,
                   S: tl.constexpr, BM: tl.constexpr):
    """xh[s, m, kb*128:+128] = H(x[m, kb] * suh[s, kb]) / sqrt(128) for all shards s."""
    pid_m = tl.program_id(0)
    kb = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)[:, None]
    rk = kb * 128 + tl.arange(0, 128)[None, :]
    mask = rm < M
    x = tl.load(x_ptr + rm * stride_xm + rk, mask=mask, other=0.0).to(tl.float32)
    hi = tl.arange(0, 128)
    h = tl.load(h_ptr + hi[:, None] * 128 + hi[None, :])
    for s in tl.static_range(S):
        su = tl.load(suh_ptr + s * Kdim + rk).to(tl.float32)
        xs = (x * su).to(tl.float16)
        y = tl.dot(xs, h) * 0.08838834764831845
        tl.store(xh_ptr + (s * M + rm) * Kdim + rk, y.to(tl.float16), mask=mask)


@triton.jit
def _had_out_kernel(y_ptr, svh_ptr, out_ptr, h_ptr, M, N, stride_om,
                    BM: tl.constexpr):
    pid_m = tl.program_id(0)
    nb = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)[:, None]
    rn = nb * 128 + tl.arange(0, 128)[None, :]
    mask = rm < M
    y = tl.load(y_ptr + rm * N + rn, mask=mask, other=0.0).to(tl.float32)
    hi = tl.arange(0, 128)
    h = tl.load(h_ptr + hi[:, None] * 128 + hi[None, :]).to(tl.float32)
    z = tl.dot(y, h) * 0.08838834764831845
    sv = tl.load(svh_ptr + rn).to(tl.float32)
    z = z * sv
    tl.store(out_ptr + rm * stride_om + rn, z.to(out_ptr.dtype.element_ty), mask=mask)


_H = {}


def had_pm1(device, dtype=torch.float16):
    key = (device, dtype)
    if key not in _H:
        h = torch.ones((1, 1))
        while h.shape[0] < 128:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _H[key] = h.to(device=device, dtype=dtype).contiguous()
    return _H[key]


def had_in(x: torch.Tensor, suh: torch.Tensor) -> torch.Tensor:
    """x [M, k], suh [S, k] -> xh [S, M, k] fp16."""
    M, k = x.shape
    S = suh.shape[0]
    xh = torch.empty((S, M, k), dtype=torch.float16, device=x.device)
    BM = 16 if M <= 16 else 32
    grid = (triton.cdiv(M, BM), k // 128)
    _had_in_kernel[grid](x, suh, xh, had_pm1(x.device), M, k, x.stride(0), S=S, BM=BM)
    return xh


def had_out(y32: torch.Tensor, svh: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    M, N = y32.shape
    BM = 16 if M <= 16 else 32
    grid = (triton.cdiv(M, BM), N // 128)
    _had_out_kernel[grid](y32, svh, out, had_pm1(y32.device, torch.float32), M, N, out.stride(0), BM=BM)
    return out


# ---------------------------------------------------------------------------------------------
# Fused small-M GEMM:  y32[m, n] += sum_k xh[shard(n), m, k] * W_inner[k, n]

@triton.jit
def _gemm_kernel(xh_ptr, tr_ptr, y_ptr, shard_of_nb_ptr, M, Kdim, N, tiles_n, k_per_split,
                 K: tl.constexpr, CB: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)
    shard = tl.load(shard_of_nb_ptr + (pid_n * BN) // 128)
    rm = pid_m * BM + tl.arange(0, BM)[:, None]
    mmask = rm < M
    k_start = pid_s * k_per_split
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    xrow = xh_ptr + (shard * M + rm).to(tl.int64) * Kdim
    for k0 in range(k_start, k_start + k_per_split, BK):
        rk = k0 + tl.arange(0, BK)[None, :]
        x = tl.load(xrow + rk, mask=mmask, other=0.0)
        w = _decode_block(tr_ptr, k0 // 16, pid_n * (BN // 16), tiles_n, K, CB, BK, BN)
        acc = tl.dot(x, w, acc)
    rn = pid_n * BN + tl.arange(0, BN)[None, :]
    tl.atomic_add(y_ptr + rm * N + rn, acc, mask=mmask, sem="relaxed")


def gemm_small(xh: torch.Tensor, trellis: torch.Tensor, K: int, cb: int,
               shard_of_nb: torch.Tensor, split_k: int | None = None) -> torch.Tensor:
    """xh [S, M, k] fp16; returns y32 [M, n] fp32 (Hadamard domain)."""
    S, M, k = xh.shape
    tk, tn, _ = trellis.shape
    n = tn * 16
    BN, BK = 128, 64
    BM = 16 if M <= 16 else (32 if M <= 32 else 64)
    nb = n // BN
    if split_k is None:
        # aim for ~4 waves over 256 EUs-worth of work-groups
        target = 512
        split_k = max(1, min(k // BK, target // max(1, nb)))
        while (k // BK) % split_k:
            split_k -= 1
    k_per_split = k // split_k
    y = torch.zeros((M, n), dtype=torch.float32, device=xh.device)
    grid = (nb, split_k, triton.cdiv(M, BM))
    _gemm_kernel[grid](xh, trellis.view(torch.int32), y, shard_of_nb, M, k, n, tn, k_per_split,
                       K=K, CB=cb, BM=BM, BN=BN, BK=BK)
    return y
