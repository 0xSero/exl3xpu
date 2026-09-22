"""
Bit-exact pure-PyTorch reference for EXL3 (exllamav3 v1.x) linear layers.

Format (per linear, W is (in_features=k, out_features=n), y = x @ W):
  trellis : int16 [k/16, n/16, 16*K]   one 16x16 tile per (i, j), 256*K bits per tile
  suh     : fp16  [k]                  input sign/scale vector
  svh     : fp16  [n]                  output sign/scale vector
  mcg/mul1: int32 scalar               presence selects codebook (none -> 3INST)

Tile bitstream: uint32 words (little-endian pairs of the int16 storage), bits MSB-first.
Element t (0..255) of a tile is decoded from the 16-bit window that ENDS at bit (t+1)*K
(circular inside the tile). Element t lands at tile position perm[t] (tensor-core order).

W_orig = diag(suh) . H128_blocks(rows) . W_inner . H128_blocks(cols) . diag(svh)
where H128 is the normalized Sylvester Hadamard matrix. Equivalently:
  y = had128(x * suh) @ W_inner  -> had128(.) * svh
"""
from __future__ import annotations
import math
from functools import lru_cache
import numpy as np
import torch

MCG_MULT = 0xCBAC1FED
MUL1_MULT = 0x83DCD12D

CB_3INST, CB_MCG, CB_MUL1 = 0, 1, 2


def tensor_core_perm() -> list[int]:
    perm = [0] * 256
    for t in range(32):
        r0 = (t % 4) * 2
        r1, r2, r3 = r0 + 1, r0 + 8, r0 + 9
        c0 = t // 4
        c1 = c0 + 8
        perm[t * 8 + 0] = r0 * 16 + c0
        perm[t * 8 + 1] = r1 * 16 + c0
        perm[t * 8 + 2] = r2 * 16 + c0
        perm[t * 8 + 3] = r3 * 16 + c0
        perm[t * 8 + 4] = r0 * 16 + c1
        perm[t * 8 + 5] = r1 * 16 + c1
        perm[t * 8 + 6] = r2 * 16 + c1
        perm[t * 8 + 7] = r3 * 16 + c1
    return perm


def _f16_from_bits(u: np.ndarray) -> np.ndarray:
    return u.astype(np.uint16).view(np.float16)


@lru_cache
def codebook_lut(cb: int) -> torch.Tensor:
    """fp16 value for every 16-bit trellis state, matching the CUDA decode_3inst<cb> bit for bit."""
    s = np.arange(65536, dtype=np.uint64)
    if cb == CB_MUL1:
        x = (s * MUL1_MULT) & 0xFFFFFFFF
        bsum = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)
        h = (1024 + bsum).astype(np.float64)                     # fp16 0x6400 + bsum, exact
        inv = float(_f16_from_bits(np.array([0x1EEE]))[0])
        bias = float(_f16_from_bits(np.array([0xC931]))[0])
        # __hfma: exact product+sum, single rounding to fp16. float64 holds h*inv+bias exactly
        val = (h * inv + bias).astype(np.float16)
    else:
        if cb == CB_3INST:
            x = (s * 89226354 + 64248484) & 0xFFFFFFFF
        else:
            x = (s * MCG_MULT) & 0xFFFFFFFF
        x = (x & 0x8FFF8FFF) ^ 0x3B603B60                            # lop3 0x6a == (a & b) ^ c
        lo = _f16_from_bits(x & 0xFFFF).astype(np.float64)
        hi = _f16_from_bits(x >> 16).astype(np.float64)
        val = (lo + hi).astype(np.float16)                           # __hadd: single rounding
    return torch.from_numpy(val.copy())


def codebook_of(mcg: bool, mul1: bool) -> int:
    return CB_MCG if mcg else (CB_MUL1 if mul1 else CB_3INST)


@lru_cache
def _state_index(K: int):
    """For each t: the 16 bit positions (MSB first) of its window inside the 256*K-bit tile stream."""
    nbits = 256 * K
    t = np.arange(256)
    b0 = (t * K + K - 16 + nbits) % nbits
    pos = (b0[:, None] + np.arange(16)[None, :]) % nbits
    return torch.from_numpy(pos.astype(np.int64))


def decode_states(trellis: torch.Tensor, K: int) -> torch.Tensor:
    """trellis int16 [..., 16*K] -> int32 states [..., 256] in stream order t."""
    t16 = trellis.to(torch.int64) & 0xFFFF
    words = t16[..., 0::2] | (t16[..., 1::2] << 16)                  # [..., 8K] uint32 values
    shifts = torch.arange(31, -1, -1, device=trellis.device)
    bits = ((words.unsqueeze(-1) >> shifts) & 1).flatten(-2)          # [..., 256K], MSB first
    pos = _state_index(K).to(trellis.device)
    win = bits[..., pos]                                               # [..., 256, 16]
    weights = (1 << torch.arange(15, -1, -1, device=trellis.device))
    return (win * weights).sum(-1).to(torch.int32)


def reconstruct_inner(trellis: torch.Tensor, K: int, cb: int, chunk_tiles: int = 1 << 15) -> torch.Tensor:
    """trellis [k/16, n/16, 16K] -> W_inner fp16 [k, n] (Hadamard domain, no sign vectors)."""
    tk, tn, _ = trellis.shape
    lut = codebook_lut(cb).to(trellis.device)
    perm = torch.tensor(tensor_core_perm(), device=trellis.device)
    flat = trellis.reshape(tk * tn, 16 * K)
    out = torch.empty((tk * tn, 256), dtype=torch.float16, device=trellis.device)
    for s in range(0, tk * tn, chunk_tiles):
        st = decode_states(flat[s:s + chunk_tiles], K)
        vals = lut[st.long()]
        tile = torch.empty_like(vals)
        tile[:, perm] = vals
        out[s:s + chunk_tiles] = tile
    return out.view(tk, tn, 16, 16).permute(0, 2, 1, 3).reshape(tk * 16, tn * 16)


@lru_cache
def hadamard128(device=None, dtype=torch.float32) -> torch.Tensor:
    h = torch.ones((1, 1), dtype=torch.float64)
    while h.shape[0] < 128:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(128)).to(dtype=dtype, device=device)


def had_rows(x: torch.Tensor) -> torch.Tensor:
    """Apply H128 on the last dim in blocks of 128 (x @ H, H symmetric)."""
    shp = x.shape
    h = hadamard128(x.device, torch.float32)
    return (x.float().reshape(-1, 128) @ h).reshape(shp)


def weight_orig(trellis, suh, svh, K, cb) -> torch.Tensor:
    """Full original-basis fp32 weight (k, n)."""
    w = reconstruct_inner(trellis, K, cb).float()
    k, n = w.shape
    h = hadamard128(w.device, torch.float32)
    w = (h @ w.view(k // 128, 128, n)).view(k, n)       # H on rows
    w = w * suh.float().unsqueeze(1)
    w = (w.view(k, n // 128, 128) @ h).view(k, n)       # H on cols
    w = w * svh.float().unsqueeze(0)
    return w


def linear_forward(x, trellis, suh, svh, K, cb, bias=None) -> torch.Tensor:
    """Reference forward via the Hadamard-domain path (fp32 accumulate)."""
    w = reconstruct_inner(trellis, K, cb).float()
    xh = had_rows(x.float() * suh.float())
    y = xh @ w
    y = had_rows(y) * svh.float()
    if bias is not None:
        y = y + bias.float()
    return y
