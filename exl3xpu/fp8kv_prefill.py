"""
Prefill attention over an fp8 paged KV cache at fp16 kernel speed (Intel XPU).

vllm_xpu_kernels' FA2 runs ~39 TFLOPS when K/V are fp8 vs ~69 TFLOPS with fp16 (head_dim 256), which made
long-context prefill with an fp8 cache ~25% slower at 128K. For a prefill chunk (Q queries at the end of
an L-token sequence) we instead:
  - gather the sequence's cached K/V pages block by block (KB keys at a time), dequantize to fp16,
  - run the fp16 kernel per block (non-causal for keys before the chunk, causal for the chunk's own keys),
  - merge the partial outputs with their log-sum-exp.
Scratch is O(KB) instead of O(L); extra traffic is ~3 bytes/key/head per block, negligible vs attention.
"""
from __future__ import annotations
import torch

KB_DEFAULT = 32768

import os
# oneDNN Graph fused SDPA (exl3xpu_C::exl3_sdpa): ~80-90 TF at head_dim 256 vs ~65-70 for FA2 at long L.
# One call per KV head over the whole [0, seq_len) with a bottom-right causal mask, so no block merge is needed.
# Queries are padded to multiples of 256 at the top (exact: bottom-right alignment keeps the real rows' mask).
# Keys must keep their exact length: the fused kernel derives the causal alignment from the dims, and padded keys
# gave rel err 0.2-0.8 (tests/test_fp8kv_prefill.py). So each distinct seq_len compiles once (cached).
ONEDNN_ATTN = os.environ.get("EXL3_ONEDNN_ATTN", "0") == "1"
L_BUCKET = int(os.environ.get("EXL3_ONEDNN_LBUCKET", "1"))
Q_BUCKET = int(os.environ.get("EXL3_ONEDNN_QBUCKET", "256"))


def _onednn_op():
    from .ops import _get_esimd
    E = _get_esimd()
    return E if (E and hasattr(E, "exl3_sdpa")) else None


def gather_dequant_head(cache, pages, block_size, seq_len, h, scale, out):
    """K or V of kv-head h, tokens [0, seq_len), into out[:seq_len] (fp16 [Lpad, D]); out[seq_len:] zeroed."""
    p1 = (seq_len + block_size - 1) // block_size
    blk = cache[:, :, h].index_select(0, pages[:p1]).flatten(0, 1)[:seq_len]      # [seq_len, D] fp8
    torch.mul(blk.to(torch.float16), scale, out=out[:seq_len]) if scale != 1.0 else out[:seq_len].copy_(blk)
    out[seq_len:].zero_()
    return out


def prefill_attention_onednn(E, q, key_cache, value_cache, pages, seq_len, k_scale, v_scale, scale, out):
    Q, Hq, D = q.shape
    Hk, bs = key_cache.shape[2], key_cache.shape[1]
    G = Hq // Hk
    Lp = (seq_len + L_BUCKET - 1) // L_BUCKET * L_BUCKET
    Qp = (Q + Q_BUCKET - 1) // Q_BUCKET * Q_BUCKET
    kb = torch.empty((1, Lp, D), dtype=torch.float16, device=q.device)
    vb = torch.empty((1, Lp, D), dtype=torch.float16, device=q.device)
    qg = torch.zeros((G, Qp, D), dtype=torch.float16, device=q.device)   # real queries are the last Q rows
    og = torch.empty((G, Qp, D), dtype=torch.float16, device=q.device)
    none = torch.empty(0, device=q.device)
    qv, ov = q.view(Q, Hk, G, D), out.view(Q, Hk, G, D)
    for h in range(Hk):
        gather_dequant_head(key_cache, pages, bs, seq_len, h, k_scale, kb[0])
        gather_dequant_head(value_cache, pages, bs, seq_len, h, v_scale, vb[0])
        qg[:, Qp - Q:].copy_(qv[:, h].permute(1, 0, 2))
        E.exl3_sdpa_len(qg, kb, vb, og, none, True, scale, seq_len, Qp)
        ov[:, h].copy_(og[:, Qp - Q:].permute(1, 0, 2))
    return out


def _fa(q, k, v, causal, scale):
    from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func
    Q, L = q.shape[0], k.shape[0]
    cq = torch.tensor([0, Q], dtype=torch.int32, device=q.device)
    ck = torch.tensor([0, L], dtype=torch.int32, device=q.device)
    return flash_attn_varlen_func(q, k, v, Q, cq, L, cu_seqlens_k=ck, causal=causal,
                                  softmax_scale=scale, return_softmax_lse=True)


def gather_dequant(cache: torch.Tensor, pages: torch.Tensor, block_size: int, start: int, end: int,
                   scale: float) -> torch.Tensor:
    """Keys/values [start, end) of one sequence from a paged cache [num_pages, block_size, H, D] -> fp16 [n, H, D]."""
    p0, p1 = start // block_size, (end + block_size - 1) // block_size
    blk = cache.index_select(0, pages[p0:p1]).flatten(0, 1)          # [(p1-p0)*bs, H, D] (fp8)
    off = start - p0 * block_size
    x = blk[off: off + (end - start)].to(torch.float16)
    if scale != 1.0:
        x = x * scale
    return x


def prefill_attention(q: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, pages: torch.Tensor,
                      seq_len: int, k_scale: float, v_scale: float, scale: float, out: torch.Tensor,
                      kb: int = KB_DEFAULT) -> torch.Tensor:
    """q: [Q, Hq, D] fp16 (the last Q positions of a seq_len-token sequence; its K/V already in the cache)."""
    if ONEDNN_ATTN:
        E = _onednn_op()
        if E is not None:
            return prefill_attention_onednn(E, q, key_cache, value_cache, pages, seq_len, k_scale, v_scale, scale, out)
    Q = q.shape[0]
    bs = key_cache.shape[1]
    prefix = seq_len - Q
    acc = None
    m_lse = None
    parts = []
    # keys before the chunk: full attention, KB keys at a time
    for s in range(0, prefix, kb):
        e = min(s + kb, prefix)
        k = gather_dequant(key_cache, pages, bs, s, e, k_scale)
        v = gather_dequant(value_cache, pages, bs, s, e, v_scale)
        parts.append(_fa(q, k, v, False, scale))
    # the chunk's own keys: causal, square and bottom-right aligned
    k = gather_dequant(key_cache, pages, bs, prefix, seq_len, k_scale)
    v = gather_dequant(value_cache, pages, bs, prefix, seq_len, v_scale)
    parts.append(_fa(q, k, v, True, scale))
    if len(parts) == 1:
        out.copy_(parts[0][0])
        return out
    lses = torch.stack([p[1] for p in parts])                        # [B, Q, H] fp32
    m = lses.amax(0)
    w = torch.exp(lses - m)                                          # [B, Q, H]
    num = sum(p[0].float() * wi.unsqueeze(-1) for p, wi in zip(parts, w))
    out.copy_((num / w.sum(0).unsqueeze(-1)).to(out.dtype))
    return out
