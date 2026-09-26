# Standalone: sgl_kernel XPU flash_attn_with_kvcache with an fp8 paged cache (bf16 q), prefill chunk with prefix.
import sys, time, torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache
torch.manual_seed(0)
dev = "xpu"; P = 128; Hq, Hk, D = 24, 4, 256
def run(nq, prefix, fp8=True, check=True):
    L = prefix + nq; pages = (L + P - 1) // P; npool = pages + 8
    kc = torch.randn(npool, P, Hk, D, device=dev, dtype=torch.bfloat16) * 0.5
    vc = torch.randn(npool, P, Hk, D, device=dev, dtype=torch.bfloat16) * 0.5
    perm = torch.randperm(npool, device=dev)[:pages].to(torch.int32)
    pt = perm.view(1, -1)
    q = torch.randn(nq, Hq, D, device=dev, dtype=torch.bfloat16) * 0.5
    cuq = torch.tensor([0, nq], dtype=torch.int32, device=dev); cs = torch.tensor([L], dtype=torch.int32, device=dev)
    kf, vf = (kc.to(torch.float8_e4m3fn), vc.to(torch.float8_e4m3fn)) if fp8 else (kc, vc)
    one = torch.ones((), device=dev, dtype=torch.float32).expand(1, Hk)
    kw = dict(k_descale=one, v_descale=one) if fp8 else {}
    torch.xpu.synchronize(); t = time.time()
    o = flash_attn_with_kvcache(q=q, k_cache=kf, v_cache=vf, page_table=pt, cache_seqlens=cs, cu_seqlens_q=cuq,
                                cu_seqlens_k_new=None, max_seqlen_q=nq, softmax_scale=D ** -0.5, causal=True, **kw)
    torch.xpu.synchronize(); dt = time.time() - t
    err = None
    if check:  # reference in fp32 on the dequantized cache, bottom-right causal
        K = kf.float()[perm.long()].reshape(-1, Hk, D)[:L]; V = vf.float()[perm.long()].reshape(-1, Hk, D)[:L]
        K = K.repeat_interleave(Hq // Hk, 1); V = V.repeat_interleave(Hq // Hk, 1)
        s = torch.einsum("qhd,khd->hqk", q.float(), K) * D ** -0.5
        mask = torch.arange(L, device=dev)[None, :] > (torch.arange(nq, device=dev)[:, None] + prefix)
        s.masked_fill_(mask[None], float("-inf"))
        ref = torch.einsum("hqk,khd->qhd", s.softmax(-1), V)
        err = ((o.float() - ref).abs().max() / ref.abs().max()).item()
    print(f"fp8={fp8} nq={nq} prefix={prefix}: {dt*1000:.1f} ms rel_err={err}", flush=True)
for nq, pre in [(4, 1000), (4096, 0), (512, 4096), (4096, 4096), (4096, 28672)]:
    run(nq, pre, fp8=True, check=pre + nq <= 8192)
