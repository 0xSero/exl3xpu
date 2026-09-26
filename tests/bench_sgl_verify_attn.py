"""Time sgl_kernel XPU flash_attn_with_kvcache for the MTP verify shape (4 queries per sequence, paged cache, causal
bottom-right) vs context length, fp8 vs bf16 cache; and the 1-query decode shape. Qwen3.8-27B: 24 q heads, 4 kv, D 256."""
import sys, time, torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache

dev = "xpu"; P = 128; Hq, Hk, D = 24, 4, 256


def bench(nq, L, bs=1, fp8=True, iters=20):
    pages = (L + P - 1) // P
    npool = pages * bs + 8
    kc = (torch.randn(npool, P, Hk, D, device=dev, dtype=torch.bfloat16) * 0.5)
    vc = (torch.randn(npool, P, Hk, D, device=dev, dtype=torch.bfloat16) * 0.5)
    if fp8:
        kc, vc = kc.to(torch.float8_e4m3fn), vc.to(torch.float8_e4m3fn)
    pt = torch.randperm(npool, device=dev)[:pages * bs].to(torch.int32).view(bs, pages)
    q = torch.randn(nq * bs, Hq, D, device=dev, dtype=torch.bfloat16) * 0.5
    cuq = (torch.arange(bs + 1, device=dev, dtype=torch.int32) * nq)
    cs = torch.full((bs,), L, dtype=torch.int32, device=dev)
    kw = {}
    if fp8:
        one = torch.ones((), device=dev, dtype=torch.float32).expand(bs, Hk)
        kw = dict(k_descale=one, v_descale=one)
    f = lambda: flash_attn_with_kvcache(q=q, k_cache=kc, v_cache=vc, page_table=pt, cache_seqlens=cs, cu_seqlens_q=cuq,
                                        cu_seqlens_k_new=None, max_seqlen_q=nq, softmax_scale=D ** -0.5, causal=True, **kw)
    for _ in range(3):
        f()
    torch.xpu.synchronize(); t = time.time()
    for _ in range(iters):
        f()
    torch.xpu.synchronize()
    return (time.time() - t) / iters * 1000


for L in [1024, 4096, 16384, 65536]:
    r = [f"L={L:6d}"]
    for nq in (1, 4):
        for fp8 in (True, False):
            r.append(f"nq={nq} {'fp8 ' if fp8 else 'bf16'} {bench(nq, L, fp8=fp8):7.3f} ms")
    print("  ".join(r), flush=True)
print("bs=8 nq=4 L=16384 fp8:", round(bench(4, 16384, bs=8), 3), "ms;  bf16:", round(bench(4, 16384, bs=8, fp8=False), 3), "ms")
