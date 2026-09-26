import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache
dev = "xpu"; P = 128; Hq, Hk, D = 24, 4, 256
npool = 2048
kc = torch.zeros(npool, P, Hk, D, device=dev, dtype=torch.float8_e4m3fn); vc = torch.zeros_like(kc)
for L in (8192, 32768, 131072):
    nq = 4096; pages = L // P
    pt = torch.arange(pages, device=dev, dtype=torch.int32).view(1, -1)
    q = torch.randn(nq, Hq, D, device=dev, dtype=torch.bfloat16)
    one = torch.ones((), device=dev, dtype=torch.float32).expand(1, Hk)
    torch.xpu.synchronize(); torch.xpu.reset_peak_memory_stats(); base = torch.xpu.memory_allocated()
    f0 = torch.xpu.mem_get_info()[0]
    o = flash_attn_with_kvcache(q=q, k_cache=kc, v_cache=vc, page_table=pt, cache_seqlens=torch.tensor([L], dtype=torch.int32, device=dev),
                                cu_seqlens_q=torch.tensor([0, nq], dtype=torch.int32, device=dev), cu_seqlens_k_new=None,
                                max_seqlen_q=nq, softmax_scale=D ** -0.5, causal=True, k_descale=one, v_descale=one)
    torch.xpu.synchronize()
    f1 = torch.xpu.mem_get_info()[0]
    print(f"L={L}: torch peak +{(torch.xpu.max_memory_allocated()-base)/2**20:.0f} MiB, device free delta {(f0-f1)/2**20:.0f} MiB")
