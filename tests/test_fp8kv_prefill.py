"""Block-dequant fp8-KV prefill attention == one full fp16 causal pass on the same (fp8-rounded) K/V."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exl3xpu.fp8kv_prefill import prefill_attention, _fa
dev = torch.device("xpu")
HQ, HK, D, BS = 24, 4, 256, 64
ok = True
for L, Q, kb in [(4096, 512, 1024), (70000, 8192, 32768), (20000, 20000, 32768)]:
    npages = (L + BS - 1) // BS + 3
    kc = (torch.randn(npages, BS, HK, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
    vc = (torch.randn(npages, BS, HK, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
    pages = torch.randperm(npages, device=dev)[: (L + BS - 1) // BS]
    q = torch.randn(Q, HQ, D, dtype=torch.float16, device=dev)
    ks, vs = 0.75, 1.25
    kfull = kc.index_select(0, pages).flatten(0, 1)[:L].to(torch.float16) * ks
    vfull = vc.index_select(0, pages).flatten(0, 1)[:L].to(torch.float16) * vs
    ref = _fa(q, kfull, vfull, True, D ** -0.5)[0].float()
    out = torch.empty(Q, HQ, D, dtype=torch.float16, device=dev)
    prefill_attention(q, kc, vc, pages, L, ks, vs, D ** -0.5, out, kb=kb)
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    good = rel < 2e-3
    ok &= good
    print(f"L={L} Q={Q} kb={kb}: rel={rel:.2e} {'OK' if good else 'FAIL'}")
print("FP8KV_PREFILL_PASS" if ok else "FP8KV_PREFILL_FAIL")
