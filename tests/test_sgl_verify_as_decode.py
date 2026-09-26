"""Verify-as-decode route (exl3xpu.sglang_plugin) == sgl_kernel's own extend path for nq-query verify batches."""
import os, time, torch
os.environ["EXL3_SGL_VERIFY_AS_DECODE"] = "1"
from sgl_kernel.flash_attn import flash_attn_with_kvcache as ref
from sglang.srt.layers.attention import xpu_backend as xb
from exl3xpu import sglang_plugin as sp
sp._patch_xpu_attn_routes()
new = xb.flash_attn_with_kvcache
assert getattr(new, "_exl3", False)
dev = "xpu"; P = 128; Hq, Hk, D = 24, 4, 256
torch.manual_seed(0)
worst = 0
for fp8 in (True, False):
    for bs, nq, lens in [(1, 4, [300]), (3, 4, [5000, 129, 16384]), (2, 2, [40000, 777]), (4, 3, [64, 128, 200, 9000])]:
        pages = [(L + P - 1) // P for L in lens]; mp = max(pages); npool = sum(pages) + 4
        kc = torch.randn(npool, P, Hk, D, device=dev, dtype=torch.bfloat16) * 0.5
        vc = torch.randn(npool, P, Hk, D, device=dev, dtype=torch.bfloat16) * 0.5
        if fp8:
            kc, vc = kc.to(torch.float8_e4m3fn), vc.to(torch.float8_e4m3fn)
        perm = torch.randperm(npool, device=dev).to(torch.int32)
        pt = torch.zeros(bs, mp, dtype=torch.int32, device=dev); o = 0
        for i, n in enumerate(pages):
            pt[i, :n] = perm[o:o + n]; o += n
        q = torch.randn(bs * nq, Hq, D, device=dev, dtype=torch.bfloat16)
        kw = dict(q=q, k_cache=kc, v_cache=vc, page_table=pt, cache_seqlens=torch.tensor(lens, dtype=torch.int32, device=dev),
                  cu_seqlens_q=torch.arange(bs + 1, dtype=torch.int32, device=dev) * nq, cu_seqlens_k_new=None,
                  max_seqlen_q=nq, softmax_scale=D ** -0.5, causal=True)
        if fp8:
            one = torch.ones((), device=dev, dtype=torch.float32).expand(bs, Hk)
            kw.update(k_descale=one, v_descale=one)
        a = ref(**kw); b = new(**kw)
        err = ((a.float() - b.float()).abs().max() / a.float().abs().max()).item()
        worst = max(worst, err)
        print(f"fp8={fp8} bs={bs} nq={nq} lens={lens}: rel err {err:.2e}", flush=True)
print("VERIFY_AS_DECODE", "PASS" if worst < 2e-2 else "FAIL", f"worst {worst:.2e}")
