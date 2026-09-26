"""Fast top-k sampler == full-sort sampler in distribution (same semantics), and timing."""
import os, time, torch
import exl3xpu.sglang_plugin as sp
torch.manual_seed(0)
V, rows = 248320, 8
lg = torch.randn(rows, V, device="xpu") * 4
temps = torch.full((rows,), 0.7, device="xpu"); tk = torch.full((rows,), 20, device="xpu", dtype=torch.int32); tp = torch.full((rows,), 0.95, device="xpu")
def hist(fast, n=4000):
    sp._TOPK_FAST = 256 if fast else 0
    c = {}
    for _ in range(n // 50):
        for _ in range(50):
            t = sp._sample_rows(lg[:1].expand(1, V).contiguous(), temps[:1], tk[:1], tp[:1]).item()
            c[t] = c.get(t, 0) + 1
    return c
a, b = hist(True), hist(False)
keys = set(a) | set(b)
tv = sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys) / 2 / 4000
print(f"support fast {len(a)} full {len(b)}; total variation distance {tv:.3f} (sampling noise ~0.03)")
for fast in (True, False):
    sp._TOPK_FAST = 256 if fast else 0
    for _ in range(3): sp._sample_rows(lg, temps, tk, tp)
    torch.xpu.synchronize(); t = time.time()
    for _ in range(20): sp._sample_rows(lg, temps, tk, tp)
    torch.xpu.synchronize(); print(f"fast={fast}: {(time.time()-t)/20*1000:.2f} ms for {rows} rows")
