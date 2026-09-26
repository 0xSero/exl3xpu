import time, torch
from exl3xpu.sglang_plugin import _sample_rows
V = 248320
def t(f, it=20):
    for _ in range(3): f()
    torch.xpu.synchronize(); s = time.time()
    for _ in range(it): f()
    torch.xpu.synchronize(); return (time.time() - s) / it * 1000
for rows in (4, 8, 16, 32):
    lg = torch.randn(rows, V, device="xpu", dtype=torch.float32) * 3
    temps = torch.full((rows,), 0.7, device="xpu"); tk = torch.full((rows,), 20, device="xpu", dtype=torch.int32); tp = torch.full((rows,), 0.95, device="xpu")
    full = t(lambda: _sample_rows(lg, temps, tk, tp))
    topk = t(lambda: torch.topk(lg, 64, dim=-1))
    print(f"rows={rows}: full-sort sampler {full:.2f} ms, torch.topk(64) {topk:.2f} ms", flush=True)
