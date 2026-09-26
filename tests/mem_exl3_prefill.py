"""Peak torch memory of one EXL3 prefill linear call (M rows) at Qwen3.8-27B shapes (synthetic 4-bit trellis)."""
import os, sys, torch
from exl3xpu import ops
E = ops._get_esimd()
dev = "xpu"
M = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
shapes = {"in_proj_qkvz": (5120, [10240, 6144]), "gate_up": (5120, [17408, 17408]), "down": (17408, [5120]),
          "qkv(attn)": (5120, [12288, 1024, 1024])}
for name, (k, ns) in shapes.items():
    n = sum(ns)
    tr = torch.randint(-32768, 32767, (k // 16, n // 16, 64), dtype=torch.int16, device=dev)
    suh = torch.randn(len(ns), k, device=dev, dtype=torch.float16).sign()
    svh = torch.randn(n, device=dev, dtype=torch.float16).sign()
    bounds = [0]
    for w in ns: bounds.append(bounds[-1] + w)
    sonb = torch.cat([torch.full((w // 128,), g, dtype=torch.int32) for g, w in enumerate(ns)]).to(dev)
    x = torch.randn(M, k, device=dev, dtype=torch.bfloat16)
    for _ in range(2):
        torch.xpu.synchronize(); torch.xpu.reset_peak_memory_stats(); base = torch.xpu.memory_allocated()
        y = E.linear(x, tr, suh, svh, sonb, bounds, 4, 2, ops.SMALL_M_MAX, ops.RECON_SLICE_N)
        torch.xpu.synchronize()
        pk = (torch.xpu.max_memory_allocated() - base) / 2**20
        del y
    print(f"{name:14s} k={k} n={n} M={M}: peak +{pk:.0f} MiB (output {M*n*2/2**20:.0f} MiB)", flush=True)
    del tr
