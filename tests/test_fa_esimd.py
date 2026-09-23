"""ESIMD flash-attention forward vs vllm_xpu_kernels FA2: correctness (O, LSE) and TFLOPS."""
import sys, os, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exl3xpu import ops
from exl3xpu.fp8kv_prefill import _fa
E = ops._get_esimd()
dev = torch.device("xpu")
HQ, HK, D = 24, 4, 256
cases = [(64, 64, True), (100, 300, True), (512, 4096, False), (8192, 8192, True), (8192, 32768, False)]
if len(sys.argv) > 1 and sys.argv[1] == "perf":
    cases = [(8192, 8192, True), (8192, 32768, False), (8192, 131072, False)]
for Lq, Lk, causal in cases:
    q = torch.randn(Lq, HQ, D, dtype=torch.float16, device=dev)
    k = torch.randn(Lk, HK, D, dtype=torch.float16, device=dev)
    v = torch.randn(Lk, HK, D, dtype=torch.float16, device=dev)
    o = torch.empty_like(q); lse = torch.empty(Lq, HQ, dtype=torch.float32, device=dev)
    f = lambda: E.exl3_fa_fwd(q, k, v, o, lse, causal, D ** -0.5)
    f(); torch.xpu.synchronize()
    ro, rl = _fa(q, k, v, causal, D ** -0.5)
    rel = ((o.float() - ro.float()).norm() / ro.float().norm()).item()
    lerr = (lse - rl).abs().max().item()
    it = 3
    t0 = time.perf_counter()
    for _ in range(it): f()
    torch.xpu.synchronize(); t_me = (time.perf_counter() - t0) / it
    t0 = time.perf_counter()
    for _ in range(it): _fa(q, k, v, causal, D ** -0.5)
    torch.xpu.synchronize(); t_fa = (time.perf_counter() - t0) / it
    eff = (Lk - Lq / 2) if causal else Lk
    fl = 4 * Lq * eff * HQ * D
    print(f"Lq={Lq:5d} Lk={Lk:6d} causal={causal}: rel={rel:.2e} lse_err={lerr:.2e}  "
          f"esimd {t_me*1e3:7.1f} ms {fl/t_me/1e12:5.1f} TF | fa2 {t_fa*1e3:7.1f} ms {fl/t_fa/1e12:5.1f} TF", flush=True)
