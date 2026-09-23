"""XPU flash-attention (vllm_xpu_kernels FA2 varlen) throughput for Qwen3.8 attention shapes.
An 8192-token query chunk attends causally (bottom-right aligned) over Lk keys, as in chunked prefill."""
import time, torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func
dev = torch.device("xpu")
HQ, HK, D, Q = 24, 4, 256, 8192
import os
FP8 = os.environ.get("FP8") == "1"
for Lk in [8192, 32768, 131072, 262144]:
    q = torch.randn(Q, HQ, D, dtype=torch.float16, device=dev)
    k = torch.randn(Lk, HK, D, dtype=torch.float16, device=dev)
    v = torch.randn(Lk, HK, D, dtype=torch.float16, device=dev)
    cq = torch.tensor([0, Q], dtype=torch.int32, device=dev)
    ck = torch.tensor([0, Lk], dtype=torch.int32, device=dev)
    kw = {}
    if FP8:
        k = k.to(torch.float8_e4m3fn); v = v.to(torch.float8_e4m3fn)
        one = torch.ones((), dtype=torch.float32, device=dev).expand(1, HK)
        kw = dict(k_descale=one, v_descale=one)
    f = lambda: flash_attn_varlen_func(q, k, v, Q, cq, Lk, cu_seqlens_k=ck, causal=True, softmax_scale=D ** -0.5, **kw)
    f(); torch.xpu.synchronize()
    it = 3
    t0 = time.perf_counter()
    for _ in range(it):
        f()
    torch.xpu.synchronize()
    dt = (time.perf_counter() - t0) / it
    eff_k = Lk - Q / 2                                  # mean keys per query under causal masking
    flops = 4 * Q * eff_k * HQ * D
    print(f"Lk={Lk:7d}: {dt*1e3:8.1f} ms per 8K chunk, {flops/dt/1e12:5.1f} TFLOPS", flush=True)
    del q, k, v
