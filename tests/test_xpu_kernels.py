"""Correctness + timing of exl3xpu kernels on XPU against the bit-exact reference."""
import sys, os, json, time
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exl3xpu import ref
from exl3xpu import triton_kernels as tk

MODEL = os.environ.get("MODEL", "/models/turboderp-Qwen3.8-27B-exl3-4.00bpw")
dev = torch.device("xpu:0")
idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]


def load(key):
    t = {}
    for sub in ["trellis", "suh", "svh", "mcg", "mul1"]:
        name = f"{key}.{sub}"
        if name in idx:
            with safe_open(f"{MODEL}/{idx[name]}", "pt", device="cpu") as f:
                t[sub] = f.get_tensor(name)
    return t


def bench(fn, iters=50):
    fn(); torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters


keys = sys.argv[1:] or [
    "model.language_model.layers.0.linear_attn.in_proj_qkv",
    "model.language_model.layers.0.mlp.down_proj",
    "model.language_model.layers.0.mlp.gate_proj",
    "lm_head",
]
for key in keys:
    t = load(key)
    K = t["trellis"].shape[-1] // 16
    cb = ref.codebook_of("mcg" in t, "mul1" in t)
    tr = t["trellis"].to(dev)
    suh, svh = t["suh"].to(dev), t["svh"].to(dev)
    k, n = suh.shape[0], svh.shape[0]
    # bit-exact reconstruct (check a slice for lm_head)
    ncheck = min(n, 8192)
    w_ref = ref.reconstruct_inner(tr[:, : ncheck // 16].contiguous(), K, cb)
    w_x = tk.reconstruct(tr, K, cb, 0, ncheck)
    exact = torch.equal(w_ref.view(torch.int16), w_x.view(torch.int16))
    t_rec = bench(lambda: tk.reconstruct(tr, K, cb))
    gbs = (tr.numel() * 2 + k * n * 2) / t_rec / 1e9
    print(f"{key} K={K} k={k} n={n}: reconstruct bit_exact={exact}  {t_rec*1e3:.2f} ms  ({gbs:.0f} GB/s eff)")
    shard = torch.zeros(n // 128, dtype=torch.int32, device=dev)
    for M in [1, 4, 16, 64]:
        x = torch.randn(M, k, dtype=torch.float16, device=dev)
        def fwd():
            xh = tk.had_in(x, suh.unsqueeze(0))
            y32 = tk.gemm_small(xh, tr, K, cb, shard)
            out = torch.empty((M, n), dtype=torch.float16, device=dev)
            return tk.had_out(y32, svh, out)
        y = fwd().float()
        y_ref = ref.linear_forward(x, tr, suh, svh, K, cb) if n <= 32768 else None
        rel = ((y - y_ref).norm() / y_ref.norm()).item() if y_ref is not None else float("nan")
        tt = bench(fwd, 20)
        wbytes = tr.numel() * 2
        print(f"   M={M:3d} rel_err={rel:.2e}  {tt*1e3:.3f} ms  weight-stream {wbytes/tt/1e9:.0f} GB/s")
