"""ESIMD small-M EXL3 GEMM vs reference: correctness + weight-stream bandwidth."""
import sys, os, json, time
import torch
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exl3xpu import ref, ops

MODEL = os.environ.get("MODEL", "/models/turboderp-Qwen3.8-27B-exl3-4.00bpw")
dev = torch.device("xpu:0")
idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
E = ops._get_esimd()
assert E, "no esimd lib"

def load(key):
    t = {}
    for sub in ["trellis", "suh", "svh", "mul1"]:
        name = f"{key}.{sub}"
        if name in idx:
            with safe_open(f"{MODEL}/{idx[name]}", "pt", device="cpu") as f:
                t[sub] = f.get_tensor(name)
    return t

def bench(fn, iters=100):
    for _ in range(3): fn()
    torch.xpu.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.xpu.synchronize(); return (time.perf_counter() - t0) / iters

keys = sys.argv[1:] or ["model.language_model.layers.0.linear_attn.in_proj_qkv",
    "model.language_model.layers.0.mlp.down_proj", "model.language_model.layers.0.mlp.gate_proj", "lm_head"]
for key in keys:
    t = load(key); K = t["trellis"].shape[-1] // 16
    tr, suh, svh = t["trellis"].to(dev), t["suh"].to(dev), t["svh"].to(dev)
    k, n = suh.shape[0], svh.shape[0]
    shard = torch.zeros(n // 128, dtype=torch.int32, device=dev)
    wbytes = tr.numel() * 2
    for M in [1, 2, 4, 8]:
        x = torch.randn(M, k, dtype=torch.float16, device=dev)
        out = torch.empty((M, n), dtype=torch.float16, device=dev)
        f = lambda: E.exl3_gemm_small(x, tr, suh.unsqueeze(0), svh, shard, out, K, 2)
        f(); torch.xpu.synchronize()
        if n <= 32768:
            y_ref = ref.linear_forward(x, tr, suh, svh, K, 2)
            rel = ((out.float() - y_ref).norm() / y_ref.norm()).item()
        else:
            nn_ = 16384
            y_ref = ref.linear_forward(x, tr[:, : nn_ // 16].contiguous(), suh, svh[:nn_], K, 2)
            rel = ((out[:, :nn_].float() - y_ref).norm() / y_ref.norm()).item()
        tt = bench(f)
        print(f"{key.split('.')[-2:]} K={K} {k}x{n} M={M}: rel={rel:.2e}  {tt*1e6:8.1f} us  {wbytes/tt/1e9:6.1f} GB/s", flush=True)
