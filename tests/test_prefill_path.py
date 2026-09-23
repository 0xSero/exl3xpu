"""Large-M (prefill) EXL3 linear path vs the reference: fused shards, both dtypes."""
import sys, os, json, torch
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exl3xpu import ref, ops
MODEL = os.environ.get("MODEL", "/models/turboderp-Qwen3.8-27B-exl3-4.00bpw")
dev = torch.device("xpu:0")
idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
def get(n):
    with safe_open(f"{MODEL}/{idx[n]}", "pt", device="cpu") as f:
        return f.get_tensor(n)
ok = True
for keys in [["model.language_model.layers.0.mlp.gate_proj", "model.language_model.layers.0.mlp.up_proj"],
             ["model.language_model.layers.0.mlp.down_proj"]]:
    trs = [get(k + ".trellis").to(dev) for k in keys]
    suhs = [get(k + ".suh").to(dev) for k in keys]
    svhs = [get(k + ".svh").to(dev) for k in keys]
    tr = torch.cat(trs, 1).contiguous(); suh = torch.stack(suhs); svh = torch.cat(svhs)
    widths = [t.shape[1] * 16 for t in trs]
    bounds = [0, widths[0]] + ([widths[0] + widths[1]] if len(widths) > 1 else [])
    shard = torch.cat([torch.full((w // 128,), i, dtype=torch.int32) for i, w in enumerate(widths)]).to(dev)
    for M in [300, 2048]:
        for dt in [torch.float16, torch.bfloat16]:
            x = torch.randn(M, suh.shape[1], dtype=dt, device=dev)
            y = ops.exl3_linear_impl(x, tr, suh, svh, shard, bounds, 4, 2).float()
            yr = torch.cat([ref.linear_forward(x.half(), t, s, v, 4, 2) for t, s, v in zip(trs, suhs, svhs)], 1)
            rel = ((y - yr).norm() / yr.norm()).item()
            good = rel < 2e-3
            ok &= good
            print(f"{keys[0].split('.')[-1]} M={M} {dt}: rel={rel:.2e} {'OK' if good else 'FAIL'}")
print("PREFILL_PATH_PASS" if ok else "PREFILL_PATH_FAIL")
