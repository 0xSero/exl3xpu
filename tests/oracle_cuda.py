"""Validate exl3xpu.ref against exllamav3's CUDA kernels (run inside a CUDA container with exllamav3)."""
import sys, json, os
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exl3xpu import ref
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import LinearEXL3

MODEL = sys.argv[1]
keys = sys.argv[2:] or [
    "model.language_model.layers.0.linear_attn.in_proj_qkv",
    "model.language_model.layers.3.self_attn.o_proj",
    "model.language_model.layers.0.mlp.down_proj",
    "lm_head",
]
idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
dev = torch.device("cuda:0")
torch.manual_seed(0)

def load(key):
    t = {}
    for sub in ["trellis", "suh", "svh", "mcg", "mul1", "bias"]:
        name = f"{key}.{sub}"
        if name in idx:
            with safe_open(f"{MODEL}/{idx[name]}", "pt", device="cpu") as f:
                t[sub] = f.get_tensor(name)
    return t

ok = True
for key in keys:
    t = load(key)
    K = t["trellis"].shape[-1] // 16
    cb = ref.codebook_of("mcg" in t, "mul1" in t)
    k, n = t["suh"].shape[0], t["svh"].shape[0]
    # limit lm_head size for the reference
    trellis = t["trellis"]
    svh = t["svh"]
    if n > 32768:
        trellis = trellis[:, : 32768 // 16].contiguous()
        svh = svh[:32768].contiguous()
        n = 32768
    tr = trellis.to(dev)
    w_cuda = torch.empty((k, n), dtype=torch.half, device=dev)
    ext.reconstruct(w_cuda, tr, K, "mcg" in t, "mul1" in t)
    w_ref = ref.reconstruct_inner(tr, K, cb)
    exact = torch.equal(w_cuda.view(torch.int16), w_ref.view(torch.int16))
    mism = (w_cuda.view(torch.int16) != w_ref.view(torch.int16)).sum().item()

    lin = LinearEXL3(None, k, n, suh=t["suh"].to(dev), svh=svh.to(dev), trellis=tr,
                     mcg=t.get("mcg", None) if "mcg" not in t else t["mcg"].to(dev),
                     mul1=t["mul1"].to(dev) if "mul1" in t else None)
    res = {}
    for rows in [1, 7, 200]:
        x = torch.randn(rows, k, dtype=torch.half, device=dev)
        y_cuda = lin.forward(x, {}).float()
        y_ref = ref.linear_forward(x, tr, t["suh"].to(dev), svh.to(dev), K, cb)
        rel = ((y_cuda - y_ref).norm() / y_ref.norm()).item()
        res[rows] = rel
    w_full_ref = ref.weight_orig(tr, t["suh"].to(dev), svh.to(dev), K, cb)
    w_full_cuda = lin.get_weight_tensor().float()
    wrel = ((w_full_cuda - w_full_ref).norm() / w_full_ref.norm()).item()
    good = exact and all(v < 2e-3 for v in res.values()) and wrel < 2e-3
    ok &= good
    print(f"{key}: K={K} cb={cb} k={k} n={n} inner_bit_exact={exact} mismatches={mism} "
          f"fwd_rel={ {r: f'{v:.2e}' for r, v in res.items()} } w_orig_rel={wrel:.2e} {'OK' if good else 'FAIL'}")
print("ALL_OK" if ok else "SOME_FAIL")
