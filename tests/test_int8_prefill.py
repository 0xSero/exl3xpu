"""
W8A8 prefill linear (EXL3_INT8 C++ path) vs fp16 path on real checkpoint tensors: time per layer type, relative
error vs fp16, and agreement with a torch reference of the same quantization (kernel correctness).
  EXL3_LIB=exl3xpu/_C_i8.so python3 tests/test_int8_prefill.py [M]
"""
import sys, os, json, time, collections
import torch
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exl3xpu import ops, triton_kernels as tk

MODEL = os.environ.get("MODEL", "/models/turboderp-Qwen3.8-27B-exl3-4.00bpw")
M = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
dev = torch.device("xpu:0")
E = ops._get_esimd()
qc = json.load(open(f"{MODEL}/quantization_config.json"))["tensor_storage"]
idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
H = {}


def get(n):
    f = idx[n]
    if f not in H:
        H[f] = safe_open(f"{MODEL}/{f}", "pt", device="cpu")
    return H[f].get_tensor(n)


FUSE = {"in_proj_qkv": "in_proj_qkvz", "in_proj_z": "in_proj_qkvz", "gate_proj": "gate_up", "up_proj": "gate_up",
        "q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv"}
groups = collections.OrderedDict()
for k, v in qc.items():
    if v.get("quant_format") != "exl3" or k == "lm_head":
        continue
    base, leaf = k.rsplit(".", 1)
    groups.setdefault((base, FUSE.get(leaf, leaf)), []).append(k)
# one layer of each type is enough: scale by counts
kinds = collections.OrderedDict()
for (base, name), keys in groups.items():
    kinds.setdefault(name, (keys, 0))
    kinds[name] = (kinds[name][0], kinds[name][1] + 1)


tot = collections.Counter()
Q8 = 3.453125 / 127
def timeit(fn, it=5):
    fn(); torch.xpu.synchronize(); t0 = time.perf_counter()
    for _ in range(it): fn()
    torch.xpu.synchronize(); return (time.perf_counter() - t0) / it
worst = 0.0
for name, (keys, count) in kinds.items():
    trs = [get(f"{k}.trellis") for k in keys]
    tr = torch.cat(trs, 1).to(dev)
    suh = torch.stack([get(f"{k}.suh") for k in keys]).to(dev)
    svh = torch.cat([get(f"{k}.svh") for k in keys]).to(dev)
    widths = [t.shape[1] * 16 for t in trs]
    bounds = [0]
    for w in widths: bounds.append(bounds[-1] + w)
    shard = torch.cat([torch.full((w // 128,), i, dtype=torch.int32) for i, w in enumerate(widths)]).to(dev)
    kdim, n = suh.shape[1], svh.shape[0]
    x = torch.randn(M, kdim, dtype=torch.float16, device=dev)
    lin = lambda: E.linear(x, tr, suh, svh, shard, bounds, 4, 2, 128, 16384)
    E.exl3_set_int8(0); y16 = lin(); t16 = timeit(lin)
    E.exl3_set_int8(1); y8 = lin(); t8 = timeit(lin)
    # torch reference of the same quantization
    xh = torch.empty((suh.shape[0], M, kdim), dtype=torch.float16, device=dev); E.exl3_had_in_rm(x, suh, xh)
    yr = torch.empty(M, n, dtype=torch.float16, device=dev)
    for g in range(len(bounds) - 1):
        g0, g1 = bounds[g], bounds[g + 1]
        W = torch.empty(kdim, g1 - g0, dtype=torch.float16, device=dev); E.exl3_reconstruct(tr, W, g0, 4, 2)
        Wq = torch.round(W.float() / Q8).to(torch.int8)
        sx = xh[g].float().abs().amax(1, keepdim=True) / 127
        xq = torch.round(xh[g].float() / sx).to(torch.int8)
        yr[:, g0:g1] = (torch._int_mm(xq, Wq).float() * sx * Q8).half()
    ref = torch.empty_like(y16); E.exl3_had_out_h(yr, svh, ref)
    e16 = ((y8.float() - y16.float()).norm() / y16.float().norm()).item()
    eref = ((y8.float() - ref.float()).norm() / ref.float().norm()).item()
    worst = max(worst, eref)
    tot["fp16"] += t16 * count; tot["int8"] += t8 * count
    print(f"{name:14s} x{count:2d}: fp16 {t16*1e3:6.2f} ms  int8 {t8*1e3:6.2f} ms ({t16/t8:.2f}x)  rel vs fp16 {e16:.2e}  "
          f"vs torch-ref {eref:.2e}", flush=True)
    del tr, x, y16, y8, xh, yr, ref; torch.xpu.empty_cache()
print(f"M={M}: all linears fp16 {tot['fp16']*1e3:.0f} ms -> int8 {tot['int8']*1e3:.0f} ms "
      f"({tot['fp16']/tot['int8']:.2f}x); kernel vs torch-ref worst {worst:.1e} {'PASS' if worst < 2e-3 else 'FAIL'}")
