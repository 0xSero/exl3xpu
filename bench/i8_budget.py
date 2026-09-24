"""
int8 (W8A8) vs fp16 prefill GEMM on real reconstructed weights; was: Prefill (large-M) cost of all EXL3 linears for one chunk, split by stage:
had_in -> reconstruct W_inner slices -> oneDNN GEMM -> had_out.  Usage: python3 bench/prefill_budget.py [M]
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


SL = int(os.environ.get("EXL3_RECON_SLICE_N", "16384"))
tot = collections.Counter(); errs = []
def q_rows(a):                                     # per-row symmetric int8
    s = a.abs().amax(1, keepdim=True).float().clamp_min(1e-8) / 127
    return torch.round(a.float() / s).to(torch.int8), s
def timeit(fn, it=5):
    fn(); torch.xpu.synchronize(); t0 = time.perf_counter()
    for _ in range(it): fn()
    torch.xpu.synchronize(); return (time.perf_counter() - t0) / it
for name, (keys, count) in kinds.items():
    trs = [get(f"{k}.trellis") for k in keys]
    tr = torch.cat(trs, 1).to(dev)
    suh = torch.stack([get(f"{k}.suh") for k in keys]).to(dev)
    widths = [t.shape[1] * 16 for t in trs]
    bounds = [0]
    for w in widths: bounds.append(bounds[-1] + w)
    kdim = suh.shape[1]; n = bounds[-1]
    x = torch.randn(M, kdim, dtype=torch.float16, device=dev)
    xh = torch.empty((suh.shape[0], M, kdim), dtype=torch.float16, device=dev); E.exl3_had_in_rm(x, suh, xh)
    W = torch.empty(kdim, n, dtype=torch.float16, device=dev)
    for n0 in range(0, n, SL):
        n1 = min(n0 + SL, n); w = torch.empty(kdim, n1 - n0, dtype=torch.float16, device=dev)
        E.exl3_reconstruct(tr, w, n0, 4, 2); W[:, n0:n1] = w
    sw = W.abs().amax(0, keepdim=True).float() / 127                     # per output column
    Wq = torch.round(W.float() / sw).to(torch.int8)
    y16 = torch.empty(M, n, dtype=torch.float16, device=dev)
    def g16():
        for g in range(len(bounds) - 1):
            torch.matmul(xh[g], W[:, bounds[g]:bounds[g+1]], out=y16[:, bounds[g]:bounds[g+1]])
    xq = [q_rows(xh[g]) for g in range(len(bounds) - 1)]
    def g8():
        for g in range(len(bounds) - 1):
            torch._int_mm(xq[g][0], Wq[:, bounds[g]:bounds[g+1]])
    def qx():
        for g in range(len(bounds) - 1): q_rows(xh[g])
    t16, t8, tq = timeit(g16), timeit(g8), timeit(qx)
    g16()
    y8 = torch.cat([(torch._int_mm(xq[g][0], Wq[:, bounds[g]:bounds[g+1]]).float() * xq[g][1] * sw[:, bounds[g]:bounds[g+1]])
                    for g in range(len(bounds) - 1)], 1)
    rel = ((y8 - y16.float()).norm() / y16.float().norm()).item()
    wrel = ((Wq.float() * sw - W.float()).norm() / W.float().norm()).item()
    tot["fp16"] += t16 * count; tot["int8"] += t8 * count; tot["quant_x(torch)"] += tq * count
    print(f"{name:14s} x{count:2d} k={kdim:5d} n={n:5d}: fp16 {t16*1e3:6.2f} int8 {t8*1e3:6.2f} ms ({t16/t8:.2f}x)"
          f"  qx {tq*1e3:5.2f}  rel_err out {rel:.2e} w {wrel:.2e}  |W|max {W.abs().max().item():.3f}", flush=True)
    del tr, x, xh, W, Wq, y16, xq; torch.xpu.empty_cache()
print(f"M={M}: GEMM fp16 {tot['fp16']*1e3:.0f} ms, int8 {tot['int8']*1e3:.0f} ms, x-quant(unfused) {tot['quant_x(torch)']*1e3:.0f} ms")
