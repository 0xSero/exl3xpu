"""
Prefill (large-M) cost of all EXL3 linears for one chunk, split by stage:
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
tot = collections.Counter()
flops = 0
for name, (keys, count) in kinds.items():
    trs = [get(f"{k}.trellis") for k in keys]
    tr = torch.cat(trs, 1).to(dev)
    suh = torch.stack([get(f"{k}.suh") for k in keys]).to(dev)
    svh = torch.cat([get(f"{k}.svh") for k in keys]).to(dev)
    widths = [t.shape[1] * 16 for t in trs]
    bounds = [0]
    for w in widths:
        bounds.append(bounds[-1] + w)
    kdim, n = suh.shape[1], svh.shape[0]
    x = torch.randn(M, kdim, dtype=torch.float16, device=dev)
    out = torch.empty(M, n, dtype=torch.float16, device=dev)
    y = torch.empty(M, n, dtype=torch.float16, device=dev)
    wbuf = torch.empty(kdim * SL, dtype=torch.float16, device=dev)

    ESH = os.environ.get("TRITON_HAD") != "1"

    def st_had_in():
        if ESH:
            xh_ = torch.empty((suh.shape[0], M, kdim), dtype=torch.float16, device=dev)
            E.exl3_had_in_rm(x, suh, xh_)
            return xh_
        return tk.had_in(x, suh)
    xh = st_had_in()

    def st_recon_gemm(do_rec=True, do_mm=True):
        for g in range(len(bounds) - 1):
            for n0 in range(bounds[g], bounds[g + 1], SL):
                n1 = min(n0 + SL, bounds[g + 1])
                w = wbuf[: kdim * (n1 - n0)].view(kdim, n1 - n0)
                if do_rec:
                    E.exl3_reconstruct(tr, w, n0, 4, 2)
                if do_mm:
                    torch.matmul(xh[g], w, out=y[:, n0:n1])

    def st_had_out():
        if ESH:
            E.exl3_had_out_h(y, svh, out)
        else:
            tk.had_out(y, svh, out)

    def timeit(fn, it=5):
        fn(); torch.xpu.synchronize(); t0 = time.perf_counter()
        for _ in range(it):
            fn()
        torch.xpu.synchronize(); return (time.perf_counter() - t0) / it

    t = dict(had_in=timeit(st_had_in), recon=timeit(lambda: st_recon_gemm(True, False)),
             gemm=timeit(lambda: st_recon_gemm(False, True)), had_out=timeit(st_had_out))
    f = 2 * M * kdim * n
    flops += f * count
    for kk, v in t.items():
        tot[kk] += v * count
    print(f"{name:14s} x{count:2d} k={kdim:5d} n={n:5d}: " + "  ".join(f"{kk} {v*1e3:6.2f}ms" for kk, v in t.items())
          + f"   gemm {f / t['gemm'] / 1e12:5.1f} TFLOPS", flush=True)
    del tr, x, out, y, wbuf, xh
    torch.xpu.empty_cache()

T = sum(tot.values())
print(f"\nM={M}: linears {T*1e3:.0f} ms per chunk -> {M / T:.0f} tok/s ceiling (linears only); "
      + ", ".join(f"{k} {v*1e3:.0f}ms ({100*v/T:.0f}%)" for k, v in tot.items())
      + f"; GEMM {flops / tot['gemm'] / 1e12:.1f} TFLOPS")
