"""
GPU-time budget of all EXL3 linears for one decode step (graph-captured, no host overhead).

Loads every quantized linear of the checkpoint grouped the way vLLM fuses them (qkvz, gate_up, qkv),
captures one forward's worth of linear calls at batch M into an XPU graph and times replays.
Usage: python3 bench/linear_budget.py [M ...]
"""
import sys, os, json, time, collections
import torch
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exl3xpu import ops

MODEL = os.environ.get("MODEL", "/models/turboderp-Qwen3.8-27B-exl3-4.00bpw")
dev = torch.device("xpu:0")
E = ops._get_esimd()
if os.environ.get("LOCAL"):
    E.exl3_set_local(int(os.environ["LOCAL"]))
if os.environ.get("VECMAX"):
    E.exl3_set_vec_max_m(int(os.environ["VECMAX"]))
if os.environ.get("TARGET"):
    E.exl3_set_target_threads(int(os.environ["TARGET"]))
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
    if v.get("quant_format") != "exl3":
        continue
    base, leaf = k.rsplit(".", 1) if "." in k else ("", k)
    g = (base, FUSE.get(leaf, leaf))
    groups.setdefault(g, []).append(k)

layers = []
for (base, name), keys in groups.items():
    trs = [get(f"{k}.trellis") for k in keys]
    tr = torch.cat(trs, dim=1).to(dev) if len(trs) > 1 else trs[0].to(dev)
    suh = torch.stack([get(f"{k}.suh") for k in keys]).to(dev)
    svh = torch.cat([get(f"{k}.svh") for k in keys]).to(dev)
    widths = [t.shape[1] * 16 for t in trs]
    shard = torch.cat([torch.full((w // 128,), i, dtype=torch.int32) for i, w in enumerate(widths)]).to(dev)
    K = tr.shape[-1] // 16
    layers.append(dict(name=name, tr=tr, suh=suh, svh=svh, shard=shard, K=K, k=suh.shape[1], n=svh.shape[0]))
wbytes = sum(l["tr"].numel() * 2 for l in layers)
print(f"{len(layers)} fused linears, {wbytes / 1e9:.2f} GB trellis")

for M in [int(a) for a in sys.argv[1:]] or [1, 2, 4, 8]:
    xs = {l["k"]: torch.randn(M, l["k"], dtype=torch.float16, device=dev) for l in layers}
    outs = [torch.empty(M, l["n"], dtype=torch.float16, device=dev) for l in layers]

    def run(sel=None):
        for l, o in zip(layers, outs):
            if sel is None or l["name"] == sel:
                E.exl3_gemm_small(xs[l["k"]], l["tr"], l["suh"], l["svh"], l["shard"], o, l["K"], 2)

    def timed(sel=None, iters=20):
        run(sel); torch.xpu.synchronize()
        g = torch.xpu.XPUGraph()
        with torch.xpu.graph(g):
            run(sel)
        g.replay(); torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            g.replay()
        torch.xpu.synchronize()
        return (time.perf_counter() - t0) / iters

    total = timed()
    parts = {nm: timed(nm, 10) for nm in sorted({l["name"] for l in layers})}
    print(f"M={M}: all linears {total * 1e3:.2f} ms ({wbytes / total / 1e9:.0f} GB/s) -> "
          f"{M / total:.1f} tok/s ceiling from linears alone")
    for nm, t in sorted(parts.items(), key=lambda x: -x[1]):
        b = sum(l["tr"].numel() * 2 for l in layers if l["name"] == nm)
        cnt = sum(1 for l in layers if l["name"] == nm)
        print(f"   {nm:14s} x{cnt:3d}  {t * 1e3:7.2f} ms  {b / t / 1e9:6.0f} GB/s")
