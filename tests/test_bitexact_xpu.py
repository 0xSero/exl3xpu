"""
Gate A1: every XPU decode path is bit-exact with exllamav3.

1. ESIMD reconstruct vs exl3xpu.ref (itself bit-exact vs exllamav3 CUDA, tests/oracle_cuda.py) for
   EVERY exl3 tensor in the checkpoint, all columns.
2. GEMM paths (vector M=1/2/4, DPAS M=3/8/16/32/64): one-hot activation rows make each fp32 output an exact copy
   of one decoded weight, so out[m, :] must equal W_ref[k_m, :] bit for bit. Full k coverage for one
   tensor of each shape class; sampled k rows for the rest.

Usage: python3 tests/test_bitexact_xpu.py [--quick]
"""
import sys, os, json, time, random
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exl3xpu import ref, ops

MODEL = os.environ.get("MODEL", "/models/turboderp-Qwen3.8-27B-exl3-4.00bpw")
dev = torch.device("xpu:0")
E = ops._get_esimd()
assert E, "ESIMD ops not built"
quick = "--quick" in sys.argv
qc = json.load(open(f"{MODEL}/quantization_config.json"))["tensor_storage"]
idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
keys = [k for k, v in qc.items() if v.get("quant_format") == "exl3"]
if quick:
    keys = [k for k in keys if ".layers.0." in k or ".layers.3." in k or k == "lm_head"]

handles = {}


def get(name):
    fn = idx[name]
    if fn not in handles:
        handles[fn] = safe_open(f"{MODEL}/{fn}", "pt", device="cpu")
    return handles[fn].get_tensor(name)


SLICE = 16384
fails = []
t0 = time.time()
n_elems = 0
full_cover = {}
for i, key in enumerate(keys):
    tr = get(f"{key}.trellis").to(dev)
    K = tr.shape[-1] // 16
    cb = ref.codebook_of(f"{key}.mcg" in idx, f"{key}.mul1" in idx)
    k, n = tr.shape[0] * 16, tr.shape[1] * 16
    for n0 in range(0, n, SLICE):
        w = min(SLICE, n - n0)
        wx = torch.empty((k, w), dtype=torch.float16, device=dev)
        E.exl3_reconstruct(tr, wx, n0, K, cb)
        wr = ref.reconstruct_inner(tr[:, n0 // 16:(n0 + w) // 16].contiguous(), K, cb)
        if not torch.equal(wx.view(torch.int16), wr.view(torch.int16)):
            bad = (wx.view(torch.int16) != wr.view(torch.int16)).sum().item()
            fails.append(f"reconstruct {key} cols {n0}:{n0 + w}: {bad} mismatches")
        n_elems += k * w
        # GEMM paths via one-hot rows (on the first slice of each tensor)
        if n0 == 0:
            shape_class = (key.split(".")[-1], K)
            full = shape_class not in full_cover
            full_cover[shape_class] = True
            krows = list(range(k)) if (full and not quick) else random.Random(i).sample(range(k), 128)
            nn_ = w
            shard = torch.zeros(n // 128, dtype=torch.int32, device=dev)
            # full k coverage on the two widest paths; every production block size (vector M=1/2, DPAS
            # MB=8/16/32 incl. a partial block at M=3) on a 128-row sample
            sample = krows if len(krows) <= 128 else random.Random(i + 1).sample(krows, 128)
            for path, M, rows in [(0, 4, krows), (1, 64, krows), (0, 1, sample), (0, 2, sample), (1, 3, sample),
                                  (1, 8, sample), (1, 16, sample), (1, 24, sample), (1, 32, sample)]:
                for c in range(0, len(rows), M):
                    ks = rows[c:c + M]
                    Mi = len(ks)
                    xh = torch.zeros((1, k // 16, Mi, 16), dtype=torch.float16, device=dev)
                    for m, kk in enumerate(ks):
                        xh[0, kk // 16, m, kk % 16] = 1.0
                    part = E.exl3_gemm_raw(xh, tr, shard, n, K, cb, path)[:, :nn_]
                    exp = wr[ks].float()
                    if not torch.equal(part, exp):
                        bad = (part != exp).sum().item()
                        fails.append(f"{'vector' if path == 0 else 'dpas'} M={M} {key} rows {ks[:3]}...: {bad} mismatches")
                        break
    if i % 25 == 0 or i == len(keys) - 1:
        print(f"[{i + 1}/{len(keys)}] {key} K={K} {k}x{n}  elapsed {time.time() - t0:.0f}s  fails={len(fails)}", flush=True)

print(f"checked {len(keys)} tensors, {n_elems / 1e9:.2f}G weights, shape classes with full k coverage: {sorted(full_cover)}")
for f in fails[:20]:
    print("FAIL", f)
print("GATE_A1_PASS" if not fails else f"GATE_A1_FAIL ({len(fails)})")
