"""
Graft unquantized tensors (e.g. an MTP head or a vision tower) from a base checkpoint into an EXL3
checkpoint, producing a new model directory:
  - every file of the EXL3 dir is symlinked (relative links, so the dir works inside containers)
  - selected tensors are copied into one extra safetensors file
  - model.safetensors.index.json is merged

  python3 scripts/graft_tensors.py --exl3 DIR --base DIR --out DIR --prefix mtp. --prefix model.visual.
"""
import argparse, json, os
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--exl3", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--prefix", action="append", required=True, help="tensor name prefix to graft (repeatable)")
ap.add_argument("--name", default="model-grafted.safetensors")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
for f in os.listdir(args.exl3):
    if f in ("model.safetensors.index.json",) or f.startswith("."):
        continue
    dst = os.path.join(args.out, f)
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.relpath(os.path.join(args.exl3, f), args.out), dst)

base_map = json.load(open(os.path.join(args.base, "model.safetensors.index.json")))["weight_map"]
idx = json.load(open(os.path.join(args.exl3, "model.safetensors.index.json")))
want = sorted(k for k in base_map if any(k.startswith(p) for p in args.prefix))
clash = [k for k in want if k in idx["weight_map"]]
assert not clash, f"tensors already present in the EXL3 checkpoint: {clash[:5]}"
assert want, f"no tensors in {args.base} match {args.prefix}"
tensors = {}
by_file = {}
for k in want:
    by_file.setdefault(base_map[k], []).append(k)
for f, keys in by_file.items():
    with safe_open(os.path.join(args.base, f), "pt", device="cpu") as h:
        for k in keys:
            tensors[k] = h.get_tensor(k).contiguous()
save_file(tensors, os.path.join(args.out, args.name), metadata={"format": "pt"})

for k in tensors:
    idx["weight_map"][k] = args.name
json.dump(idx, open(os.path.join(args.out, "model.safetensors.index.json"), "w"), indent=1)
nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
print(f"grafted {len(tensors)} tensors ({nbytes / 2**30:.2f} GiB) from {args.base} into {args.out}")
