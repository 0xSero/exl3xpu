"""Aggregate device-kernel time from a torch profiler chrome trace (json or json.gz): top kernels by total time,
grouped into families (EXL3 linears, attention, GDN, norms, ...), per profiled step."""
import collections, gzip, json, re, sys
path = sys.argv[1]; steps = float(sys.argv[2]) if len(sys.argv) > 2 else 1
d = json.load(gzip.open(path) if path.endswith(".gz") else open(path))
ev = d["traceEvents"] if isinstance(d, dict) else d
tot = collections.Counter(); cnt = collections.Counter()
for e in ev:
    if e.get("ph") != "X":
        continue
    cat = (e.get("cat") or "").lower()
    if cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
        continue
    n = e.get("name", "")
    tot[n] += e.get("dur", 0); cnt[n] += 1
fam = collections.Counter()
rules = [("exl3", r"Exl3|exl3|Dpas|Gemv|HadIn|HadOut|Recon|Vec|dp4a"), ("attention", r"flash|fmha|sdpa|attn|Attn|paged"),
         ("gdn/fla", r"delta|gdn|gated|fla|conv1d|causal_conv|recurrent|chunk"), ("norm", r"norm|Norm"),
         ("rope", r"rope|rotary"), ("copy/fill", r"copy|Copy|fill|Fill|memcpy|memset|cat|index|scatter|gather"),
         ("elementwise", r"elementwise|vectorized|Elementwise|silu|sigmoid|softplus|mul|add")]
for n, t in tot.items():
    for f, rx in rules:
        if re.search(rx, n):
            fam[f] += t; break
    else:
        fam["other"] += t
all_t = sum(tot.values())
print(f"device time per step: {all_t/steps/1000:.2f} ms over {steps:g} steps")
for f, t in fam.most_common():
    print(f"  {f:12s} {t/steps/1000:7.2f} ms  {100*t/all_t:5.1f}%")
print("top kernels:")
for n, t in tot.most_common(25):
    print(f"  {t/steps/1000:7.3f} ms  x{cnt[n]/steps:6.1f}  {n[:110]}")
