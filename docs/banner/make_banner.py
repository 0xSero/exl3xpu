"""
Banner for the README / sharing: Qwen3.8-27B EXL3 on one Intel Arc Pro B70 (logos: lobehub/icons qwen.svg, simple-icons intel.svg).
The ridges ARE the measured sweep: each data ridge passes through the aggregate decode tok/s at C=1,2,4,8,16
(models/qwen3.8-27b-exl3-4.00bpw/recipe.json); the layers between them are interpolated for the look.
Usage: python3 docs/banner/make_banner.py && rsvg-convert docs/banner/banner.svg -o docs/banner/banner.png
"""
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
W, H = 2000, 669
BG = "#F9F7F1"
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

rec = json.load(open(os.path.join(ROOT, "models/qwen3.8-27b-exl3-4.00bpw/recipe.json")))
CS = [1, 2, 4, 8, 16]
series = {}
for x in rec["speed_sweep"]:
    if x["kind"] == "decode" and not x.get("context_tokens") and x.get("corpus", "noise") == "real":
        series.setdefault((x["content_class"], x["thinking"]), {})[x["concurrency"]] = x["decode_tok_s_total"]
pre = {x["context_tokens"]: x["prefill_tok_s_total"] for x in rec["speed_sweep"] if x["kind"] == "prefill"}

# data ridges back (light) to front (dark), by mean speed
order = sorted(series, key=lambda k: -sum(series[k].values()))
XC = {c: 180 + i * 400 for i, c in enumerate(CS)}      # column x per concurrency
BASE = H - 62                                         # ridge floor (tok/s = 0)
SCALE = 262 / 344.0                                   # px per tok/s


def ridge_pts(vals, wob, phase):
    """Smooth ridge through (XC[c], height) with an organic wobble; extended past both edges."""
    ctrl = [(-60, BASE - vals[1] * SCALE * 0.55)] + [(XC[c], BASE - vals[c] * SCALE) for c in CS] \
        + [(W + 60, BASE - vals[16] * SCALE * 1.02)]
    pts = []
    for i in range(len(ctrl) - 1):
        p0 = ctrl[max(i - 1, 0)]; p1 = ctrl[i]; p2 = ctrl[i + 1]; p3 = ctrl[min(i + 2, len(ctrl) - 1)]
        for s in range(24):
            t = s / 24
            t2, t3 = t * t, t * t * t
            x = 0.5 * (2 * p1[0] + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                       + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * (2 * p1[1] + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                       + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            y += wob * math.sin(x / 230.0 + phase) + wob * 0.45 * math.sin(x / 83.0 + phase * 1.7)
            pts.append((x, y))
    pts.append(ctrl[-1])
    return pts


def path(pts):
    d = f"M{pts[0][0]:.1f},{H + 10} " + " ".join(f"L{x:.1f},{y:.1f}" for x, y in pts) + f" L{pts[-1][0]:.1f},{H + 10} Z"
    return d


# layers: a faint far ridge, then between consecutive data ridges 3 interpolated layers, then a floor
layers = []
far = {c: series[order[0]][c] * 1.08 + 18 for c in CS}
chain = [far] + [series[k] for k in order] + [{c: series[order[-1]][c] * 0.2 - 70 for c in CS}]
for a, b in zip(chain, chain[1:]):
    for j in range(6):
        f = j / 6
        layers.append(({c: a[c] * (1 - f) + b[c] * f for c in CS}, j == 0 and a is not far and a is not chain[-1]))
layers.append((chain[-1], False))

n = len(layers)
out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
       "<defs>",
       '<filter id="sh" x="-20%" y="-20%" width="140%" height="160%"><feGaussianBlur in="SourceAlpha" stdDeviation="14"/>'
       '<feOffset dy="16"/><feComponentTransfer><feFuncA type="linear" slope="0.22"/></feComponentTransfer>'
       '<feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
       "</defs>",
       f'<rect width="{W}" height="{H}" fill="{BG}"/>']

# ---- ridges
data_ridges = []
for i, (vals, is_data) in enumerate(layers):
    f = i / (n - 1)
    q = f ** 1.35                          # grade slowly: most layers stay in the light/mid greys
    r = int(236 - q * 200); g = int(232 - q * 198); b = int(224 - q * 193)
    pts = ridge_pts(vals, 6 + 7 * f, i * 0.55)
    out.append(f'<path d="{path(pts)}" fill="rgb({r},{g},{b})" stroke="rgba(0,0,0,0.07)" stroke-width="1"/>')
    if is_data:
        data_ridges.append((vals, pts, f))

# ---- three labels on the top ridge: C1 decode, C2 decode, prefill (placed on the C8 peak)
top_vals, top_pts, _ = data_ridges[0]
best1 = max(series[k][1] for k in series); best2 = max(series[k][2] for k in series)
for x, val, tag in ((XC[1], best1, "C1 DECODE"), (XC[2], best2, "C2 DECODE"), (XC[8], pre[4096], "PREFILL")):
    y = min(top_pts, key=lambda p: abs(p[0] - x))[1]
    out.append(f'<line x1="{x}" y1="{y - 6:.1f}" x2="{x}" y2="{y - 38:.1f}" stroke="#B9B4AA" stroke-width="1.5"/>')
    out.append(f'<circle cx="{x}" cy="{y:.1f}" r="4" fill="#161512"/>')
    out.append(f'<text x="{x + 10}" y="{y - 46:.1f}" font-family="{FONT}" font-size="40" font-weight="500" fill="#161512">'
               f'{val:,.0f}<tspan dx="6" font-size="19" font-weight="400" fill="#77736B">tok/s</tspan></text>')
    out.append(f'<text x="{x + 10}" y="{y - 18:.1f}" font-family="{FONT}" font-size="16" letter-spacing="2.5" fill="#8C877E">{tag}</text>')

# ---- logos only: Qwen mark (lobehub/icons qwen.svg) | Intel wordmark (simple-icons intel.svg)
import re
def paths(f):
    return " ".join(re.findall(r'<path[^>]* d="([^"]+)"', open(os.path.join(HERE, f)).read()))
INK = "#161512"
out.append(f'<path transform="translate(130,96) scale({170 / 24:.4f})" fill="{INK}" fill-rule="evenodd" d="{paths("qwen-logo.svg")}"/>')
out.append(f'<line x1="358" y1="116" x2="358" y2="246" stroke="#C9C4BA" stroke-width="2"/>')
# intel wordmark spans y 7.3..16.5 of its 24-unit box: scale so the glyphs are ~120 px tall, centred on the mark
IS = 13.5
out.append(f'<path transform="translate(400,{181 - 11.9 * IS:.1f}) scale({IS})" fill="{INK}" d="{paths("intel-logo.svg")}"/>')
out.append("</svg>")
open(os.path.join(HERE, "banner.svg"), "w").write("\n".join(out))
print("wrote", os.path.join(HERE, "banner.svg"))
