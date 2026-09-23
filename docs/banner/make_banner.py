"""
Banner for the README / sharing: Qwen3.8-27B EXL3 on one Intel Arc Pro B70.
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
    if x["kind"] == "decode":
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

# ---- Intel Arc Pro B70 in pure white (blower card, side view), floating right
cx, cy, cw, ch = 1330, 88, 500, 158
g = [f'<g filter="url(#sh)">']
g.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="16" fill="#FFFFFF"/>')
# PCIe bracket (left) + screw tab
g.append(f'<rect x="{cx - 20}" y="{cy - 6}" width="14" height="{ch + 44}" rx="3" fill="#FFFFFF"/>')
g.append(f'<rect x="{cx - 34}" y="{cy - 6}" width="30" height="12" rx="3" fill="#FFFFFF"/>')
# PCIe edge connector (bottom)
g.append(f'<rect x="{cx + 70}" y="{cy + ch - 2}" width="250" height="20" rx="2" fill="#FFFFFF"/>')
g.append(f'<rect x="{cx + 128}" y="{cy + ch - 2}" width="6" height="20" fill="{BG}"/>')
# exhaust vents on the bracket
for k in range(7):
    g.append(f'<rect x="{cx - 17}" y="{cy + 14 + k * 20}" width="8" height="12" rx="2" fill="#F1EEE7"/>')
g.append("</g>")
out += g
# shroud details (still white, drawn with faint strokes)
st = 'stroke="#E6E2D9" stroke-width="2"'
nf = 'fill="none" ' + st
out.append(f'<rect x="{cx + 14}" y="{cy + 14}" width="{cw - 28}" height="{ch - 28}" rx="10" {nf}/>')
fx, fy, fr = cx + cw - 104, cy + ch / 2, 55
out.append(f'<circle cx="{fx}" cy="{fy}" r="{fr + 8}" fill="#FFFFFF" {st}/>')
out.append(f'<circle cx="{fx}" cy="{fy}" r="{fr}" {nf}/>')
for k in range(15):
    a = 2 * math.pi * k / 15
    x1, y1 = fx + 16 * math.cos(a), fy + 16 * math.sin(a)
    x2, y2 = fx + (fr - 4) * math.cos(a + 0.55), fy + (fr - 4) * math.sin(a + 0.55)
    out.append(f'<path d="M{x1:.1f},{y1:.1f} Q{fx + 44 * math.cos(a + 0.15):.1f},{fy + 44 * math.sin(a + 0.15):.1f} {x2:.1f},{y2:.1f}" {nf}/>')
out.append(f'<circle cx="{fx}" cy="{fy}" r="14" fill="#FFFFFF" {st}/>')
for k in range(9):
    y = cy + 34 + k * 13.5
    out.append(f'<line x1="{cx + 40}" y1="{y:.1f}" x2="{cx + cw - 205}" y2="{y:.1f}" stroke="#EEEBE4" stroke-width="2"/>')
out.append(f'<text x="{cx + 40}" y="{cy + ch - 26}" font-family="{FONT}" font-size="17" letter-spacing="3" fill="#CFCAC0">ARC PRO B70 · 32 GB</text>')

# ---- speed labels: one per concurrency column, on the top ridge (best class at that C)
top_vals, top_pts, _ = data_ridges[0]
for c in CS:
    x = XC[c]
    y = min(top_pts, key=lambda p: abs(p[0] - x))[1]
    best = max(series, key=lambda k: series[k][c])
    out.append(f'<line x1="{x}" y1="{y - 6:.1f}" x2="{x}" y2="{y - 38:.1f}" stroke="#B9B4AA" stroke-width="1.5"/>')
    out.append(f'<circle cx="{x}" cy="{y:.1f}" r="4" fill="#161512"/>')
    out.append(f'<text x="{x + 10}" y="{y - 46:.1f}" font-family="{FONT}" font-size="34" font-weight="500" fill="#161512">'
               f'{series[best][c]:.0f}<tspan dx="5" font-size="17" font-weight="400" fill="#77736B"> tok/s</tspan></text>')
    out.append(f'<text x="{x + 10}" y="{y - 20:.1f}" font-family="{FONT}" font-size="15" letter-spacing="2" fill="#8C877E">'
               f'C{c} · {best[0].upper()}{" · THINK" if best[1] else ""}</text>')
out.append(f'<text x="{W - 40}" y="{H - 22}" text-anchor="end" font-family="{FONT}" font-size="17" '
           f'fill="rgba(255,255,255,0.62)">ridges = measured aggregate decode tok/s, C1 to C16, prose/code × thinking on/off · MTP k=3 · one card</text>')

# ---- type
out.append(f'<text x="{W - 128}" y="50" text-anchor="end" font-family="{FONT}" font-size="22" fill="#9A968D">github.com/0xSero/exl3xpu</text>')
out.append(f'<text x="124" y="232" font-family="{FONT}" font-size="150" font-weight="500" letter-spacing="-5" fill="#161512">Qwen3.8-27B</text>')
out.append(f'<text x="130" y="306" font-family="{FONT}" font-size="44" fill="#2A2824">1× Intel Arc Pro B70 · EXL3 4.0bpw</text>')
out.append(f'<text x="130" y="352" font-family="{FONT}" font-size="27" fill="#77736B">'
           f'C1 {series[("code", False)][1]:.0f} tok/s · prefill {pre[4096]:,.0f} tok/s · 256K context · vision · bit-exact kernels</text>')
out.append("</svg>")
open(os.path.join(HERE, "banner.svg"), "w").write("\n".join(out))
print("wrote", os.path.join(HERE, "banner.svg"))
