"""Builds article.html: plain wiki-style write-up of the EXL3-on-B70 port with inline SVG charts (data from docs/PROGRESS.md)."""
import html, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "article.html")


def esc(s):
    return html.escape(str(s))


def hbar(rows, unit, width=640, bar_h=22, gap=8, label_w=250, fmt="{:,.1f}", note=None, max_v=None):
    """Horizontal bar chart. rows: (label, value, css_class)."""
    max_v = max_v or max(v for _, v, _ in rows)
    plot_w = width - label_w - 70
    h = len(rows) * (bar_h + gap) + 10
    out = [f'<svg viewBox="0 0 {width} {h}" role="img" class="chart">']
    for i, (lab, v, cls) in enumerate(rows):
        y = 5 + i * (bar_h + gap)
        w = max(1.5, plot_w * v / max_v)
        out.append(f'<text x="{label_w - 8}" y="{y + bar_h * 0.7:.1f}" class="lbl" text-anchor="end">{esc(lab)}</text>')
        out.append(f'<rect x="{label_w}" y="{y}" width="{w:.1f}" height="{bar_h}" class="{cls}"/>')
        out.append(f'<text x="{label_w + w + 6:.1f}" y="{y + bar_h * 0.7:.1f}" class="val">{fmt.format(v)} {esc(unit)}</text>')
    out.append("</svg>")
    s = "\n".join(out)
    if note:
        s += f'<p class="cap">{note}</p>'
    return s


def grouped(cats, series, unit, width=640, height=260, fmt="{:,.0f}", note=None):
    """Vertical grouped bars. series: (name, css_class, [values per cat]) ; None = missing."""
    pad_l, pad_b, pad_t = 50, 40, 20
    plot_w, plot_h = width - pad_l - 10, height - pad_b - pad_t
    max_v = max(v for _, _, vals in series for v in vals if v is not None) * 1.12
    gw = plot_w / len(cats)
    bw = gw * 0.78 / len(series)
    out = [f'<svg viewBox="0 0 {width} {height + 24}" role="img" class="chart">']
    for t in range(0, 5):
        v = max_v * t / 4
        y = pad_t + plot_h - plot_h * t / 4
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - 10}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{pad_l - 6}" y="{y + 4:.1f}" class="tick" text-anchor="end">{v:,.0f}</text>')
    for ci, c in enumerate(cats):
        x0 = pad_l + ci * gw + gw * 0.11
        for si, (name, cls, vals) in enumerate(series):
            v = vals[ci]
            x = x0 + si * bw
            if v is None:
                out.append(f'<text x="{x + bw / 2:.1f}" y="{pad_t + plot_h - 4}" class="tick" text-anchor="middle">n/a</text>')
                continue
            bh = plot_h * v / max_v
            out.append(f'<rect x="{x:.1f}" y="{pad_t + plot_h - bh:.1f}" width="{bw - 2:.1f}" height="{bh:.1f}" class="{cls}"/>')
            out.append(f'<text x="{x + (bw - 2) / 2:.1f}" y="{pad_t + plot_h - bh - 4:.1f}" class="bv" text-anchor="middle">{fmt.format(v)}</text>')
        out.append(f'<text x="{pad_l + ci * gw + gw / 2:.1f}" y="{pad_t + plot_h + 18}" class="lbl" text-anchor="middle">{esc(c)}</text>')
    lx = pad_l
    for name, cls, _ in series:
        out.append(f'<rect x="{lx}" y="{height + 8}" width="12" height="12" class="{cls}"/>')
        out.append(f'<text x="{lx + 17}" y="{height + 18}" class="lbl">{esc(name)}</text>')
        lx += 17 + 8 * len(name) + 24
    out.append(f'<text x="8" y="{pad_t - 6}" class="tick">{esc(unit)}</text>')
    out.append("</svg>")
    s = "\n".join(out)
    if note:
        s += f'<p class="cap">{note}</p>'
    return s


def line(points_series, xlabels, unit, width=640, height=240, note=None):
    """Step/line chart over ordered steps. points_series: (name, css_class, [values])."""
    pad_l, pad_b, pad_t, pad_r = 50, 70, 20, 20
    plot_w, plot_h = width - pad_l - pad_r, height - pad_b - pad_t
    max_v = max(v for _, _, vals in points_series for v in vals if v is not None) * 1.1
    n = len(xlabels)
    X = lambda i: pad_l + plot_w * i / (n - 1)
    Y = lambda v: pad_t + plot_h - plot_h * v / max_v
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart">']
    for t in range(0, 5):
        v = max_v * t / 4
        out.append(f'<line x1="{pad_l}" y1="{Y(v):.1f}" x2="{width - pad_r}" y2="{Y(v):.1f}" class="grid"/>')
        out.append(f'<text x="{pad_l - 6}" y="{Y(v) + 4:.1f}" class="tick" text-anchor="end">{v:,.0f}</text>')
    for name, cls, vals in points_series:
        pts = [(X(i), Y(v)) for i, v in enumerate(vals) if v is not None]
        out.append('<polyline fill="none" class="ln ' + cls + '" points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '"/>')
        for (x, y), v in zip(pts, [v for v in vals if v is not None]):
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" class="dot {cls}"/>')
            out.append(f'<text x="{x:.1f}" y="{y - 8:.1f}" class="bv" text-anchor="middle">{v:g}</text>')
    for i, lab in enumerate(xlabels):
        out.append(f'<text transform="translate({X(i):.1f},{pad_t + plot_h + 14}) rotate(28)" class="tick">{esc(lab)}</text>')
    out.append(f'<text x="8" y="{pad_t - 6}" class="tick">{esc(unit)}</text>')
    out.append("</svg>")
    s = "\n".join(out)
    if note:
        s += f'<p class="cap">{note}</p>'
    return s


def stacked(parts, total_label, width=640, h=34, note=None):
    tot = sum(v for _, v, _ in parts)
    out = [f'<svg viewBox="0 0 {width} {h + 70}" role="img" class="chart">']
    x = 10
    pw = width - 20
    for lab, v, cls in parts:
        w = pw * v / tot
        out.append(f'<rect x="{x:.1f}" y="10" width="{w:.1f}" height="{h}" class="{cls}"/>')
        if w > 46:
            out.append(f'<text x="{x + w / 2:.1f}" y="{10 + h * 0.66:.1f}" class="inbar" text-anchor="middle">{v:g} ms</text>')
        x += w
    lx, ly = 10, h + 34
    for lab, v, cls in parts:
        out.append(f'<rect x="{lx}" y="{ly - 10}" width="12" height="12" class="{cls}"/>')
        out.append(f'<text x="{lx + 17}" y="{ly}" class="lbl">{esc(lab)} ({v:g})</text>')
        lx += 17 + 7 * (len(lab) + 6) + 14
        if lx > width - 150:
            lx, ly = 10, ly + 20
    out.append("</svg>")
    s = "\n".join(out)
    if note:
        s += f'<p class="cap">{note}</p>'
    return s


# ------------------------------------------------------------------------------------------------ charts
c_timeline = line(
    [("C1 prose, thinking off (greedy panel)", "s1", [7.3, 14.4, 28.4, 28.5, 37.9, 48.2, 53.9, 56.6, None, None]),
     ("C1 prose, thinking on", "s2", [None, None, None, None, None, None, 68.6, 70.3, 76.9, 91.2])],
    ["Triton eager", "ESIMD GEMV", "+ XPU graphs", "+ DPAS", "+ MTP k=3", "+ pruned draft", "split-K retune",
     "split-K MB16 + regs", "K=6 planar", "realistic corpus"],
    "tok/s",
    note="Single-stream (C1) decode, one B70, by milestone. Two panels are mixed on purpose: the early steps were measured "
         "thinking-off on synthetic prompts; from the split-K retune on, the headline is thinking-on. The last point is "
         "the realistic Gutenberg panel at temperature 0.7 (91.2 tok/s), which drafts better than random-word prompts.")

c_m1 = line(
    [("M=1 (vector GEMV)", "s1", [33.2, 32.5, 32.5, 29.2, 29.4, 27.3]),
     ("M=4 (MTP verify, DPAS)", "s2", [41.0, 45.1, 35.7, 35.2, 33.4, 31.7])],
    ["baseline", "dp4a 0x6400 trick", "split-K 1024", "prev-words in regs", "MB16 target 1408", "K=6 planar lm_head"],
    "ms for all 257 linears",
    note="Graph-captured time of every EXL3 linear in one forward pass (bench/linear_budget.py). M=4 is the MTP k=3 "
         "verify batch, i.e. what single-stream decode actually runs. The dp4a step briefly hurt M=4 until DPAS took over "
         "M&ge;3.")

c_m64 = hbar([
    ("MB=64, 1 tile/thread (original)", 124.0, "b3"),
    ("2 x MB=32 blocks", 100.0, "b3"),
    ("MB=64 NT=2, 256-GRF", 87.3, "b2"),
    ("+ split-K target 2048", 77.0, "b2"),
    ("+ prev-words from registers", 67.7, "b1"),
], "ms", note="The C16 problem: with MTP k=3 a 16-stream verify is M=64 rows. C16 was slower than C8 until the M=64 "
              "DPAS path was rebuilt (all linears, graph-captured).")

c_step = stacked([
    ("target EXL3 linears (M=4)", 31.7, "b1"),
    ("attention + GDN in graph", 3.6, "b2"),
    ("target lm_head", 1.8, "b4"),
    ("draft lm_head x3", 1.5, "b5"),
    ("draft layer", 1.3, "b6"),
    ("FA2 / argmax / misc", 0.8, "b3"),
], "ms", note="Where a single-stream thinking-on decode step goes (~41 ms per step, ~2.7-3.4 tokens accepted per step). "
               "The EXL3 linears are ~77% of the step, so the kernels are the whole game at low concurrency.")

c_vs = grouped(["C1", "C2", "C4", "C8", "C16"], [
    ("llama.cpp Q4_K_M", "b3", [25.0, None, None, 56.8, 56.0]),
    ("EXL3 (this work)", "b1", [91.2, 151.4, 262.6, 357.1, 365.2]),
], "aggregate tok/s, prose, thinking on",
    note="Same card. llama.cpp SYCL Q4_K_M is the tuned registry recipe (C1 25.0, C8 56.8, C16 56.0). EXL3 row is the "
         "realistic panel (Gutenberg prose, temperature 0.7). n/a = not measured for llama.cpp.")

c_live = grouped(["prose C1", "prose C2", "prose C4", "prose C8", "prose C16", "code C1", "code C2"], [
    ("live service (image 753922b1)", "b3", [69.1, 147.2, 264.3, 338.0, 151.8, 66.2, 120.3]),
    ("current canonical build", "b1", [89.9, 141.3, 262.2, 348.6, 361.0, 67.0, 120.5]),
], "aggregate tok/s",
    note="Same client, same prompts, run back to back on 2026-09-24 (temperature 0.7, thinking on). The live service "
         "queues at C16 (82 s to first token) because it predates the exact-KV-block fix. Code C4-C16 for the new build "
         "are missing: B70 #1 dropped off the PCIe bus during that cell (see Hardware).")

c_prefill = grouped(["4K", "32K", "128K", "254K"], [
    ("fp8 KV, stock FA2", "b3", [1665, 1363, 760, 508]),
    ("+ block-dequant fp8 prefill", "b2", [1612, 1531, 1059, 763]),
    ("+ 4096-token chunks (recipe)", "b1", [1654, 1483, 1020, 726]),
], "prefill tok/s (cold, one request)",
    note="Long-context prefill was attention-bound: XPU FA2 runs 69-74 TFLOPS on fp16 K/V but 39 on fp8. Dequantizing the "
         "cached K/V in 32K-key blocks and merging with log-sum-exp recovered fp16 speed. Chunk 4096 trades a little "
         "prefill for a 7.1 s &rarr; 3.7 s shorter decode freeze behind long prompts.")

c_draft = hbar([
    ("MTP k=3, pruned draft head (default)", 82.9, "b1"),
    ("DSpark k=7 (RadixArk)", 73.6, "b3"),
    ("MTP k=3, prose C1, earlier full-vocab head", 37.9, "b4"),
], "tok/s", note="Speculative drafts at single stream. Top two: realistic prose, temperature 1.0, same prompts "
                 "(2026-09-24). DSpark accepts more tokens per step (3.26 vs 2.85) but its draft pass costs more than it "
                 "saves, and its own KV cache caps context at 131K. The bottom bar is the first MTP result before the "
                 "draft lm_head was pruned to 512 of 1940 vocab blocks (thinking-off panel), shown for scale.", max_v=95)

# ------------------------------------------------------------------------------------------------ page
CSS = r"""
:root{--bg:#ffffff;--fg:#1f2328;--muted:#59636e;--rule:#d1d9e0;--code:#f6f8fa;--link:#0969da;
--c1:#1f6feb;--c2:#8250df;--c3:#9aa4ae;--c4:#bf8700;--c5:#1a7f37;--c6:#cf222e;--grid:#e6ebf0}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;--rule:#30363d;
--code:#161b22;--link:#4493f8;--c1:#4493f8;--c2:#ab7df8;--c3:#6e7681;--c4:#d29922;--c5:#3fb950;--c6:#f85149;--grid:#21262d}}
:root[data-theme="dark"]{--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;--rule:#30363d;--code:#161b22;--link:#4493f8;
--c1:#4493f8;--c2:#ab7df8;--c3:#6e7681;--c4:#d29922;--c5:#3fb950;--c6:#f85149;--grid:#21262d}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
main{max-width:860px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.9em;border-bottom:1px solid var(--rule);padding-bottom:.3em;margin:.2em 0 .4em;font-weight:600}
h2{font-size:1.4em;border-bottom:1px solid var(--rule);padding-bottom:.25em;margin-top:2em;font-weight:600}
h3{font-size:1.1em;margin-top:1.5em;font-weight:600}
a{color:var(--link)}
p.lede{color:var(--muted)}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.88em;background:var(--code);border-radius:4px}
code{padding:.1em .35em}
pre{padding:12px;overflow-x:auto;line-height:1.45}
table{border-collapse:collapse;width:100%;margin:1em 0;font-size:.92em;display:block;overflow-x:auto}
th,td{border:1px solid var(--rule);padding:5px 9px;text-align:left;vertical-align:top}
th{background:var(--code);font-weight:600}
td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.toc{border:1px solid var(--rule);background:var(--code);padding:10px 16px;display:inline-block;margin:.5em 0 1em}
.toc ol{margin:.3em 0;padding-left:1.3em}
.chart{width:100%;height:auto;display:block;margin:1em 0 .2em}
.chart .lbl{font-size:12px;fill:var(--fg)}
.chart .val,.chart .bv{font-size:11px;fill:var(--muted);font-variant-numeric:tabular-nums}
.chart .tick{font-size:10.5px;fill:var(--muted)}
.chart .inbar{font-size:11px;fill:#fff}
.chart .grid{stroke:var(--grid);stroke-width:1}
.chart .b1{fill:var(--c1)}.chart .b2{fill:var(--c2)}.chart .b3{fill:var(--c3)}.chart .b4{fill:var(--c4)}.chart .b5{fill:var(--c5)}.chart .b6{fill:var(--c6)}
.chart .ln{stroke-width:2.2}.chart .ln.s1{stroke:var(--c3)}.chart .ln.s2{stroke:var(--c1)}
.chart .dot.s1{fill:var(--c3)}.chart .dot.s2{fill:var(--c1)}
p.cap{font-size:.86em;color:var(--muted);margin:.2em 0 1.4em}
.legend2{font-size:.85em;color:var(--muted)}
.box{border-left:3px solid var(--rule);padding:.2em 0 .2em 12px;color:var(--muted)}
.ok{color:var(--c5)}.bad{color:var(--c6)}
.diagram text{font-size:12px;fill:var(--fg)}.diagram .bx{fill:var(--code);stroke:var(--rule)}.diagram .ar{stroke:var(--muted);stroke-width:1.4;fill:none;marker-end:url(#arw)}
.diagram .sub{font-size:10.5px;fill:var(--muted)}
"""

DIAGRAM = """
<svg viewBox="0 0 760 250" class="chart diagram" role="img">
<defs><marker id="arw" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker></defs>
<rect class="bx" x="10" y="20" width="150" height="62" rx="4"/><text x="85" y="44" text-anchor="middle">trellis words</text><text class="sub" x="85" y="62" text-anchor="middle">16x16 tile = 256*K bits</text><text class="sub" x="85" y="75" text-anchor="middle">(K=4: 32 uint32)</text>
<rect class="bx" x="200" y="20" width="150" height="62" rx="4"/><text x="275" y="44" text-anchor="middle">16-bit state</text><text class="sub" x="275" y="62" text-anchor="middle">window ending at bit (t+1)K</text><text class="sub" x="275" y="75" text-anchor="middle">shift / or / mask</text>
<rect class="bx" x="390" y="20" width="160" height="62" rx="4"/><text x="470" y="44" text-anchor="middle">mul1 codebook</text><text class="sub" x="470" y="62" text-anchor="middle">x = st * 0x83DCD12D</text><text class="sub" x="470" y="75" text-anchor="middle">dp4a: fp16(1024 + bytesum)</text>
<rect class="bx" x="590" y="20" width="160" height="62" rx="4"/><text x="670" y="44" text-anchor="middle">fp16 weight</text><text class="sub" x="670" y="62" text-anchor="middle">h*c1 + c2 (one hfma,</text><text class="sub" x="670" y="75" text-anchor="middle">bit-exact w/ exllamav3)</text>
<path class="ar" d="M160,51 L198,51"/><path class="ar" d="M350,51 L388,51"/><path class="ar" d="M550,51 L588,51"/>
<rect class="bx" x="120" y="150" width="200" height="70" rx="4"/><text x="220" y="174" text-anchor="middle">GemvKernel (M&#8804;2)</text><text class="sub" x="220" y="192" text-anchor="middle">vector FMA, split-K 1024 threads</text><text class="sub" x="220" y="207" text-anchor="middle">fp16 partial dots per tile-row</text>
<rect class="bx" x="440" y="150" width="220" height="70" rx="4"/><text x="550" y="174" text-anchor="middle">DpasKernel (M=3..64)</text><text class="sub" x="550" y="192" text-anchor="middle">decoded tile = one XMX B operand</text><text class="sub" x="550" y="207" text-anchor="middle">MB 8/16/24/32/40/48/64 row blocks</text>
<path class="ar" d="M670,82 L560,148"/><path class="ar" d="M670,82 L240,148"/>
<text class="sub" x="10" y="118">input Hadamard (suh) &#8594; kernel &#8594; split-K reduce + output Hadamard (svh)</text>
</svg>
<p class="cap">The per-weight path. Every weight is a pure function of a 16-bit window of the bitstream; the whole port is
about computing that function bit-exactly and fast enough on Xe2's vector ALUs, then feeding the XMX (DPAS) units.</p>
"""


def table(head, rows, num_cols=()):
    out = ["<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"]
    for r in rows:
        out.append("<tr>" + "".join(f'<td class="n">{c}</td>' if i in num_cols else f"<td>{c}</td>" for i, c in enumerate(r)) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


kept = table(["Step", "Effect (measured)", "Gate A1"], [
    ["ESIMD GEMV with dp4a byte-sum (replacing Triton)", "C1 7.3 &rarr; 14.4 tok/s", "bit-exact"],
    ["XPU decode graphs (FULL_DECODE_ONLY)", "C1 14.4 &rarr; 28.4 tok/s", "&ndash;"],
    ["DPAS (XMX) kernel, decoded tile as B operand, M&le;64", "C8 102 &rarr; 152, C64 411 tok/s", "bit-exact"],
    ["MTP k=3 using the EXL3 MTP head already in the checkpoint", "C1 prose 28 &rarr; 37.9", "&ndash;"],
    ["Pruned draft lm_head: 512 of 1940 vocab blocks (98.6% coverage)", "C1 prose 37.9 &rarr; 48.2, acceptance unchanged", "&ndash;"],
    ["dp4a 0x6400 trick + fp16 partial dots", "ISA 378 &rarr; 300 instr/row; M=1 33.2 &rarr; 32.5 ms", "bit-exact"],
    ["Split-K sized to 1024 threads (2048 at MB=64)", "M=16 49.3 &rarr; 39.1 ms, M=64 87.7 &rarr; 77.0 ms", "pass"],
    ["Previous trellis word built in registers (no p-1 reload)", "M=1 32.5 &rarr; 29.2, M=64 76.8 &rarr; 67.7 ms", "pass"],
    ["Split-K target 1408 for DPAS MB&le;16", "M=4 35.1 &rarr; 32.1 ms; C1 think-on prose 64.5 &rarr; 70.3", "pass"],
    ["K=6 planar word de-interleave (6-bit lm_head)", "lm_head M=1 4.05 &rarr; 1.72 ms (235 &rarr; 555 GB/s)", "bitwise identical"],
    ["MB=64 NT=2 256-GRF DPAS", "C16 cell +22%", "bit-exact"],
    ["MB=24 and MB=40/48 row blocks (no padding to 32/64)", "M=24 45.0 &rarr; 42.6, M=48 64.3 &rarr; 61.5 ms", "pass"],
    ["Block-dequant fp8 KV prefill attention (LSE merge)", "128K prefill 760 &rarr; 1059 tok/s", "rel err &le; 4e-4"],
    ["GDN metadata sync patch (no host&harr;device wait on masks)", "~1% per step", "&ndash;"],
    ["Exact 1600-token KV blocks (EXL3_KV_BLOCK_EXACT)", "C16 thinking-on prose 349 &rarr; 410 (+17%)", "&ndash;"],
    ["Prefill chunk 4096", "decode freeze behind long prompt 7.1 &rarr; 3.7 s; 128K stays &ge; 1000", "&ndash;"],
    ["Image size cap (4.2 MP) in the processor", "no more OUT_OF_RESOURCES on 16.7 MP images", "&ndash;"],
    ["Prefix caching explicitly on (vLLM leaves it off for GDN models)", "repeat 14.4K prompt: 11.1 &rarr; 1.8 s TTFT", "&ndash;"],
])

rejected = table(["Tried", "Result", "Why rejected"], [
    ["Software-pipelined trellis loads", "M=4 45 &rarr; 70 ms", "register spills"],
    ["Fused single-kernel linear (had_in + split-K + had_out)", "M=4 35.1 &rarr; 45.2 ms", "serialized split-K tail"],
    ["mul1 as two 16x16 multiplies (EXL3_MUL16)", "M=1 29.4 &rarr; 34.4 ms", "slower everywhere"],
    ["K=4 'halves' decode (EXL3_K4_HALVES)", "&minus;4% time", "<span class='bad'>wrong outputs</span> (caught by the extended Gate A1)"],
    ["DPAS double-buffered word prefetch", "M=16 39 &rarr; 86 ms", "spills"],
    ["LSC L1/L2 prefetch of trellis words", "&plusmn;noise", "no gain"],
    ["Separate B registers per tile (EXL3_BV_ALL)", "no change", "no gain"],
    ["DPAS MB=4 blocks", "within noise", "M=4 is decode-bound, not DPAS-bound"],
    ["Codebook-affine fold (EXL3_FOLD)", "&minus;3.5% at M&le;8", "not bit-exact by design + unexplained gate anomaly"],
    ["Custom ESIMD flash-attention (head_dim 256)", "48 TF vs FA2 61-75 TF", "8-row fp32 O fills half the GRF"],
    ["Piecewise / FULL_AND_PIECEWISE graphs; dynamic draft length", "UR OUT_OF_RESOURCES at capture", "Level Zero graph limits"],
    ["DSpark k=7 draft (thinking on)", "C1 prose 73.6 vs MTP 82.9", "draft pass too expensive; 131K context"],
    ["MTP k=4 / k=2 as default", "k=4 +8% C1 but &minus;9..15% C4; k=2 +11% C16 but loses C1-C8", "k=3 is the best all-round"],
    ["gpu_memory_utilization 0.970 / 0.985", "flat / refused at launch", "no gain"],
    ["Align-mode sync skip (patch_align_sync)", "correct, 337.7 vs 337.8 tok/s", "no gain; opt-in"],
])

gates = table(["Gate", "Result"], [
    ["A1 &mdash; weights bit-exact vs exllamav3 reconstruct, all 401 tensors, every kernel path (vector M=1/2/4, DPAS M=3..64)", "<span class='ok'>pass</span>"],
    ["A3 &mdash; end-to-end logits vs exllamav3 on an RTX 3090 (16,320 positions)", "<span class='ok'>top-1 99.63%, KL 9.8e-5 nats</span>"],
    ["T1 &mdash; C1 decode &ge; 50 tok/s thinking on", "<span class='ok'>91.2 prose / 66.1 code (realistic)</span>"],
    ["T2 &mdash; cold prefill &ge; 1000 tok/s at 4K/32K/128K", "<span class='ok'>1654 / 1483 / 1020</span> (prefix caching off); 923 at 128K with caching on"],
    ["T3 &mdash; 262,144 context on one card", "<span class='ok'>pass</span>, 272,570 KV tokens, needle 3/3 at 128K"],
    ["T5 &mdash; vision", "<span class='ok'>32 numbered images read back in order; video pass</span>"],
    ["T6 &mdash; MTP greedy identity", "5/8 exact; divergences are fp near-ties (0.016 nat)"],
    ["B &mdash; beat llama.cpp on the same card", "<span class='ok'>every cell</span> (C1 3.6x, C16 6.5x)"],
])

live = table(["Workload", "Live service", "Canonical build", ""], [
    ["prose C1 / C8 / C16 (tok/s)", "69.1 / 338.0 / 151.8", "89.9 / 348.6 / 361.0", "<span class='ok'>win</span>"],
    ["C16 time to first token", "82.2 s", "6.4 s", "<span class='ok'>win</span>"],
    ["worst decode freeze under mixed load", "7.1 s", "3.0 s", "<span class='ok'>win</span>"],
    ["cold prefill 4K / 32K / 128K (tok/s)", "1721 / 1522 / 1053", "1636 / 1423 / 923", "<span class='bad'>loss</span> (prefix-caching chunk rounding)"],
    ["1K prompt TTFT behind a 32K prefill", "5.4 s", "7.5 s", "<span class='bad'>loss</span> (same cause)"],
], num_cols=())

BODY = f"""
<main>
<h1>Porting EXL3 to Intel Arc: Qwen3.8-27B on one B70</h1>
<p class="lede">How exllamav3's trellis-quantized models were made to run on an Intel Arc Pro B70 (Battlemage, 32 GB) inside vLLM,
bit-exact with the CUDA reference, and then pushed from 7 to 91 tokens per second for a single stream. Written 24 September 2026.
Code: <a href="https://github.com/0xSero/exl3xpu">github.com/0xSero/exl3xpu</a>. Recipe:
<a href="https://github.com/0xSero/local-ai-registry/pull/95">local-ai-registry #95</a>.</p>

<div class="toc"><b>Contents</b><ol>
<li><a href="#result">Result</a></li><li><a href="#format">The EXL3 format</a></li><li><a href="#arch">Architecture of the port</a></li>
<li><a href="#kernels">Kernel work</a></li><li><a href="#serving">Serving work</a></li><li><a href="#drafts">Speculative decoding</a></li>
<li><a href="#correct">Correctness</a></li><li><a href="#live">Against the live service</a></li><li><a href="#alu">Why it is ALU-bound</a></li>
<li><a href="#rejected">What did not work</a></li><li><a href="#hw">Hardware finding</a></li><li><a href="#recipe">Canonical recipe</a></li><li><a href="#open">Open items</a></li>
</ol></div>

<h2 id="result">Result</h2>
<p>turboderp's <code>Qwen3.8-27B-exl3</code> at 4.00 bits per weight (14.9 GiB) serves on one B70 with a 262,144-token context,
16 concurrent sequences, MTP speculative decoding, an fp8 KV cache and image/video input. On a realistic workload (Gutenberg
prose, HumanEval code, temperature 0.7, thinking on) it decodes <b>91.2 tok/s</b> for one prose stream and <b>365 tok/s</b>
aggregate at 16 streams, and prefills a cold 128K prompt at ~1,020 tok/s. The tuned llama.cpp SYCL Q4_K_M recipe on the same
card does 25.0 and 56.0.</p>
{c_vs}
{c_timeline}

<h2 id="format">The EXL3 format</h2>
<p>EXL3 stores each linear layer as 16&times;16 tiles of K-bit codes (K=4 for this model's body, 6 for its lm_head). A tile is a
bitstream of 256&middot;K bits; weight <i>t</i> of the tile is decoded from the 16-bit window that ends at bit (t+1)&middot;K,
so neighbouring weights share bits &mdash; a trellis. The 16-bit state is mapped to a weight by a codebook; this checkpoint uses
<code>mul1</code>: multiply the state by <code>0x83DCD12D</code>, sum the four bytes of the product, and read
<code>1024 + bytesum</code> as an fp16 value that is scaled by one fused multiply-add. Inputs and outputs are rotated by
randomized Hadamard transforms (<code>suh</code>/<code>svh</code>), which is what makes such coarse codes accurate.</p>
{DIAGRAM}
<p>Two properties drive everything that follows: every weight costs a handful of integer instructions to decode, and there is
no shortcut &mdash; the exact fp16 values must match exllamav3's CUDA <code>decode_3inst</code> bit for bit.</p>

<h2 id="arch">Architecture of the port</h2>
<table><thead><tr><th>Piece</th><th>What it does</th></tr></thead><tbody>
<tr><td><code>exl3xpu/vllm_plugin.py</code></td><td>Registers an <code>exl3</code> quantization method with vLLM: detects EXL3 modules from
<code>.trellis</code> entries in the weight index (including the MTP head that the quantization config omits), fuses shards
(qkvz, gate_up), handles the 6-bit lm_head and the pruned draft head.</td></tr>
<tr><td><code>exl3xpu/ref.py</code></td><td>Pure-PyTorch decoder: the specification, validated bit-exact against exllamav3's CUDA kernels on a 3090.</td></tr>
<tr><td><code>csrc/exl3_esimd.h</code>, <code>csrc/exl3_ops.sycl</code></td><td>ESIMD kernels for Xe2: Hadamard in/out, the vector GEMV,
the DPAS GEMM, a reconstruct kernel for prefill (dequantize to fp16, then oneDNN GEMM), split-K reduction. One C++ custom op
per linear keeps host cost at ~34 &micro;s per call.</td></tr>
<tr><td><code>exl3xpu/vllm_patches.py</code></td><td>Source-checked fixes to vLLM internals that matter on XPU (host-sync removal in GDN
metadata, fp8-KV prefill attention, exact KV block size). Each patch verifies the exact source it expects and skips otherwise.</td></tr>
<tr><td><code>models/qwen3.8-27b-exl3-4.00bpw/</code></td><td><code>model.yaml</code> (the serving config) and <code>recipe.json</code> (what it measured).</td></tr>
</tbody></table>
<p>vLLM 0.26.1 XPU supplies everything else: scheduling, paged KV, the Gated DeltaNet (GDN) and attention kernels, the
OpenAI API, tool-call and reasoning parsers. The model is hybrid: 48 GDN linear-attention layers plus 16 full-attention
layers, which shapes the cache behaviour described below.</p>

<h2 id="kernels">Kernel work</h2>
<h3>Low concurrency: the vector GEMV and the M=4 verify</h3>
<p>The first working version was Triton: 7.3 tok/s. The ESIMD GEMV decodes the trellis with <code>dp4a</code> computing
<code>0x6400 + bytesum</code> in one instruction &mdash; its low 16 bits <i>are</i> the fp16 value 1024+bytesum &mdash; and
accumulates short fp16 partial dot products per tile-row before widening to fp32. With XPU graphs that reached 28 tok/s. From
there most gains came from not wasting work: sizing split-K to the machine instead of to 4096 threads, building each tile's
"previous word" by shifting registers instead of a second load, and de-interleaving the 6-bit lm_head's words once per tile so
the decoder reads contiguous planes (Xe regions only stride by powers of two; stride-3 gathers had cost two moves per value).</p>
{c_m1}
<h3>High concurrency: DPAS</h3>
<p>From M=3 rows up, a decoded 16&times;16 tile is written straight into the VNNI layout of an XMX <code>dpas</code> B operand,
so one decode feeds MB/8 matrix instructions. MTP k=3 turns 16 streams into a 64-row verify, which exposed that the original
MB=64 kernel (one tile per thread) was slower than two MB=32 passes. Rebuilding it with two tiles per thread in 256-register
mode, re-tuning split-K, and adding MB=24/40/48 blocks so odd batch sizes stop padding up, took M=64 from 124 to 68 ms.</p>
{c_m64}
<h3>Where a step goes</h3>
{c_step}

<h2 id="serving">Serving work</h2>
<ul>
<li><b>Prefill.</b> Long prompts dequantize each layer once (ESIMD reconstruct, row-major Hadamards: 7.7 &rarr; 4.3 s per 8K chunk
of linears) and use oneDNN GEMMs. At long context the bottleneck moved to attention on the fp8 KV cache.</li>
<li><b>fp8 KV at fp16 speed.</b> 256K only fits one card with fp8 KV, but vLLM's XPU flash-attention runs ~39 TFLOPS on fp8 against
69-74 on fp16. For single-sequence prefill chunks the patch gathers cached K/V in 32K-key blocks, dequantizes to fp16, runs the fp16
kernel per block and merges with log-sum-exp.</li>
<li><b>KV capacity.</b> vLLM XPU rounded the hybrid-negotiated 1600-token block (sized to one GDN state page) up to 2048 and padded
pages to 4 MiB. Keeping 1600 grew the pool to 272,570 tokens and let 13-14 of 16 long thinking streams stay resident: +17% at C16.</li>
<li><b>Interleaving.</b> Chunk size trades prefill speed against how long running streams freeze behind a long prompt: 8192-token
chunks froze decode for 7.1 s, 4096 for 3.7 s, while keeping 128K prefill at 1,020 tok/s.</li>
<li><b>Vision.</b> 32 images and 4 videos per request; the processor now caps images at 4.2 MP after a 16.7 MP image crashed the
vision encoder with an out-of-resources error.</li>
</ul>
{c_prefill}

<h2 id="drafts">Speculative decoding</h2>
<p>The EXL3 repository already ships Qwen's MTP head in 4-bit, so MTP needed no grafting. The draft head's lm_head is the expensive
part (it runs once per draft token), so it was pruned to the 512 of 1940 128-token vocabulary blocks that cover 98.6% of a text
corpus; the target still verifies with the full lm_head, so output is unchanged and acceptance did not move. k=3 is the best
all-round draft length. DSpark (RadixArk) and DFlash drafters were evaluated: DSpark accepts more tokens per step but loses on
wall time; a B70-tuned DSpark fine-tune (<code>rwmacy/qwen3.8-27b-dflash-drafter-fp8-b70</code>) and z-lab's single-pass
DFlash2 (~3 ms draft cost, would need a vLLM port) remain untested.</p>
{c_draft}

<h2 id="correct">Correctness</h2>
{gates}
<p>The bit-exactness gate earned its keep: a "halves" decode variant timed 4% faster but produced errors up to 120 in the output.
It passed the original gate, which only exercised two of the production code paths; the gate was extended to every path (vector
M=1/2/4, DPAS M=3/8/16/24/32/40/48/64) and caught it.</p>

<h2 id="live">Against the live service</h2>
<p>On 24 September the canonical build and the service then running on the other card (an older image) were driven with the same
client and prompts, one after the other.</p>
{c_live}
{live}
<p class="box">The two losses share one cause: with prefix caching on, vLLM's hybrid cache ("align" mode) rounds every prefill chunk
down to a multiple of the 1600-token block, so a 4096-token budget prefills 3,200 tokens per step. The fix &mdash; a 6,400-token budget
with long prompts capped at 4,800 per step, leaving room for short prompts &mdash; was ready but not measured before the card dropped.
The live service also had prefix caching off, as did every earlier recipe: vLLM 0.26 does not enable it for hybrid models.</p>

<h2 id="alu">Why it is ALU-bound</h2>
<p>Single-stream decode reads the weights at ~400 GB/s of the card's 608. The limit is arithmetic: decoding costs 6-9 vector
instructions per weight (extract the 16-bit window, multiply, byte-sum, scale), and at the card's integer throughput that is about
the whole budget. An ISA read of the DPAS loop shows ~9 SIMD32 instructions per 32-value group, near the integer roofline.
Plain INT4 needs 2-3 instructions per weight and saturates more of the bus, but reads ~40% more bytes, so per forward pass the two
are close &mdash; while EXL3 at 4.0 bpw is far closer to the bf16 model.</p>
<p>Two ways to cut the decode itself were identified: a 65,536-entry fp16 lookup table replacing multiply + byte-sum + scale (still
bit-exact, since the table is filled by the same function; built as <code>EXL3_LUT</code>, not yet measured), and re-laying the bitstream
at load so fewer windows straddle two words.</p>

<h2 id="rejected">What did not work</h2>
{rejected}

<h2 id="hw">Hardware finding</h2>
<p>Twice on 24 September the B70 under test fell off the PCIe bus mid-benchmark (<code>pciehp: Slot(19): Link Down / Card not
present</code>, then re-enumeration with a new render node). The journal shows five such link drops across both B70 slots in a week
and ~1,600 correctable data-link timeouts a day. Several earlier "device lost" crashes attributed to software were probably this.
It is a signal-integrity / link-power problem (riser, PCIe generation or ASPM), not the kernels; the card works normally after
re-enumeration.</p>

<h2 id="recipe">Canonical recipe</h2>
<p>The registry has one recommended, validated Qwen3.8-27B recipe for the B70:
<code>qwen38-27b-exl3-4bpw-arcb70-vllm-exl3xpu-tp1</code>. It now enables prefix caching.</p>
<pre>IMG=ghcr.io/0xsero/exl3xpu@sha256:86276b00c9f0161e7e7ccba1e683ca622679466a5c855e952b375ddbb6ec57c4
docker run --rm --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro --shm-size 32g -p 8000:8000 \\
  -e HF_HUB_OFFLINE=1 -v $MODELS/turboderp-Qwen3.8-27B-exl3-4.00bpw:/models:ro $IMG \\
  models/qwen3.8-27b-exl3-4.00bpw --gpu 0 --port 8000 --model-path /models \\
  -- --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3</pre>
<table><thead><tr><th>Setting</th><th>Value</th></tr></thead><tbody>
<tr><td>Weights</td><td>turboderp/Qwen3.8-27B-exl3 @ <code>113cf7ab</code>, 4.00 bpw (6-bit lm_head), 14.9 GiB</td></tr>
<tr><td>Engine</td><td>vLLM 0.26.1 XPU + exl3xpu (attested image above)</td></tr>
<tr><td>Speculative</td><td>MTP k=3, draft lm_head pruned to 512 vocab blocks</td></tr>
<tr><td>KV cache</td><td>fp8 e4m3, 272,570 tokens in 1600-token blocks, prefix caching on</td></tr>
<tr><td>Context / sequences</td><td>262,144 / 16</td></tr>
<tr><td>Prefill chunk</td><td>4096 tokens</td></tr>
<tr><td>Multimodal</td><td>32 images (&le; 4.2 MP) / 4 videos per request</td></tr>
</tbody></table>

<h2 id="open">Open items</h2>
<ul>
<li>Fix the B70 PCIe links (owner's hardware/BIOS decision), then re-accept the recipe with prefix caching on.</li>
<li>Scheduler fix for the prefix-caching prefill regression (6,400 budget, 4,800 long-prompt cap).</li>
<li>Measure the codebook lookup-table kernel (<code>EXL3_LUT</code>) against the current decode.</li>
<li>DFlash2 single-pass drafter port; the B70-tuned DSpark fine-tune.</li>
<li>6-bit / 4-bit EXL-style KV cache: ~2&times; cached tokens and room for a drafter at 256K.</li>
<li>256K prefill (726 tok/s) is bound by fp8 attention at head_dim 256.</li>
</ul>
<p class="cap">Every number above comes from <code>docs/PROGRESS.md</code> and <code>bench/results/</code> in the repository, where
each kept and rejected step is logged with its measurement.</p>
</main>
"""

page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EXL3 on Arc B70</title><style>{CSS}</style></head><body>{BODY}</body></html>"""
open(OUT, "w").write(page)
print("wrote", OUT, len(page), "bytes")
