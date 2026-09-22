"""
Saturation sweep for an OpenAI-compatible server (inference-tuning-protocol panel).

  decode : closed-loop C workers, each streaming unique cold prompts back to back for WARM+WINDOW s.
           aggregate decode = output tokens whose arrival falls inside the steady window / window.
           per-stream decode = (n_chunks-1)/(t_last - t_first) per request (median over requests).
  prefill: C concurrent unique cold prompts of L tokens, max_tokens=1 (streamed, TTFT);
           aggregate prefill = total prompt tokens / wall time of the wave; single-stream = L / TTFT.

GPU busy% sampled from xe sysfs gtidle residency during every cell (flag GPU_IDLE < 60%).
Output: JSONL rows in --out, one per cell, plus a printed table.
"""
from __future__ import annotations
import argparse, asyncio, glob, json, os, random, statistics, string, time
import aiohttp

WORDS = None


def words():
    global WORDS
    if WORDS is None:
        rnd = random.Random(1234)
        WORDS = ["".join(rnd.choice(string.ascii_lowercase) for _ in range(rnd.randint(3, 9))) for _ in range(20000)]
    return WORDS


TOPICS = ["the history of lighthouses", "how bridges are designed", "the economics of coffee farming",
          "the life cycle of stars", "medieval bookbinding", "tidal energy", "the migration of eels",
          "urban beekeeping", "the invention of the printing press", "volcanic soil and agriculture",
          "glassblowing", "the physics of sailing", "deep sea exploration", "the Silk Road",
          "clock making", "rice cultivation", "the chemistry of bread", "arctic navigation"]

_uid = 0


def unique_prompt(ctx_tokens: int, cls: str, rnd: random.Random) -> str:
    """Unique cold prefix (random word document ~ctx_tokens) + a task that ends naturally."""
    global _uid
    _uid += 1
    tag = f"[doc {os.getpid()}-{_uid}-{rnd.random():.12f}]"
    body = ""
    if ctx_tokens > 0:
        w = words()
        n = int(ctx_tokens / 1.9)  # ~1.9 tokens per random word (measured roughly)
        body = tag + " " + " ".join(rnd.choice(w) for _ in range(n)) + "\n\nIgnore the noise document above.\n"
    topic = rnd.choice(TOPICS)
    if cls == "code":
        task = (f"{tag} Write a complete, well commented Python module implementing an LRU cache with TTL expiry, "
                f"thread safety and unit tests. Name the class after {topic.split()[-1].capitalize()}Cache.")
    else:
        task = f"{tag} Write a detailed essay of about 600 words on {topic}. Use several paragraphs."
    return body + task


class GpuSampler:
    def __init__(self):
        self.paths = sorted(glob.glob("/sys/class/drm/card*/device/tile0/gt0/gtidle/idle_residency_ms"))
        self.devs = []
        for p in self.paths:
            dev = os.path.realpath(p.split("/tile0")[0])
            try:
                vid = open(os.path.join(dev, "device")).read().strip()
            except Exception:
                vid = ""
            if vid.lower() == "0xe223":
                self.devs.append(p)

    def read(self):
        out = []
        for p in self.devs:
            try:
                out.append(int(open(p).read().strip()))
            except Exception:
                out.append(None)
        return time.time(), out

    @staticmethod
    def busy(a, b):
        (t0, i0), (t1, i1) = a, b
        dt = (t1 - t0) * 1000
        return [None if x is None or y is None else round(100 * (1 - (y - x) / dt), 1) for x, y in zip(i0, i1)]


async def stream_one(session, url, model, prompt, max_tokens, temperature, thinking, rec):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": temperature, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": thinking}}
    if temperature > 0:
        body["top_p"] = 0.95
    t_send = time.time()
    rec.update(t_send=t_send, times=[], ok=False, prompt_tokens=None, completion_tokens=None)
    try:
        async with session.post(url, json=body) as r:
            if r.status != 200:
                rec["error"] = f"HTTP {r.status}: {(await r.text())[:200]}"
                return rec
            async for raw in r.content:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                d = json.loads(data)
                if d.get("usage"):
                    rec["prompt_tokens"] = d["usage"].get("prompt_tokens")
                    rec["completion_tokens"] = d["usage"].get("completion_tokens")
                for ch in d.get("choices", []):
                    delta = ch.get("delta", {})
                    if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                        rec["times"].append(time.time())
                    if ch.get("finish_reason"):
                        rec["finish"] = ch["finish_reason"]
        rec["ok"] = True
    except Exception as e:  # noqa
        rec["error"] = repr(e)[:200]
    rec["t_end"] = time.time()
    return rec


async def decode_cell(args, C, ctx, cls):
    url = f"{args.base}/v1/chat/completions"
    rnd = random.Random(hash((C, ctx, cls, time.time())))
    recs = []
    t0 = time.time()
    t_stop = t0 + args.warm + args.window
    sampler = GpuSampler()
    g_a = None

    async def worker(i):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as s:
            while time.time() < t_stop:
                rec = {"worker": i}
                recs.append(rec)
                await stream_one(s, url, args.model, unique_prompt(ctx, cls, rnd), args.max_tokens,
                                 args.temperature, args.thinking, rec)

    async def gpu_mark():
        nonlocal g_a
        await asyncio.sleep(args.warm)
        g_a = sampler.read()

    tasks = [asyncio.create_task(worker(i)) for i in range(C)] + [asyncio.create_task(gpu_mark())]
    await asyncio.sleep(args.warm + args.window)
    g_b = sampler.read()
    w0, w1 = t0 + args.warm, t0 + args.warm + args.window
    # stop: let in-flight requests finish only up to a grace period for per-stream stats
    done, pending = await asyncio.wait(tasks, timeout=args.grace)
    for p in pending:
        p.cancel()
    agg_tokens = sum(1 for r in recs for t in r.get("times", []) if w0 <= t < w1)
    per_stream = []
    ttfts = []
    outs = []
    fails = 0
    for r in recs:
        ts = r.get("times", [])
        if r.get("error"):
            fails += 1
        if len(ts) >= 16:
            per_stream.append((len(ts) - 1) / (ts[-1] - ts[0]))
            ttfts.append(ts[0] - r["t_send"])
            if r.get("ok"):
                outs.append(r.get("completion_tokens") or len(ts))
    busy = GpuSampler.busy(g_a, g_b) if g_a and sampler.devs else []
    flags = []
    if fails:
        flags.append("REQ_FAIL")
    if busy and max(b or 0 for b in busy) < 60:
        flags.append("GPU_IDLE")
    row = dict(kind="decode", concurrency=C, context_tokens=ctx, content_class=cls, thinking=args.thinking,
               temperature=args.temperature, cache_state="cold-unique",
               decode_tok_s_total=round(agg_tokens / args.window, 2),
               decode_tok_min_total=round(agg_tokens / args.window * 60),
               decode_tok_s_per_stream=round(statistics.median(per_stream), 2) if per_stream else None,
               ttft_ms_p50=round(1000 * statistics.median(ttfts)) if ttfts else None,
               output_tokens_mean=round(statistics.mean(outs)) if outs else None,
               samples=len(per_stream), window_seconds=args.window, gpu_busy_pct=busy, flags=flags,
               label=args.label)
    return row


async def prefill_cell(args, C, ctx):
    url = f"{args.base}/v1/chat/completions"
    rnd = random.Random(hash((C, ctx, time.time(), "p")))
    sampler = GpuSampler()
    rows = []
    for wave in range(args.prefill_waves):
        recs = [{} for _ in range(C)]
        g_a = sampler.read()
        t0 = time.time()
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as s:
            await asyncio.gather(*[stream_one(s, url, args.model, unique_prompt(ctx, "prose", rnd), 1, 0.0, False, r)
                                   for r in recs])
        wall = max(r.get("times", [r["t_end"]])[0] if r.get("times") else r["t_end"] for r in recs) - t0
        g_b = sampler.read()
        ptoks = sum(r.get("prompt_tokens") or 0 for r in recs)
        ttfts = sorted((r["times"][0] - r["t_send"]) for r in recs if r.get("times"))
        fails = sum(1 for r in recs if r.get("error") or not r.get("times"))
        rows.append((ptoks, wall, ttfts, fails, GpuSampler.busy(g_a, g_b) if sampler.devs else []))
    # discard first wave (warm-up) when more than one
    use = rows[1:] if len(rows) > 1 else rows
    ptoks = sum(r[0] for r in use)
    wall = sum(r[1] for r in use)
    ttfts = [t for r in use for t in r[2]]
    flags = []
    if any(r[3] for r in use):
        flags.append("REQ_FAIL")
    if ctx >= 100000 and ttfts and min(ttfts) < 1.0:
        flags.append("CACHE_HIT")
    return dict(kind="prefill", concurrency=C, context_tokens=ctx, cache_state="cold-unique",
                prompt_tokens_mean=round(ptoks / max(1, C * len(use))),
                prefill_tok_s_total=round(ptoks / wall, 1), prefill_tok_min_total=round(ptoks / wall * 60),
                ttft_ms_p50=round(1000 * statistics.median(ttfts)) if ttfts else None,
                ttft_ms_max=round(1000 * max(ttfts)) if ttfts else None,
                waves=len(use), gpu_busy_pct=use[-1][4], flags=flags, label=args.label)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8100")
    ap.add_argument("--model", default="qwen3.8-27b-exl3")
    ap.add_argument("--decode-c", default="1,2,4,8,16,32,64")
    ap.add_argument("--decode-ctx", default="0")
    ap.add_argument("--classes", default="prose")
    ap.add_argument("--prefill-c", default="1,4")
    ap.add_argument("--prefill-ctx", default="2048,8192,32768")
    ap.add_argument("--prefill-waves", type=int, default=2)
    ap.add_argument("--warm", type=float, default=15)
    ap.add_argument("--window", type=float, default=45)
    ap.add_argument("--grace", type=float, default=5)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="bench/results.jsonl")
    ap.add_argument("--skip-decode", action="store_true")
    ap.add_argument("--skip-prefill", action="store_true")
    args = ap.parse_args()

    rows = []
    # throwaway first cell after boot
    await decode_cell(argparse.Namespace(**{**vars(args), "warm": 2, "window": 8}), 1, 0, "prose")
    if not args.skip_prefill:
        for ctx in [int(x) for x in args.prefill_ctx.split(",") if x]:
            for C in [int(x) for x in args.prefill_c.split(",") if x]:
                row = await prefill_cell(args, C, ctx)
                rows.append(row); print(json.dumps(row), flush=True)
                open(args.out, "a").write(json.dumps(row) + "\n")
    if not args.skip_decode:
        for cls in args.classes.split(","):
            for ctx in [int(x) for x in args.decode_ctx.split(",") if x]:
                for C in [int(x) for x in args.decode_c.split(",") if x]:
                    row = await decode_cell(args, C, ctx, cls)
                    rows.append(row); print(json.dumps(row), flush=True)
                    open(args.out, "a").write(json.dumps(row) + "\n")
    print("\n%-8s %4s %6s %-6s %12s %12s %12s %9s %s" % ("kind", "C", "ctx", "class", "tok/s total", "tok/min", "per-stream", "ttft_ms", "flags"))
    for r in rows:
        tot = r.get("decode_tok_s_total", r.get("prefill_tok_s_total"))
        tpm = r.get("decode_tok_min_total", r.get("prefill_tok_min_total"))
        print("%-8s %4d %6d %-6s %12s %12s %12s %9s %s" % (r["kind"], r["concurrency"], r["context_tokens"],
              r.get("content_class", "-"), tot, tpm, r.get("decode_tok_s_per_stream", "-"), r.get("ttft_ms_p50"), ",".join(r["flags"])))


if __name__ == "__main__":
    asyncio.run(main())
