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
    """Unique cold prefix (random-word document of ~ctx_tokens tokens, sized with the real tokenizer)
    + a task that ends naturally. With --corpus real: Gutenberg / HumanEval / real source instead."""
    if CORPUS["mode"] == "real":
        return real_prompt(ctx_tokens, cls, rnd)
    global _uid
    _uid += 1
    tag = f"[doc {os.getpid()}-{_uid}-{rnd.random():.12f}]"
    body = ""
    if ctx_tokens > 0:
        w = words()
        n = max(1, int(ctx_tokens / 3.5))       # first guess (random words ~3.5 tokens each)
        doc = [rnd.choice(w) for _ in range(n)]
        for _ in range(3):                      # rescale with the tokenizer to land within ~1%
            got = count_tokens(" ".join(doc))
            if abs(got - ctx_tokens) <= ctx_tokens * 0.01:
                break
            n = max(1, int(len(doc) * ctx_tokens / got))
            doc = (doc + [rnd.choice(w) for _ in range(max(0, n - len(doc)))])[:n]
        body = tag + " " + " ".join(doc) + "\n\nIgnore the noise document above.\n"
    topic = rnd.choice(TOPICS)
    if cls == "code":
        task = (f"{tag} Write a complete, well commented Python module implementing an LRU cache with TTL expiry, "
                f"thread safety and unit tests. Name the class after {topic.split()[-1].capitalize()}Cache.")
    else:
        task = f"{tag} Write a detailed essay of about 600 words on {topic}. Use several paragraphs."
    return body + task


# --- real-text corpus (bench/fetch_corpus.sh): Gutenberg books for prose, HumanEval + real Python source for code
CORPUS = {"mode": "noise", "dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")}
_BOOKS = None
_HE = None
_SRC = None


def _books():
    global _BOOKS
    if _BOOKS is None:
        g = os.path.join(CORPUS["dir"], "gutenberg")
        _BOOKS = [open(os.path.join(g, f), encoding="utf-8", errors="ignore").read() for f in sorted(os.listdir(g))]
        _BOOKS = [b for b in _BOOKS if len(b) > 100000]
        assert _BOOKS, "no Gutenberg books: run bench/fetch_corpus.sh"
    return _BOOKS


def _humaneval():
    global _HE
    if _HE is None:
        _HE = [json.loads(l)["prompt"] for l in open(os.path.join(CORPUS["dir"], "humaneval.jsonl"))]
    return _HE


def _source():
    """Real Python source (the installed vLLM package), concatenated file by file."""
    global _SRC
    if _SRC is None:
        import vllm
        root = os.path.dirname(vllm.__file__)
        files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
        _SRC = [(os.path.relpath(f, root), open(f, encoding="utf-8", errors="ignore").read()) for f in files]
        _SRC = [(n, s) for n, s in _SRC if len(s) > 2000]
    return _SRC


def _take_tokens(text: str, start: int, ctx_tokens: int) -> str:
    """Slice of text from char offset start holding ~ctx_tokens tokens (sized with the tokenizer, ~1%)."""
    n = int(ctx_tokens * 4.2)
    for _ in range(4):
        s = text[start:start + n]
        got = count_tokens(s)
        if abs(got - ctx_tokens) <= ctx_tokens * 0.01 or start + n >= len(text):
            break
        n = max(1, int(n * ctx_tokens / got))
    return text[start:start + n]


def real_prompt(ctx_tokens: int, cls: str, rnd: random.Random) -> str:
    global _uid
    _uid += 1
    tag = f"[req {os.getpid()}-{_uid}-{rnd.random():.12f}]"
    if cls == "code":
        if ctx_tokens > 0:
            src = _source()
            i = rnd.randrange(len(src))
            parts, total = [], 0
            while total < ctx_tokens * 4.2 * 1.3:
                n, s = src[i % len(src)]
                parts.append(f"# ===== file: {n} =====\n{s}")
                total += len(s)
                i += 1
            body = _take_tokens("\n\n".join(parts), 0, ctx_tokens)
            return (f"{tag}\n{body}\n\nExplain what the code above does, then pick one function from it and "
                    f"rewrite it more clearly, with type hints and unit tests.")
        problem = rnd.choice(_humaneval())
        return (f"{tag} Complete the following Python function. Explain your approach, then give the full "
                f"implementation and unit tests.\n\n{problem}")
    book = rnd.choice(_books())
    if ctx_tokens > 0:
        start = rnd.randrange(0, max(1, len(book) - int(ctx_tokens * 4.5)))
        body = _take_tokens(book + "\n" + book, start, ctx_tokens)
        return f"{tag}\n{body}\n\nSummarize the passage above and discuss its main characters and themes in a detailed essay."
    start = rnd.randrange(0, len(book) - 4000)
    passage = book[start:start + 2000]
    return f"{tag} Read this passage and write a detailed essay of about 600 words analysing it.\n\n{passage}"


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


_TOK = None


def count_tokens(text: str) -> int:
    """Token count of a streamed delta when the server does not report per-chunk usage."""
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(os.environ.get("TOKENIZER", "/models/turboderp-Qwen3.8-27B-exl3-4.00bpw"))
    return max(1, len(_TOK(text, add_special_tokens=False)["input_ids"]))


def _auth_headers() -> dict:
    """Bearer token for gateways that require one: read from the file named by EXL3_API_KEY_FILE (never logged)."""
    f = os.environ.get("EXL3_API_KEY_FILE")
    if not f:
        return {}
    with open(f) as fh:
        return {"Authorization": "Bearer " + fh.read().strip()}


async def stream_one(session, url, model, prompt, max_tokens, temperature, thinking, rec):
    # times[i] is a token arrival: a chunk carrying n tokens (speculative decoding emits several per chunk)
    # appends its timestamp n times. n comes from vLLM's cumulative per-chunk usage when available,
    # otherwise from tokenizing the delta text.
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "chat_template_kwargs": {"enable_thinking": thinking}}
    if temperature > 0:
        body["top_p"] = 0.95
    t_send = time.time()
    rec.update(t_send=t_send, times=[], ok=False, prompt_tokens=None, completion_tokens=None)
    try:
        async with session.post(url, json=body, headers=_auth_headers()) as r:
            if r.status != 200:
                rec["error"] = f"HTTP {r.status}: {(await r.text())[:300]}"
                rec["t_end"] = time.time()
                return rec
            async for raw in r.content:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                d = json.loads(data)
                now = time.time()
                usage = d.get("usage")
                if usage:
                    rec["prompt_tokens"] = usage.get("prompt_tokens")
                    rec["completion_tokens"] = usage.get("completion_tokens")
                for ch in d.get("choices", []):
                    delta = ch.get("delta", {})
                    text = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                    if text:
                        rec["text"] = rec.get("text", "") + text
                        if usage and usage.get("completion_tokens") is not None:
                            n = usage["completion_tokens"] - rec.get("_seen", 0)
                            rec["_seen"] = usage["completion_tokens"]
                        else:
                            n = count_tokens(text)
                        rec["times"].extend([now] * max(1, n))
                    if ch.get("finish_reason"):
                        rec["finish"] = ch["finish_reason"]
        rec["ok"] = True
    except Exception as e:  # noqa
        rec["error"] = repr(e)[:200]
    rec["t_end"] = time.time()
    return rec


def loop_ratio(text: str, n: int = 32) -> float:
    """Share of 32-word shingles that already appeared earlier in the text: ~0 for normal prose/code,
    high when generation is stuck repeating (which speculative drafts accept trivially)."""
    w = text.split()
    if len(w) < 4 * n:
        return 0.0
    seen, dup = set(), 0
    for i in range(len(w) - n + 1):
        sh = tuple(w[i:i + n])
        dup += sh in seen
        seen.add(sh)
    return dup / (len(w) - n + 1)


async def spec_counters(base):
    """(accepted draft tokens, drafts) from vLLM's Prometheus metrics, or None."""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
            async with s.get(base + "/metrics", headers=_auth_headers()) as r:
                txt = await r.text()
    except Exception:
        return None
    acc = drafts = None
    for line in txt.splitlines():
        if line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            acc = float(line.split()[-1])
        elif line.startswith("vllm:spec_decode_num_drafts_total"):
            drafts = float(line.split()[-1])
    return (acc, drafts) if acc is not None and drafts is not None else None


async def decode_cell(args, C, ctx, cls):
    url = f"{args.base}/v1/chat/completions"
    # deterministic prompt sequence per cell (same text across variants/boots, still cold: server restarts)
    rnd = random.Random(f"{args.seed}-{C}-{ctx}-{cls}")
    m0 = await spec_counters(args.base)
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
            if ts[-1] > ts[0]:
                per_stream.append((len(ts) - ts.count(ts[0])) / (ts[-1] - ts[0]))
            ttfts.append(ts[0] - r["t_send"])
            if r.get("ok"):
                outs.append(r.get("completion_tokens") or len(ts))
    busy = GpuSampler.busy(g_a, g_b) if g_a and sampler.devs else []
    m1 = await spec_counters(args.base)
    accept_len = None
    if m0 and m1 and m1[1] > m0[1]:
        accept_len = round(1 + (m1[0] - m0[0]) / (m1[1] - m0[1]), 3)
    flags = []
    if fails:
        flags.append("REQ_FAIL")
    if busy and max(b or 0 for b in busy) < 60:
        flags.append("GPU_IDLE")
    loops = [round(loop_ratio(r.get("text", "")), 3) for r in recs]
    if any(x > 0.2 for x in loops):
        flags.append("LOOP")
    if getattr(args, "dump_text", None):
        with open(args.dump_text, "a") as f:
            for r in recs:
                f.write(json.dumps({"label": args.label, "C": C, "cls": cls, "finish": r.get("finish"),
                                    "completion_tokens": r.get("completion_tokens"),
                                    "loop_ratio": round(loop_ratio(r.get("text", "")), 3),
                                    "text": r.get("text", "")}) + "\n")
    row = dict(kind="decode", concurrency=C, context_tokens=ctx, content_class=cls, thinking=args.thinking,
               temperature=args.temperature, cache_state="cold-unique",
               decode_tok_s_total=round(agg_tokens / args.window, 2),
               decode_tok_min_total=round(agg_tokens / args.window * 60),
               decode_tok_s_per_stream=round(statistics.median(per_stream), 2) if per_stream else None,
               ttft_ms_p50=round(1000 * statistics.median(ttfts)) if ttfts else None,
               output_tokens_mean=round(statistics.mean(outs)) if outs else None,
               samples=len(per_stream), window_seconds=args.window, gpu_busy_pct=busy, flags=flags,
               spec_accept_len=accept_len, loop_ratio_max=max(loops) if loops else None,
               label=args.label, corpus=args.corpus)
    return row


async def prefill_cell(args, C, ctx):
    url = f"{args.base}/v1/chat/completions"
    rnd = random.Random(hash((C, ctx, time.time(), "p")))
    sampler = GpuSampler()
    rows = []
    for wave in range(args.prefill_waves):
        recs = [{} for _ in range(C)]
        prompts = [unique_prompt(ctx, "prose", rnd) for _ in range(C)]      # built (and tokenizer-sized) off the clock
        g_a = sampler.read()
        t0 = time.time()
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as s:
            await asyncio.gather(*[stream_one(s, url, args.model, pr, 1, 0.0, False, r)
                                   for pr, r in zip(prompts, recs)])
        wall = max(r.get("times", [r["t_end"]])[0] if r.get("times") else r["t_end"] for r in recs) - t0
        g_b = sampler.read()
        ptoks = sum(r.get("prompt_tokens") or 0 for r in recs)
        ttfts = sorted((r["times"][0] - r["t_send"]) for r in recs if r.get("times"))
        fails = sum(1 for r in recs if r.get("error") or not r.get("times"))
        for r in recs:
            if r.get("error"):
                print("prefill request failed:", r["error"], flush=True)
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
                waves=len(use), gpu_busy_pct=use[-1][4], flags=flags, label=args.label, corpus=args.corpus)


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
    ap.add_argument("--dump-text", default=None, help="append every response text + loop ratio to this jsonl")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--seed", default="panel-v1", help="prompt-sequence seed (same across variants)")
    ap.add_argument("--out", default="bench/results.jsonl")
    ap.add_argument("--skip-decode", action="store_true")
    ap.add_argument("--skip-prefill", action="store_true")
    ap.add_argument("--corpus", choices=["noise", "real"], default="noise",
                    help="noise: random-word documents + fixed tasks; real: Gutenberg / HumanEval / real source (bench/fetch_corpus.sh)")
    args = ap.parse_args()
    CORPUS["mode"] = args.corpus

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
    print("\n%-8s %4s %6s %-6s %12s %12s %12s %9s %7s %s" % ("kind", "C", "ctx", "class", "tok/s total", "tok/min", "per-stream", "ttft_ms", "accept", "flags"))
    for r in rows:
        tot = r.get("decode_tok_s_total", r.get("prefill_tok_s_total"))
        tpm = r.get("decode_tok_min_total", r.get("prefill_tok_min_total"))
        print("%-8s %4d %6d %-6s %12s %12s %12s %9s %7s %s" % (r["kind"], r["concurrency"], r["context_tokens"],
              r.get("content_class", "-"), tot, tpm, r.get("decode_tok_s_per_stream", "-"), r.get("ttft_ms_p50"),
              r.get("spec_accept_len", "-"), ",".join(r["flags"])))


if __name__ == "__main__":
    asyncio.run(main())
