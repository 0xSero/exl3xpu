"""
Interleaved serving: decode streams running while new prompts arrive and prefill.

  background: N closed-loop streams (thinking on by default), each restarting with a fresh unique prompt
              when it finishes; their tokens inside the window give the decode rate under interference.
  arrivals  : one new request every --interval s, prompt sizes cycling through --sizes (tokens), a small
              max_tokens so they are mostly prefill; TTFT per arrival.

Reports: background aggregate + per-stream decode tok/s in the window, stall gaps (largest inter-token gaps
seen by the background streams), and arrival TTFT / effective prefill tok/s per prompt size.
Usage: python3 bench/interleave.py --base http://localhost:8101 --streams 4 --interval 10 --sizes 1024,8192,32768
"""
from __future__ import annotations
import argparse, asyncio, json, os, random, statistics, sys, time
import aiohttp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep import unique_prompt, stream_one  # noqa: E402


def pct(v, p):
    if not v:
        return None
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


async def background(args, session, url, idx, t_stop, recs):
    rnd = random.Random(1000 + idx)
    while time.time() < t_stop:
        rec = {"kind": "bg", "stream": idx}
        recs.append(rec)
        await stream_one(session, url, args.model, unique_prompt(0, args.cls, rnd), args.max_tokens,
                         0.0, not args.no_thinking, rec)


async def arrival(args, session, url, size, i, recs):
    rec = {"kind": "arrival", "size": size, "i": i}
    recs.append(rec)
    await stream_one(session, url, args.model, unique_prompt(size, "prose", random.Random(5000 + i)),
                     args.arrival_max_tokens, 0.0, not args.no_thinking, rec)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8101")
    ap.add_argument("--model", default="qwen3.8-27b-exl3")
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--cls", default="prose")
    ap.add_argument("--interval", type=float, default=10.0, help="seconds between arrivals; 0 = no arrivals")
    ap.add_argument("--sizes", default="1024,8192,32768")
    ap.add_argument("--warm", type=float, default=30.0)
    ap.add_argument("--window", type=float, default=150.0)
    ap.add_argument("--max-tokens", type=int, default=8192, help="runaway bound for background streams")
    ap.add_argument("--arrival-max-tokens", type=int, default=16)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--label", default="interleave")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    url = args.base + "/v1/chat/completions"
    sizes = [int(s) for s in args.sizes.split(",") if s]
    # build arrival prompts' tokenizer state up front (unique_prompt sizes with the tokenizer on first use)
    unique_prompt(64, "prose", random.Random(0))

    recs: list[dict] = []
    t0 = time.time()
    w0, w1 = t0 + args.warm, t0 + args.warm + args.window
    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=3600)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        tasks = [asyncio.create_task(background(args, session, url, i, w1, recs)) for i in range(args.streams)]
        arr_tasks = []
        if args.interval > 0:
            i = 0
            t_next = w0 - args.interval / 2          # first arrival lands just inside the window
            while t_next < w1 - 5:
                await asyncio.sleep(max(0.0, t_next - time.time()))
                arr_tasks.append(asyncio.create_task(arrival(args, session, url, sizes[i % len(sizes)], i, recs)))
                i += 1
                t_next += args.interval
        await asyncio.sleep(max(0.0, w1 - time.time()))
        # background loops stop starting new requests after w1; cancel in-flight ones, keep arrivals running
        for t in tasks:
            t.cancel()
        await asyncio.gather(*arr_tasks, return_exceptions=True)
        await asyncio.gather(*tasks, return_exceptions=True)

    bg = [r for r in recs if r["kind"] == "bg"]
    toks = [t for r in bg for t in r.get("times", []) if w0 <= t < w1]
    agg = len(toks) / args.window
    per_stream = []
    gaps = []
    for s in range(args.streams):
        ts = sorted(t for r in bg if r["stream"] == s for t in r.get("times", []) if w0 <= t < w1)
        per_stream.append(len(ts) / args.window)
        gaps += [b - a for a, b in zip(ts, ts[1:])]
    arr = [r for r in recs if r["kind"] == "arrival" and r.get("times")]
    by_size = {}
    for r in arr:
        ttft = r["times"][0] - r["t_send"]
        pt = r.get("prompt_tokens") or r["size"]
        by_size.setdefault(r["size"], []).append((ttft, pt / ttft))
    errs = [r.get("error") for r in recs if r.get("error")]

    row = {"kind": "interleave", "label": args.label, "streams": args.streams, "thinking": not args.no_thinking,
           "interval_s": args.interval, "sizes": sizes, "window_seconds": args.window,
           "bg_decode_tok_s_total": round(agg, 2), "bg_decode_tok_s_per_stream_mean": round(statistics.mean(per_stream), 2),
           "bg_gap_ms_p50": round(1000 * pct(gaps, 50), 1) if gaps else None,
           "bg_gap_ms_p99": round(1000 * pct(gaps, 99), 1) if gaps else None,
           "bg_gap_ms_max": round(1000 * max(gaps), 1) if gaps else None,
           "arrivals": {str(k): {"n": len(v), "ttft_s_p50": round(pct([a for a, _ in v], 50), 2),
                                 "ttft_s_max": round(max(a for a, _ in v), 2),
                                 "prefill_tok_s_p50": round(pct([b for _, b in v], 50), 1)} for k, v in sorted(by_size.items())},
           "errors": len(errs)}
    print(f"[{args.label}] {args.streams} streams, arrivals every {args.interval:g}s sizes {sizes}")
    print(f"  background decode: {row['bg_decode_tok_s_total']} tok/s total, {row['bg_decode_tok_s_per_stream_mean']} per stream;"
          f" token gaps p50 {row['bg_gap_ms_p50']} ms, p99 {row['bg_gap_ms_p99']} ms, max {row['bg_gap_ms_max']} ms")
    for k, v in row["arrivals"].items():
        print(f"  arrivals {k:>6} tok: n={v['n']}  TTFT p50 {v['ttft_s_p50']} s (max {v['ttft_s_max']} s)"
              f"  prefill {v['prefill_tok_s_p50']} tok/s")
    if errs:
        print(f"  errors: {len(errs)}: {errs[:2]}")
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
