"""
Long-context single-stream panel: for each context N, one cold request (unique N-token document + essay task),
then a second request on the same document with a different task (prefix-cache hit, the multi-turn case).
Reports cold prefill tok/s, warm TTFT, and per-stream decode tok/s after the first token (thinking on).

  python3 bench/longctx.py --base http://localhost:8101 --ctx 131072,200000 [--out bench/results/x.jsonl]
"""
import argparse, asyncio, json, random, sys, os, time
import aiohttp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep  # noqa: E402


def summarize(rec, n_ctx, phase, label):
    t = rec.get("times") or []
    ttft = (t[0] - rec["t_send"]) if t else None
    dec = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 2 and t[-1] > t[0] else None
    row = dict(kind="longctx", phase=phase, context_tokens=n_ctx, prompt_tokens=rec.get("prompt_tokens"),
               ttft_ms=round(ttft * 1000) if ttft else None,
               prefill_tok_s=round(rec["prompt_tokens"] / ttft, 1) if ttft and rec.get("prompt_tokens") else None,
               decode_tok_s=round(dec, 2) if dec else None, output_tokens=rec.get("completion_tokens"),
               finish=rec.get("finish"), loop_ratio=round(sweep.loop_ratio(rec.get("text", "")), 3),
               error=rec.get("error"), label=label, thinking=True, temperature=0.0)
    return row


async def main(a):
    rnd = random.Random(a.seed)
    url = a.base.rstrip("/") + "/v1/chat/completions"
    tmo = aiohttp.ClientTimeout(total=None, sock_read=3600)
    async with aiohttp.ClientSession(timeout=tmo) as s:
        for n in [int(x) for x in a.ctx.split(",")]:
            p = sweep.unique_prompt(n, "prose", rnd)
            doc = p[: p.rindex("\n\nIgnore the noise document above.\n") + len("\n\nIgnore the noise document above.\n")]
            tasks = [("cold", p),
                     ("warm", doc + "Write a detailed essay of about 600 words on the history of lighthouses. "
                                    "Use several paragraphs.")]
            for phase, prompt in tasks:
                rec = {}
                await sweep.stream_one(s, url, a.model, prompt, a.max_tokens, 0.0, True, rec)
                row = summarize(rec, n, phase, a.label)
                print(json.dumps(row), flush=True)
                with open(a.out, "a") as f:
                    f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8101")
    ap.add_argument("--model", default="qwen3.8-27b-exl3")
    ap.add_argument("--ctx", default="200000")
    ap.add_argument("--max-tokens", type=int, default=16384, help="runaway bound only")
    ap.add_argument("--seed", default="longctx-v1")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="bench/results/longctx.jsonl")
    asyncio.run(main(ap.parse_args()))
