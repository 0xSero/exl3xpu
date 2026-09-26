"""Speculative acceptance probe, engine-agnostic (vLLM or SGLang --enable-metrics): N fixed real-corpus prompts sent one
at a time (no output cap, EOS-terminated), tokens-per-verify-step from /metrics counter deltas per request.
Same prompt list for both engines (seeded), so greedy runs must give identical texts if the drafts are identical.

  python3 bench/accept_probe.py --base http://127.0.0.1:8210 --n 6 --cls prose --thinking --temperature 0.0
"""
import argparse, asyncio, json, os, random, sys, time, hashlib
sys.path.insert(0, os.path.dirname(__file__))
import aiohttp
import sweep


async def one(base, model, prompt, temp, thinking):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temp,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    if temp > 0:
        body["top_p"] = 0.95
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as s:
        t = time.time()
        async with s.post(base + "/v1/chat/completions", json=body) as r:
            d = await r.json()
        return d, time.time() - t


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8210")
    ap.add_argument("--model", default="qwen3.8-27b-exl3")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--cls", default="prose")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", default="accept-v1")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sweep.CORPUS["mode"] = "real"
    rnd = random.Random(a.seed)
    prompts = [sweep.real_prompt(0, a.cls, rnd) for _ in range(a.n)]
    tot_tok = tot_step = 0.0
    for i, p in enumerate(prompts):
        m0 = await sweep.spec_counters(a.base)
        d, dt = await one(a.base, a.model, p, a.temperature, a.thinking)
        await asyncio.sleep(0.5)
        m1 = await sweep.spec_counters(a.base)
        msg = d["choices"][0]["message"]
        text = (msg.get("reasoning_content") or msg.get("reasoning") or "") + (msg.get("content") or "")
        ntok = d["usage"]["completion_tokens"]
        drafts = (m1[1] - m0[1]) if m0 and m1 else None
        acc = round(1 + (m1[0] - m0[0]) / drafts, 3) if drafts else None
        tot_tok += ntok
        tot_step += ntok / acc if acc else 0
        row = dict(label=a.label, i=i, cls=a.cls, thinking=a.thinking, t=a.temperature, tokens=ntok,
                   accept=acc, e2e_tok_s=round(ntok / dt, 1), text_md5=hashlib.md5(text.encode()).hexdigest()[:10],
                   finish=d["choices"][0]["finish_reason"])
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
    print(f"MEAN accept (token-weighted) {tot_tok / tot_step:.3f} over {int(tot_tok)} tokens" if tot_step else "no counters")


if __name__ == "__main__":
    asyncio.run(main())
