"""Coherence ladder (inference-tuning-protocol gate 1/4): one cold unique prompt per context length, a fact buried at
a random depth, a control question at the end; the answer must contain the fact (OK) or the cell FAILs.
No output cap: requests end at EOS. Thinking off. Records TTFT (cold prefill) per rung.

  python3 bench/sgl_ladder.py --base http://127.0.0.1:8210 --ctx 1024,4096,8192,16384,32768,131072,250000
"""
import argparse, json, random, string, sys, time, urllib.request, os

sys.path.insert(0, os.path.dirname(__file__))
import sweep  # noqa: E402  (tokenizer-sized unique filler)


def ask(base, model, prompt, timeout=3600):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False}, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0, ttft, text, usage = time.time(), None, "", {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            for ch in d.get("choices", []):
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece and ttft is None:
                    ttft = time.time() - t0
                text += piece
    return text, ttft, time.time() - t0, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8210")
    ap.add_argument("--model", default="qwen3.8-27b-exl3")
    ap.add_argument("--ctx", default="1024,4096,8192,16384,32768")
    ap.add_argument("--seed", default=str(time.time()))
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    ok_all = True
    for ctx in [int(x) for x in a.ctx.split(",")]:
        code = "".join(rnd.choice(string.ascii_uppercase + string.digits) for _ in range(4)) + "-" + \
               "".join(rnd.choice(string.ascii_uppercase + string.digits) for _ in range(4))
        vault = rnd.choice(["amber", "cobalt", "saffron", "juniper", "obsidian", "vermilion"]) + f"-{rnd.randint(10, 99)}"
        fact = f" IMPORTANT: the access code for vault {vault} is {code}. "
        filler = sweep.unique_prompt(max(256, ctx - 200), "prose", rnd)
        words = filler.split(" ")
        pos = int(len(words) * rnd.uniform(0.1, 0.9))
        doc = " ".join(words[:pos]) + fact + " ".join(words[pos:])
        q = doc + f"\n\nQuestion: what is the access code for vault {vault}? Reply with the code only."
        text, ttft, total, usage = ask(a.base, a.model, q)
        ok = code in text
        ok_all &= ok
        pt = usage.get("prompt_tokens")
        row = dict(kind="ladder", context_tokens=ctx, prompt_tokens=pt, depth=round(pos / len(words), 2),
                   result="OK" if ok else "FAIL", answer=text.strip()[:80], expected=code,
                   ttft_s=round(ttft, 2) if ttft else None,
                   cold_prefill_tok_s=round(pt / ttft, 1) if pt and ttft else None, label=a.label)
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
    print("LADDER_" + ("PASS" if ok_all else "FAIL"))


if __name__ == "__main__":
    main()
