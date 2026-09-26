"""Prefix-cache correctness on a GDN hybrid: 3 chained turns over one long document (turn 2 and 3 extend turn 1's
exact prefix), each asking for a different needle; greedy, thinking off, no output cap. Reports prompt/cached tokens,
TTFT and needle OK per turn, plus md5 of each answer so a no-cache server can be compared (greedy identity).

  python3 bench/pc_check.py --base http://127.0.0.1:8210 --ctx 16384 --seed s1
"""
import argparse, hashlib, json, os, random, string, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
import sweep


def ask(base, msgs):
    body = {"model": "qwen3.8-27b-exl3", "messages": msgs, "temperature": 0, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False}, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0, ttft, text, usage = time.time(), None, "", {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            d = json.loads(line[5:]); usage = d.get("usage") or usage
            for ch in d.get("choices", []):
                p = (ch.get("delta") or {}).get("content") or ""
                if p and ttft is None:
                    ttft = time.time() - t0
                text += p
    return text, ttft, usage


ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:8210")
ap.add_argument("--ctx", type=int, default=16384)
ap.add_argument("--seed", default="pc-v1")
a = ap.parse_args()
rnd = random.Random(a.seed)
codes = ["".join(rnd.choice(string.ascii_uppercase + string.digits) for _ in range(8)) for _ in range(3)]
names = ["amber", "cobalt", "saffron"]
words = sweep.unique_prompt(a.ctx, "prose", rnd).split(" ")
for i, (n, c) in enumerate(zip(names, codes)):
    pos = int(len(words) * (0.2 + 0.3 * i))
    words.insert(pos, f" IMPORTANT: the access code for vault {n} is {c}. ")
doc = " ".join(words)
msgs = [{"role": "user", "content": doc + f"\n\nWhat is the access code for vault {names[0]}? Reply with the code only."}]
ok_all = True
for turn in range(3):
    text, ttft, u = ask(a.base, msgs)
    ok = codes[turn] in text
    ok_all &= ok
    print(json.dumps({"turn": turn + 1, "prompt_tokens": u.get("prompt_tokens"),
                      "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                      "ttft_s": round(ttft, 2) if ttft else None, "ok": ok, "answer": text.strip()[:40],
                      "md5": hashlib.md5(text.encode()).hexdigest()[:10]}), flush=True)
    if turn < 2:
        msgs += [{"role": "assistant", "content": text},
                 {"role": "user", "content": f"Thanks. And the access code for vault {names[turn + 1]}? Code only."}]
print("PC_CHECK", "PASS" if ok_all else "FAIL")
