"""Context proof: a code word planted at several depths of a long random-word document must be recalled."""
import json, random, sys, urllib.request
sys.path.insert(0, "bench")
import sweep
base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8100"
ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 131072
rnd = random.Random(7)
ok = 0
for depth in [0.1, 0.5, 0.9]:
    code = f"{rnd.randint(10000, 99999)}"
    doc = sweep.unique_prompt(ctx, "prose", rnd).split("\n\nIgnore")[0]
    cut = int(len(doc) * depth)
    cut = doc.rfind(" ", 0, cut)
    text = doc[:cut] + f" The secret code is {code}. " + doc[cut:]
    q = text + "\n\nWhat is the secret code mentioned in the document above? Reply with the number only."
    body = {"model": "qwen3.8-27b-exl3", "messages": [{"role": "user", "content": q}], "max_tokens": 20,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    ans = r["choices"][0]["message"]["content"].strip()
    hit = code in ans
    ok += hit
    print(f"depth {depth}: prompt {r['usage']['prompt_tokens']} tokens, code {code}, answer {ans!r} {'OK' if hit else 'MISS'}", flush=True)
print("NEEDLE_PASS" if ok == 3 else f"NEEDLE_FAIL ({ok}/3)")
