"""
Greedy reproducibility probe: run fixed prompts at C1 (sequential), temperature 0, and save or compare texts.
  python3 tests/greedy_repro.py BASE save OUT.json      # record
  python3 tests/greedy_repro.py BASE compare REF.json   # exact-match rate + first divergence per prompt
Used to tell a behaviour change (patch) from numerics: run the reference config twice first.
"""
import json, sys, urllib.request

PROMPTS = ["Write a short essay on lighthouses.", "Explain how a hash map works, with a Python example.",
           "List the planets of the solar system with one fact each.", "Write a haiku about autumn, then explain it.",
           "What is 17*23? Show your working.", "Summarize the causes of World War I in five bullet points.",
           "Write a Python function that checks whether a string is a palindrome, with tests.",
           "Describe the water cycle for a 10-year-old."]


def gen(base, p):
    body = {"model": "qwen3.8-27b-exl3", "messages": [{"role": "user", "content": p}], "max_tokens": 2048,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": True}}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    m = json.load(urllib.request.urlopen(req, timeout=1800))["choices"][0]["message"]
    return (m.get("reasoning_content") or m.get("reasoning") or "") + "\n<ANSWER>\n" + (m.get("content") or "")


base, mode, path = sys.argv[1:4]
texts = [gen(base, p) for p in PROMPTS]
if mode == "save":
    json.dump(texts, open(path, "w"))
    print(f"saved {len(texts)} texts to {path}")
else:
    ref = json.load(open(path))
    exact = 0
    for i, (a, b) in enumerate(zip(texts, ref)):
        n = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
        exact += a == b
        print(f"prompt {i}: {'exact' if a == b else f'diverge at char {n} of {len(b)}'}")
    print(f"exact {exact}/{len(texts)}")
