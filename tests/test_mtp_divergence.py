"""For prompts where MTP and non-MTP greedy outputs differ, show the target's top-2 logprob margin at the
first differing token (a near-tie => numerics flip, not a speculative-decoding bug)."""
import json, sys, urllib.request
from transformers import AutoTokenizer
A, B = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained("/models/turboderp-Qwen3.8-27B-exl3-4.00bpw")
PROMPTS = ["Write a short essay on lighthouses.",
           "Write a Python function that checks whether a string is a palindrome, with tests.",
           "Describe the water cycle for a 10-year-old."]

def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))

for p in PROMPTS:
    chat = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    outs = []
    for base in (A, B):
        r = post(base, "/v1/completions", {"model": "qwen3.8-27b-exl3", "prompt": chat, "max_tokens": 256,
                                           "temperature": 0})
        outs.append(tok(r["choices"][0]["text"], add_special_tokens=False)["input_ids"])
    i = 0
    while i < min(map(len, outs)) and outs[0][i] == outs[1][i]:
        i += 1
    if i == min(map(len, outs)):
        print(f"SAME  {p[:40]}"); continue
    prefix = chat + tok.decode(outs[1][:i])
    r = post(B, "/v1/completions", {"model": "qwen3.8-27b-exl3", "prompt": prefix, "max_tokens": 1,
                                    "temperature": 0, "logprobs": 5})
    top = r["choices"][0]["logprobs"]["top_logprobs"][0]
    lp = sorted(top.values(), reverse=True)
    print(f"DIFF at token {i}: mtp={tok.decode([outs[0][i]])!r} nomtp={tok.decode([outs[1][i]])!r} "
          f"top2 margin={lp[0] - lp[1]:.4f} nats  top={ {k: round(v, 3) for k, v in top.items()} }")
