"""
T6: greedy output with MTP speculative decoding vs without, same prompts.

Speculative decoding with greedy verification must reproduce the target model's greedy output. Our
target runs different kernels at M=1 (vector) and M=k+1 (DPAS), whose fp32 summation orders differ,
so near-ties can occasionally flip; we report exact-match rate and mean identical-prefix fraction.
Usage: python3 tests/test_mtp_identity.py http://host:port_mtp http://host:port_nomtp
"""
import json, sys, urllib.request

A, B = sys.argv[1], sys.argv[2]
PROMPTS = ["Write a short essay on lighthouses.", "Explain how a hash map works, with a Python example.",
           "List the planets of the solar system with one fact each.", "Write a haiku about autumn, then explain it.",
           "What is 17*23? Show your working.", "Summarize the causes of World War I in five bullet points.",
           "Write a Python function that checks whether a string is a palindrome, with tests.",
           "Describe the water cycle for a 10-year-old."]


def gen(base, p):
    body = {"model": "qwen3.8-27b-exl3", "messages": [{"role": "user", "content": p}], "max_tokens": 256,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))["choices"][0]["message"]["content"]


exact, fracs = 0, []
for p in PROMPTS:
    a, b = gen(A, p), gen(B, p)
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    fracs.append(n / max(len(a), len(b), 1))
    exact += a == b
    print(f"{'SAME' if a == b else 'DIFF'} prefix {fracs[-1]:.2f}  {p[:50]}")
print(f"exact {exact}/{len(PROMPTS)}, mean identical-prefix {sum(fracs) / len(fracs):.3f}")
