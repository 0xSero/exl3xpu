# prefix-cache proof: same 16K prompt twice; second TTFT must collapse, and prefix_cache_hits counter must move
import json, time, urllib.request, sys, random
base = sys.argv[1]
import glob; words = open(sorted(glob.glob("/w/bench/corpus/gutenberg/*"))[0], errors="ignore").read().split()
p = " ".join(words[5000:5000 + 12000]) + "\n\nSummarise the passage in one sentence."
def metric(name):
    t = urllib.request.urlopen(base + "/metrics", timeout=10).read().decode()
    return sum(float(l.split()[-1]) for l in t.splitlines() if l.startswith(name) and not l.startswith("#"))
for i in range(2):
    h0 = metric("vllm:prefix_cache_hits_total")
    body = json.dumps({"model": "qwen3.8-27b-exl3", "messages": [{"role": "user", "content": p}], "max_tokens": 2048,
                       "stream": True, "chat_template_kwargs": {"enable_thinking": True}}).encode()
    t0 = time.time(); r = urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=600)
    for line in r:
        if b'"content"' in line or b'"reasoning' in line: ttft = time.time() - t0; break
    r.close()
    print(f"request {i+1}: TTFT {ttft:.2f} s, prefix-cache hit tokens +{metric('vllm:prefix_cache_hits_total') - h0:.0f}")
