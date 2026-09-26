import json, sys, time, urllib.request
B = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8210"
def post(path, body):
    r = urllib.request.Request(B + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time(); d = json.load(urllib.request.urlopen(r, timeout=1800)); return d, time.time() - t
print(json.load(urllib.request.urlopen(B + "/v1/models"))["data"][0]["id"])
q = "A train leaves at 9:40 and the trip takes 1 hour 35 minutes. What time does it arrive? Answer briefly."
for think in (False, True):
    d, dt = post("/v1/chat/completions", {"model": "qwen3.8-27b-exl3", "messages": [{"role": "user", "content": q}],
               "temperature": 0, "chat_template_kwargs": {"enable_thinking": think}})
    m = d["choices"][0]["message"]; u = d["usage"]
    print(f"think={think} finish={d['choices'][0]['finish_reason']} tokens={u['completion_tokens']} {dt:.1f}s "
          f"({u['completion_tokens']/dt:.1f} tok/s e2e)")
    print("  reasoning:", (m.get("reasoning_content") or "")[-200:].replace("\n", " "))
    print("  content:", (m.get("content") or "")[:300].replace("\n", " "))
