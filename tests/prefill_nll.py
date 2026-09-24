"""
Teacher-forced NLL of real text through the served model's prefill path (prompt_logprobs), to compare
prefill numerics variants (e.g. EXL3_INT8_PREFILL=1 vs fp16). Saves per-token logprobs for pairing.
  python3 tests/prefill_nll.py --base http://localhost:8101 --out /w/logs/nll_fp16.json [--compare other.json]
"""
import argparse, json, math, os, sys, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
import sweep  # noqa: E402


def main(a):
    books = sweep._books()
    docs = [sweep._take_tokens(books[i % len(books)], 5000 * (i + 1), a.tokens) for i in range(a.n)]
    res = []
    for d in docs:
        body = {"model": a.model, "prompt": d, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1}
        req = urllib.request.Request(a.base + "/v1/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=1800))
        pl = r["choices"][0]["prompt_logprobs"][1:]
        lp, top = [], []
        for e in pl:
            vals = sorted(e.values(), key=lambda v: v.get("rank", 99))
            # vLLM returns the actual token plus the top-1; the actual token's entry is the one we need
            lp.append(min(v["logprob"] for v in e.values()) if len(e) > 1 else vals[0]["logprob"])
            top.append(next(k for k, v in e.items() if v.get("rank") == 1))
        res.append({"lp": lp, "top": top})
        print(f"doc {len(res)}: {len(lp)} tokens, nll {-sum(lp)/len(lp):.4f}", flush=True)
    json.dump(res, open(a.out, "w"))
    tot = [x for r in res for x in r["lp"]]
    print(f"mean nll {-sum(tot)/len(tot):.5f} over {len(tot)} tokens")
    if a.compare:
        ref = json.load(open(a.compare))
        d = [x - y for r1, r2 in zip(res, ref) for x, y in zip(r1["lp"], r2["lp"])]
        agree = sum(t1 == t2 for r1, r2 in zip(res, ref) for t1, t2 in zip(r1["top"], r2["top"]))
        n = sum(len(r["top"]) for r in res)
        print(f"vs {a.compare}: mean dNLL {-sum(d)/len(d):+.5f}, mean |dlogp| {sum(abs(x) for x in d)/len(d):.5f}, "
              f"top-1 agreement {agree/n*100:.2f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8101")
    ap.add_argument("--model", default="qwen3.8-27b-exl3")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare")
    main(ap.parse_args())
