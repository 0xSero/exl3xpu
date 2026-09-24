"""
Vision benchmark + multi-image correctness for an OpenAI-compatible server.

  speed: N images of R x R pixels (generated, each with a distinct large number and shapes) + a short question,
         streamed; reports TTFT, image tokens (prompt_tokens) and the encode+prefill rate, and decode tok/s of a
         ~300-token answer about the images.
  check: N images numbered in a random order; the model must list the numbers in order (exact-match score).
Usage: python3 bench/vision_bench.py --base http://localhost:8101 [--counts 1,4,16] [--sizes 512,1024,2048,4096]
"""
from __future__ import annotations
import argparse, base64, io, json, random, time, urllib.request


def image(n: int, size: int, rnd: random.Random) -> str:
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (size, size), tuple(rnd.randint(200, 255) for _ in range(3)))
    d = ImageDraw.Draw(img)
    for _ in range(12):
        x, y, r = rnd.randint(0, size), rnd.randint(0, size), rnd.randint(size // 40, size // 8)
        d.ellipse([x - r, y - r, x + r, y + r], outline=tuple(rnd.randint(0, 160) for _ in range(3)), width=max(2, size // 200))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size // 4)
    except Exception:
        font = ImageFont.load_default(size // 4)
    d.text((size * 0.3, size * 0.35), str(n), fill=(0, 0, 0), font=font)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def ask(base, model, content, max_tokens, temperature=0.0):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": temperature, "stream": True, "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; text = ""; usage = {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece and t_first is None:
                    t_first = time.time()
                text += piece
    t1 = time.time()
    return {"ttft": (t_first or t1) - t0, "total": t1 - t0, "text": text,
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8101")
    ap.add_argument("--model", default="qwen3.8-27b-exl3")
    ap.add_argument("--counts", default="1,4,16")
    ap.add_argument("--sizes", default="512,1024,2048")
    ap.add_argument("--check-counts", default="4,8,16")
    ap.add_argument("--check-size", type=int, default=1024)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    rnd = random.Random(7)
    ask(a.base, a.model, [{"type": "text", "text": "hi"}], 4)   # warm
    for size in [int(s) for s in a.sizes.split(",")]:
        for n in [int(c) for c in a.counts.split(",")]:
            imgs = [image(rnd.randint(10, 99), size, rnd) for _ in range(n)]
            content = [{"type": "image_url", "image_url": {"url": u}} for u in imgs]
            content.append({"type": "text", "text": "Describe these images in detail, one paragraph per image."})
            r = ask(a.base, a.model, content, 512)
            dec = (r["completion_tokens"] - 1) / max(1e-6, r["total"] - r["ttft"]) if r["completion_tokens"] else None
            row = {"kind": "vision_speed", "images": n, "px": size, "prompt_tokens": r["prompt_tokens"],
                   "ttft_s": round(r["ttft"], 2), "prefill_tok_s": round(r["prompt_tokens"] / r["ttft"], 1) if r["prompt_tokens"] else None,
                   "decode_tok_s": round(dec, 1) if dec else None, "completion_tokens": r["completion_tokens"]}
            rows.append(row)
            print(f"speed {n:>2} x {size:>4}px: prompt {row['prompt_tokens']:>6} tok, TTFT {row['ttft_s']:>6} s "
                  f"({row['prefill_tok_s']} tok/s), decode {row['decode_tok_s']} tok/s", flush=True)
    ok_all = True
    for n in [int(c) for c in a.check_counts.split(",")]:
        nums = rnd.sample(range(10, 100), n)
        content = [{"type": "image_url", "image_url": {"url": image(x, a.check_size, rnd)}} for x in nums]
        content.append({"type": "text", "text": f"Each of the {n} images shows one large two-digit number. List the numbers "
                        f"in the order the images appear, comma-separated, nothing else."})
        r = ask(a.base, a.model, content, 128)
        got = [int(x) for x in __import__("re").findall(r"\d{2}", r["text"])][:n]
        ok = got == nums
        ok_all &= ok
        rows.append({"kind": "vision_check", "images": n, "px": a.check_size, "expected": nums, "got": got, "ok": ok})
        print(f"check {n:>2} images @ {a.check_size}px: {'OK' if ok else 'FAIL'}  expected {nums}  got {got}", flush=True)
    print("VISION_BENCH_PASS" if ok_all else "VISION_BENCH_FAIL")
    if a.out:
        with open(a.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
