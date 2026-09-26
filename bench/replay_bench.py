#!/usr/bin/env python3
"""Replay real omp agent sessions against an OpenAI-compatible server (vLLM exl3xpu or SGLang), one B70.

Adapted from ~/tuning-kit/bench.py cell_replay (same session order seed, same body: tools, thinking on, temperature 1,
top-p 0.95, NO max_tokens: every turn ends at EOS). Each of C agents replays one session turn by turn (the history up
to assistant turn i, with the recorded tool results), so the server sees growing multi-turn contexts whose prefix is
the previous turn: prefix caching matters.

Per cell (replay:C) one ledger row with: per-stream decode tok/s p50/p5, aggregate decode tok/s, output tok/s over
wall time, TTFT p50/p95 split into cached-prefix turns vs new-token prefill, new-token prefill tok/s, turns/min,
speculative acceptance (vLLM /metrics or SGLang server log), flags (LOOP, REQ_FAIL, KV_FULL, TOOL_JSON), whole-session
wall time. Raw per-turn records go to --runs/<exp>/replay-cC.jsonl.

  python3 bench/replay_bench.py --url http://127.0.0.1:8210 --exp sgl-x --cells 1,2,4,8 \
      --replay ~/sglb70/replay/omp_replay_corpus.redacted.filtered.jsonl --card card7 --sgl-log /w/runs/X/server.log
"""
import argparse, asyncio, datetime, glob, json, os, random, re, statistics, sys, time
import aiohttp

P = argparse.ArgumentParser()
P.add_argument("--url", default="http://127.0.0.1:8210")
P.add_argument("--model", default="qwen3.8-27b-exl3")
P.add_argument("--exp", required=True)
P.add_argument("--engine", default="")
P.add_argument("--ledger", default=os.path.expanduser("~/sglb70/ledger.jsonl"))
P.add_argument("--runs", default=os.path.expanduser("~/sglb70/replay/runs"))
P.add_argument("--replay", required=True)
P.add_argument("--cells", default="1,2,4,8")
P.add_argument("--turns", type=int, default=12)
P.add_argument("--seed", default="replay-v1", help="session order seed (same for both engines)")
P.add_argument("--card", default="", help="DRM card name for the busy sampler, e.g. card7")
P.add_argument("--sgl-log", default="", help="SGLang server log for live speculative acceptance")
P.add_argument("--note", default="")
A = P.parse_args()


async def stream_req(sess, b, rec):
    rec.update(t0=time.perf_counter(), events=[], text="", reasoning="", ok=False, tool_calls=0, tc_args={})
    try:
        async with sess.post(A.url + "/v1/chat/completions", json=b, timeout=aiohttp.ClientTimeout(total=7200)) as r:
            if r.status != 200:
                rec["err"] = f"{r.status} {(await r.text())[:300]}"; return rec
            buf = b""
            async for chunk in r.content.iter_any():
                buf += chunk
                while b"\n\n" in buf:
                    line, buf = buf.split(b"\n\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        continue
                    d = json.loads(data); now = time.perf_counter()
                    u = d.get("usage"); got = False
                    for c in d.get("choices") or []:
                        dl = c.get("delta", {})
                        t = dl.get("content") or ""; rs = dl.get("reasoning_content") or dl.get("reasoning") or ""
                        if t or rs or dl.get("tool_calls"):
                            got = True
                        rec["text"] += t; rec["reasoning"] += rs
                        for tc in dl.get("tool_calls") or []:
                            i = tc.get("index", 0)
                            if i not in rec["tc_args"]:
                                rec["tc_args"][i] = ""; rec["tool_calls"] += 1
                            rec["tc_args"][i] += (tc.get("function") or {}).get("arguments") or ""
                        if c.get("finish_reason"):
                            rec["finish"] = c["finish_reason"]
                    if u:
                        rec["ptoks"] = u.get("prompt_tokens"); ct = u.get("completion_tokens") or 0
                        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
                        if cached is not None:
                            rec["cached"] = cached
                        if got or (rec["events"] and ct > rec["events"][-1][1]):
                            if "ttft" not in rec:
                                rec["ttft"] = now - rec["t0"]; rec["t_first"] = now
                            rec["events"].append((now, ct)); rec["t_last"] = now; rec["ctoks"] = ct
                    elif got and "ttft" not in rec:
                        rec["ttft"] = now - rec["t0"]; rec["t_first"] = now
            rec["ok"] = True
    except Exception as e:
        rec["err"] = repr(e)[:300]
    rec["t_end"] = time.perf_counter()
    return rec


def loop_flag(text):
    toks = text.split()
    if len(toks) < 200:
        return False
    grams = {}
    for i in range(0, len(toks) - 32, 4):
        g = " ".join(toks[i:i + 32]); grams[g] = grams.get(g, 0) + 1
        if grams[g] >= 3:
            return True
    return False


async def metrics(sess):
    try:
        async with sess.get(A.url + "/metrics", timeout=aiohttp.ClientTimeout(total=5)) as r:
            t = await r.text()
    except Exception:
        return {}
    out = {}
    for line in t.splitlines():
        m = re.match(r"^([a-zA-Z_:]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if m and any(s in m.group(1) for s in ("spec_decode", "preempt", "retract", "prefix_cache", "cache_hit")):
            out[m.group(1)] = out.get(m.group(1), 0) + float(m.group(3))
    return out


def sgl_log_accept(t0, t1):
    if not A.sgl_log or not os.path.exists(A.sgl_log):
        return None, 0
    rx = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] Decode batch.*?accept len: ([\d.]+).*?gen throughput \(token/s\): ([\d.]+)")
    rr = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\].*(retract|Retract)")
    tok = steps = 0.0; retracts = 0
    for line in open(A.sgl_log, errors="ignore"):
        m = rx.match(line) or rr.match(line)
        if not m:
            continue
        ts = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp()
        if not (t0 <= ts < t1):
            continue
        if m.re is rr:
            retracts += 1; continue
        a, g = float(m.group(2)), float(m.group(3))
        if a > 0 and g > 0:
            tok += g; steps += g / a
    return (round(tok / steps, 3) if steps else None), retracts


class Busy:
    def __init__(self):
        self.path = f"/sys/class/drm/{A.card}/device/tile0/gt0/gtidle/idle_residency_ms" if A.card else None
        self.s = []; self.stop = False

    async def run(self):
        if not self.path or not os.path.exists(self.path):
            return
        prev = (time.time(), int(open(self.path).read()))
        while not self.stop:
            await asyncio.sleep(2)
            cur = (time.time(), int(open(self.path).read()))
            self.s.append(100 * (1 - (cur[1] - prev[1]) / (1000 * (cur[0] - prev[0])))); prev = cur

    def summary(self):
        return {"gpu_busy_mean": round(statistics.mean(self.s), 1)} if self.s else {}


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))] if v else None


async def cell_replay(sess, C):
    sessions = [json.loads(l) for l in open(A.replay)]
    random.Random(A.seed + str(C)).shuffle(sessions)
    recs, walls = [], {}

    async def agent(s):
        t = time.perf_counter(); prev_total = 0
        for c in s["cuts"][:A.turns]:
            b = {"model": A.model, "messages": s["messages"][:c], "stream": True, "temperature": 1.0, "top_p": 0.95,
                 "stream_options": {"include_usage": True, "continuous_usage_stats": True},
                 "chat_template_kwargs": {"enable_thinking": True}}
            if s["tools"]:
                b["tools"] = s["tools"]
            r = {"session": s["id"], "turn": c, "prev_total": prev_total}
            recs.append(r); await stream_req(sess, b, r)
            prev_total = (r.get("ptoks") or 0) + (r.get("ctoks") or 0)
        walls[s["id"]] = time.perf_counter() - t

    busy = Busy(); bt = asyncio.create_task(busy.run())
    m0 = await metrics(sess); w0 = time.time(); t0 = time.perf_counter()
    await asyncio.gather(*[agent(s) for s in sessions[:C]])
    el = time.perf_counter() - t0; w1 = time.time(); m1 = await metrics(sess)
    busy.stop = True; await bt
    ok = [r for r in recs if r.get("ok") and "ttft" in r]
    per = [(r["ctoks"] - 1) / (r["t_last"] - r["t_first"]) for r in ok if r.get("ctoks", 0) > 32 and r["t_last"] > r["t_first"]]
    # decode-only aggregate: output tokens / wall time with at least one stream decoding
    spans = sorted((r["t_first"], r["t_last"]) for r in ok if r.get("t_last", 0) > r.get("t_first", 0))
    busy_t, cur = 0.0, None
    for a, b in spans:
        if cur is None or a > cur[1]:
            if cur: busy_t += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur:
        busy_t += cur[1] - cur[0]
    out_tok = sum(r.get("ctoks", 0) for r in ok)
    # cached vs new tokens per turn: server-reported cached_tokens when present, else the previous turn's total
    for r in ok:
        cached = r.get("cached")
        r["new_tokens"] = (r.get("ptoks") or 0) - (cached if cached is not None else min(r["prev_total"], r.get("ptoks") or 0))
    cached_turns = [r for r in ok if r["prev_total"] > 0]
    first_turns = [r for r in ok if r["prev_total"] == 0]
    newpf = [r["new_tokens"] / r["ttft"] for r in ok if r["new_tokens"] > 2048]
    tj_bad = 0
    for r in ok:
        for a in r["tc_args"].values():
            try:
                json.loads(a or "{}")
            except Exception:
                tj_bad += 1
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
    acc = next((v for k, v in d.items() if "spec_decode_num_accepted_tokens_total" in k and "per_pos" not in k), None)
    drafts = next((v for k, v in d.items() if "spec_decode_num_drafts_total" in k), None)
    accept = round(1 + acc / drafts, 3) if acc is not None and drafts else None
    sgl_acc, retracts = sgl_log_accept(w0, w1)
    if accept is None:
        accept = sgl_acc
    preempt = sum(v for k, v in d.items() if "preempt" in k) + retracts
    loops = sum(1 for r in ok if loop_flag(r["text"] + r["reasoning"]))
    flags = []
    if len(ok) < len(recs): flags.append("REQ_FAIL")
    if loops: flags.append("LOOP")
    if preempt: flags.append("KV_FULL")
    if tj_bad: flags.append("TOOL_JSON")
    ms = lambda v: round(1000 * v) if v is not None else None
    row = {"cell": "replay", "concurrency": C, "engine": A.engine, "turns": len(recs), "turns_ok": len(ok),
           "prompt_tokens_p50": statistics.median([r.get("ptoks") or 0 for r in ok]) if ok else None,
           "prompt_tokens_max": max([r.get("ptoks") or 0 for r in ok]) if ok else None,
           "decode_tok_s_per_stream_p50": round(statistics.median(per), 1) if per else None,
           "decode_tok_s_per_stream_p5": round(pct(per, 0.05), 1) if per else None,
           "decode_tok_s_aggregate": round(out_tok / busy_t, 1) if busy_t else None,
           "output_tok_s_wall": round(out_tok / el, 1),
           "ttft_ms_p50_all": ms(statistics.median([r["ttft"] for r in ok])) if ok else None,
           "ttft_ms_p95_all": ms(pct([r["ttft"] for r in ok], 0.95)),
           "ttft_ms_p50_cached_prefix": ms(statistics.median([r["ttft"] for r in cached_turns])) if cached_turns else None,
           "ttft_ms_p95_cached_prefix": ms(pct([r["ttft"] for r in cached_turns], 0.95)),
           "ttft_ms_p50_first_turn": ms(statistics.median([r["ttft"] for r in first_turns])) if first_turns else None,
           "ttft_ms_p95_first_turn": ms(pct([r["ttft"] for r in first_turns], 0.95)),
           "new_token_prefill_tok_s_p50": round(statistics.median(newpf), 1) if newpf else None,
           "cached_tokens_reported": any("cached" in r for r in ok),
           "turns_per_min": round(60 * len(ok) / el, 2),
           "spec_accept_len": accept, "preemptions_or_retracts": preempt,
           "tool_call_turns": sum(1 for r in ok if r["tool_calls"]), "tool_json_malformed": tj_bad, "loops": loops,
           "session_wall_s_p50": round(statistics.median(walls.values())) if walls else None,
           "session_wall_s_max": round(max(walls.values())) if walls else None,
           "elapsed_s": round(el), "flags": flags, **busy.summary()}
    os.makedirs(os.path.join(A.runs, A.exp), exist_ok=True)
    with open(os.path.join(A.runs, A.exp, f"replay-c{C}.jsonl"), "w") as f:
        for r in recs:
            r2 = {k: v for k, v in r.items() if k != "events"}; r2["n_events"] = len(r.get("events", []))
            f.write(json.dumps(r2) + "\n")
    row.update(exp=A.exp, model=A.model, ts=time.strftime("%Y-%m-%dT%H:%M:%S"), note=A.note, replay=os.path.basename(A.replay))
    os.makedirs(os.path.dirname(A.ledger), exist_ok=True)
    with open(A.ledger, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


async def main():
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as sess:
        # warm-up (discarded): one short turn
        await stream_req(sess, {"model": A.model, "messages": [{"role": "user", "content": "Say ok."}], "stream": True,
                                "chat_template_kwargs": {"enable_thinking": False}}, {})
        for c in A.cells.split(","):
            await cell_replay(sess, int(c))


if __name__ == "__main__":
    asyncio.run(main())
