"""Our engine (vLLM + exl3xpu on B70): prompt log-probs (top-64) for the Gate A3 panel, compared
against exllamav3's full-vocab log-probs.
  top1      : argmax agreement (exact)
  kl_q_p    : KL(ours || exl3) over our top-64 support (exl3 log-probs exact there; tail < 1e-3 mass)
  kl_p_q    : KL(exl3 || ours) over exl3's top-64, ours floored at our 64th log-prob where absent (upper bound)
"""
import json, os, sys, math, statistics, torch
from vllm import LLM, SamplingParams
MODEL, PANEL, REF = sys.argv[1], sys.argv[2], sys.argv[3]
panel = json.load(open(PANEL))
llm = LLM(model=MODEL, dtype="float16", language_model_only=True, max_model_len=4096, max_logprobs=64,
          gpu_memory_utilization=0.85, enable_prefix_caching=False,
          enforce_eager=os.environ.get("EAGER", "0") == "1")
sp = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=64)
outs = llm.generate([{"prompt_token_ids": w["ids"]} for w in panel["windows"]], sp)
agree = n = 0
klqp, klpq = [], []
for i, o in enumerate(outs):
    ref = torch.load(f"{REF}/p{i:03d}.pt").float()          # [256, V]: position j predicts token j+1
    for j in range(1, len(o.prompt_logprobs)):
        d = o.prompt_logprobs[j]                             # dist over token j given prefix < j
        ids = torch.tensor(list(d.keys())); q = torch.tensor([v.logprob for v in d.values()])
        top = [k for k, v in d.items() if v.rank == 1][0] if any(v.rank == 1 for v in d.values()) else ids[q.argmax()].item()
        p = ref[j - 1]
        agree += int(p.argmax().item() == top); n += 1
        qp = q.exp()
        klqp.append(float((qp * (q - p[ids])).sum()))
        pt, pi = p.topk(64)
        qmap = dict(zip(ids.tolist(), q.tolist())); qmin = q.min().item()
        qv = torch.tensor([qmap.get(k, qmin) for k in pi.tolist()])
        klpq.append(float((pt.exp() * (pt - qv)).sum()))
res = dict(positions=n, top1_agreement=agree / n, kl_q_p_mean=statistics.mean(klqp),
           kl_p_q_mean_upper=statistics.mean(klpq), kl_q_p_p99=sorted(klqp)[int(0.99 * n)])
print(json.dumps(res, indent=1))
print("GATE_A3_PASS" if res["top1_agreement"] >= 0.99 and res["kl_q_p_mean"] <= 0.005 else "GATE_A3_FAIL")
