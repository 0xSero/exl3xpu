"""Redact secrets from an omp replay corpus (every string, incl. tool-call argument JSON). Prints counts only."""
import json, re, sys
PATS = [
 ("anthropic", r"sk-ant-[A-Za-z0-9_-]{20,}"), ("openai_sk", r"sk-[A-Za-z0-9_-]{20,}"), ("github", r"gh[pousr]_[A-Za-z0-9]{30,}"),
 ("github_pat", r"github_pat_[A-Za-z0-9_]{40,}"), ("hf", r"hf_[A-Za-z0-9]{30,}"), ("aws_akid", r"AKIA[0-9A-Z]{16}"),
 ("slack", r"xox[baprs]-[A-Za-z0-9-]{10,}"),
 ("private_key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(-----END [A-Z ]*PRIVATE KEY-----|$)"),
 ("bearer", r"[Bb]earer [A-Za-z0-9._~+/=-]{24,}"), ("jwt", r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
 ("google_api", r"AIza[0-9A-Za-z_-]{35}"), ("tailscale", r"tskey-[A-Za-z0-9-]{20,}"), ("stripe", r"(sk|rk)_live_[A-Za-z0-9]{20,}"),
 ("gitlab", r"glpat-[A-Za-z0-9_-]{20,}"),
 ("kv_secret", r"(?i)((?:password|passwd|api[_-]?key|secret|token)\s*[=:]\s*['\"]?)[A-Za-z0-9/+_.-]{12,}"),
]
RX = [(n, re.compile(p)) for n, p in PATS]
counts = {n: 0 for n, _ in PATS}

def red(s):
    for n, rx in RX:
        def sub(m, n=n):
            counts[n] += 1
            return (m.group(1) if n == "kv_secret" else "") + f"[REDACTED_{n.upper()}]"
        s = rx.sub(sub, s)
    return s

def walk(o):
    if isinstance(o, str): return red(o)
    if isinstance(o, list): return [walk(x) for x in o]
    if isinstance(o, dict): return {k: walk(v) for k, v in o.items()}
    return o

src, dst = sys.argv[1], sys.argv[2]
n = 0
with open(dst, "w") as out:
    for line in open(src):
        out.write(json.dumps(walk(json.loads(line))) + "\n"); n += 1
print("sessions", n, "redactions", {k: v for k, v in counts.items() if v})
