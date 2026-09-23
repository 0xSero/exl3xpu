"""
Choose the vocabulary blocks for a pruned MTP draft lm_head.

EXL3 lm_head columns can only be sliced in 128-column blocks (the output Hadamard spans 128 columns),
so we rank 128-token blocks by token frequency on a text + code corpus, force-include every special /
added token, and keep the top N blocks. The draft can only propose tokens inside these blocks; the
target verifies with the full lm_head, so outputs are unchanged (greedy and rejection sampling).

  python3 scripts/draft_vocab.py --model DIR --blocks 256 --out models/<id>/draft_vocab.json
"""
import argparse, glob, json, os, sysconfig
import numpy as np
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--blocks", type=int, default=256)
ap.add_argument("--out", required=True)
ap.add_argument("--text", action="append", default=[], help="extra corpus files/globs")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
V = len(tok)
nblk = (V + 127) // 128
counts = np.zeros(nblk * 128, dtype=np.int64)

files = []
for g in args.text or ["/exllamav3/eval/eval_texts/*.txt"]:
    files += glob.glob(g)
stdlib = sysconfig.get_paths()["stdlib"]
files += sorted(glob.glob(os.path.join(stdlib, "*.py")))[:300]            # code
files += sorted(glob.glob("/opt/venv/lib/python3.12/site-packages/vllm/**/*.py", recursive=True))[:200]
n_tok = 0
for f in files:
    try:
        ids = tok(open(f, errors="ignore").read(), add_special_tokens=False)["input_ids"]
    except Exception:
        continue
    np.add.at(counts, ids, 1)
    n_tok += len(ids)
# chat scaffolding the model emits constantly
for s in ["<|im_start|>assistant\n", "<|im_end|>", "<think>\n", "\n</think>\n\n", "```python\n", "```\n"]:
    counts[tok(s, add_special_tokens=False)["input_ids"]] += n_tok // 1000

block_score = counts.reshape(nblk, 128).sum(1)
forced = set()
for t in set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}).keys()):
    if 0 <= t < V:
        forced.add(t // 128)
order = [b for b in np.argsort(-block_score) if b not in forced]
chosen = sorted(forced | set(int(b) for b in order[: max(0, args.blocks - len(forced))]))
coverage = block_score[chosen].sum() / max(1, block_score.sum())
json.dump({"model": args.model, "block_size": 128, "blocks": chosen, "n_blocks": len(chosen),
           "tokens": len(chosen) * 128, "vocab": V, "corpus_tokens": int(n_tok),
           "corpus_coverage": float(coverage)}, open(args.out, "w"))
print(f"{len(chosen)} blocks ({len(chosen) * 128} tokens of {V}), corpus coverage {coverage:.4f} over {n_tok} tokens")
