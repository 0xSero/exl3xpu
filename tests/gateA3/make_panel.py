"""Build the sealed Gate A3 panel: 64 windows x 256 tokens of real text (exllamav3 eval_texts)."""
import glob, json, os, sys
from transformers import AutoTokenizer
MODEL = sys.argv[1]; SRC = sys.argv[2]; OUT = sys.argv[3]
tok = AutoTokenizer.from_pretrained(MODEL)
wins = []
for f in sorted(glob.glob(os.path.join(SRC, "*.txt"))):
    ids = tok(open(f).read())["input_ids"]
    for s in range(0, len(ids) - 256, 256 * 3):   # non-overlapping, spread out
        wins.append({"src": os.path.basename(f), "start": s, "ids": ids[s:s + 256]})
step = max(1, len(wins) // 64)
panel = wins[::step][:64]
json.dump({"tokenizer": MODEL, "n": len(panel), "len": 256, "windows": panel}, open(OUT, "w"))
print(f"{len(wins)} candidate windows -> {len(panel)} in panel")
