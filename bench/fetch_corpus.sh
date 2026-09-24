#!/usr/bin/env bash
# Real-text benchmark corpus (not committed): Project Gutenberg books (public domain) + HumanEval problems.
# Usage: bench/fetch_corpus.sh [dir]   (default bench/corpus)
set -euo pipefail
D=${1:-$(dirname "$0")/corpus}
mkdir -p "$D/gutenberg"
# Pride and Prejudice, Moby Dick, Frankenstein, Sherlock Holmes, A Tale of Two Cities, War and Peace,
# The Count of Monte Cristo, Great Expectations
for id in 1342 2701 84 1661 98 2600 1184 1400; do
  f="$D/gutenberg/pg$id.txt"
  [ -s "$f" ] && continue
  curl -sfL --max-time 120 "https://www.gutenberg.org/cache/epub/$id/pg$id.txt" -o "$f.raw"
  # keep only the book body between the Project Gutenberg START/END markers
  awk '/\*\*\* START OF/{on=1;next} /\*\*\* END OF/{on=0} on' "$f.raw" > "$f" && rm -f "$f.raw"
done
if [ ! -s "$D/humaneval.jsonl" ]; then
  curl -sfL --max-time 120 -o "$D/humaneval.parquet" \
    "https://huggingface.co/datasets/openai/openai_humaneval/resolve/main/openai_humaneval/test-00000-of-00001.parquet"
  python3 - "$D" <<'PY'
import json, sys, pyarrow.parquet as pq
d = sys.argv[1]
rows = pq.read_table(f"{d}/humaneval.parquet").to_pylist()
with open(f"{d}/humaneval.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps({"task_id": r["task_id"], "prompt": r["prompt"]}) + "\n")
print(len(rows), "humaneval problems")
PY
  rm -f "$D/humaneval.parquet"
fi
ls -la "$D" "$D/gutenberg"
