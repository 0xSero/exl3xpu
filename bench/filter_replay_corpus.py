"""Drop replay sessions whose largest request (history up to the last replayed cut, chat template + tools) plus an
output reserve exceeds the served context. Logs kept/dropped counts. CPU only (tokenizer)."""
import json, sys
from transformers import AutoTokenizer
src, dst, tok_path = sys.argv[1], sys.argv[2], sys.argv[3]
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 262144
reserve = int(sys.argv[5]) if len(sys.argv) > 5 else 16384
turns = int(sys.argv[6]) if len(sys.argv) > 6 else 12
tok = AutoTokenizer.from_pretrained(tok_path)
kept = dropped = 0; sizes = []
with open(dst, "w") as out:
    for line in open(src):
        s = json.loads(line)
        cut = s["cuts"][:turns][-1]
        msgs = s["messages"][:cut]
        try:
            n = len(tok.apply_chat_template(msgs, tools=s["tools"] or None, add_generation_prompt=True, tokenize=True))
        except Exception:
            n = len(tok(json.dumps(msgs))["input_ids"])
        sizes.append(n)
        if n + reserve <= limit:
            out.write(line); kept += 1
        else:
            dropped += 1
sizes.sort()
print(json.dumps({"kept": kept, "dropped": dropped, "limit": limit, "reserve": reserve, "turns": turns,
                  "max_prompt_tokens_p50": sizes[len(sizes) // 2], "max_prompt_tokens_max": sizes[-1]}))
