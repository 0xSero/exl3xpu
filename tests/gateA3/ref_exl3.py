"""exllamav3 (CUDA, 3090) teacher-forced log-probs for the Gate A3 panel. Saves full-vocab fp16 log-probs."""
import argparse, json, os, sys, torch
from exllamav3 import model_init
p = argparse.ArgumentParser(); model_init.add_args(p, cache=False)
p.add_argument("--panel", required=True); p.add_argument("--out", required=True)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
panel = json.load(open(args.panel))
with torch.inference_mode():
    model, config, _, tok = model_init.init(args, override_dynamic_seq_len=2048, max_output_size=2048, max_output_factor=5)
    for i, w in enumerate(panel["windows"]):
        ids = torch.tensor([w["ids"]], dtype=torch.long)
        logits = model.forward(ids, {"attn_mode": "flash_attn_nc"}).float()[0]
        lp = torch.log_softmax(logits, dim=-1)
        torch.save(lp.half().cpu(), f"{args.out}/p{i:03d}.pt")
        print(i, lp.shape, flush=True)
