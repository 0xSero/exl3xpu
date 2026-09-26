#!/usr/bin/env python3
"""Resilient cell runner for the borrowed B70 (host side, stdlib only).

plan.json: {"name": str, "engine": "sglang"|"vllm", "run": str (server run name), "port": int,
            "launch": str (host shell cmd that (re)creates container+server), "cells": [{"id": str, "cmd": str,
            "timeout": s}]}
Each cell is checkpointed in ~/sglb70/ckpt/<name>.json: a drop costs one cell. Before every cell: the card's render
node must exist (else wait for re-enumeration), the server must answer /v1/models (else relaunch, which re-maps the
new node). After a failed cell the kernel log is checked for a Slot link-down since the cell started: slot 17
(84:00.0) -> logged as a drop, cell retried (max 3); ANY OTHER slot -> abort (the owner's rule).
Events -> ~/sglb70/ledger.jsonl (kind=drop|cell|abort).
"""
import json, os, subprocess, sys, time, urllib.request

HOME = os.path.expanduser("~")
PCI = os.environ.get("SGLB70_PCI", "0000:84:00.0")
OUR_SLOT = os.environ.get("SGLB70_SLOT", "17")
LEDGER = f"{HOME}/sglb70/ledger.jsonl"


def sh(cmd, timeout):
    try:
        p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout[-6000:] + p.stderr[-3000:]
    except subprocess.TimeoutExpired as e:
        return 124, f"TIMEOUT after {timeout}s\n" + ((e.stdout or b"").decode(errors="ignore")[-3000:] if isinstance(e.stdout, bytes) else str(e.stdout or "")[-3000:])


def log(logf, msg):
    line = time.strftime("%Y-%m-%dT%H:%M:%S ") + msg
    print(line, flush=True)
    with open(logf, "a") as f:
        f.write(line + "\n")


def ledger(row):
    row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(LEDGER, "a") as f:
        f.write(json.dumps(row) + "\n")


def node():
    p = f"/dev/dri/by-path/pci-{PCI}-render"
    return os.path.basename(os.path.realpath(p)) if os.path.exists(p) else None


def healthy(port):
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10).status == 200
    except Exception:
        return False


def link_events(since_epoch):
    since = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(since_epoch))
    _, out = sh(f"journalctl -k --since '{since}' --no-pager 2>/dev/null | grep -E 'pciehp: Slot\\([0-9]+\\): Link Down'", 30)
    return [l for l in out.splitlines() if "Link Down" in l]


def main():
    plan = json.load(open(sys.argv[1]))
    name = plan["name"]
    os.makedirs(f"{HOME}/sglb70/ckpt", exist_ok=True)
    ck_path = f"{HOME}/sglb70/ckpt/{name}.json"
    ck = json.load(open(ck_path)) if os.path.exists(ck_path) else {"done": {}, "tries": {}}
    logf = f"{HOME}/sglb70/logs/supervise-{name}.log"
    t_start = time.time()
    for cell in plan["cells"]:
        cid = cell["id"]
        if cid in ck["done"]:
            continue
        while True:
            tries = ck["tries"].get(cid, 0)
            if tries >= 3:
                log(logf, f"cell {cid}: giving up after 3 tries"); break
            # card present?
            waited = 0
            while node() is None and waited < 1800:
                if waited == 0:
                    log(logf, f"card {PCI} absent: waiting for re-enumeration")
                time.sleep(10); waited += 10
            if node() is None:
                log(logf, "card did not come back in 30 min: abort"); ledger({"kind": "abort", "plan": name, "why": "card absent"}); return 2
            # other slots down since the plan started? -> abort
            other = [l for l in link_events(t_start) if f"Slot({OUR_SLOT})" not in l]
            if other:
                log(logf, "OTHER SLOT LINK DOWN: abort\n" + "\n".join(other[:4]))
                ledger({"kind": "abort", "plan": name, "why": "other slot link down", "lines": other[:4]}); return 3
            if not healthy(plan["port"]):
                log(logf, f"server not healthy: (re)launch on {node()}")
                rc, out = sh(plan["launch"], 1500)
                if rc != 0 or not healthy(plan["port"]):
                    log(logf, f"launch failed rc={rc}: {out[-800:]}")
                    ck["tries"][cid] = tries + 1; json.dump(ck, open(ck_path, "w")); time.sleep(20); continue
            t0 = time.time()
            log(logf, f"cell {cid} start (try {tries + 1}, node {node()})")
            rc, out = sh(cell["cmd"], cell.get("timeout", 3600))
            drops = [l for l in link_events(t0) if f"Slot({OUR_SLOT})" in l]
            dead = not healthy(plan["port"])
            if drops or (rc != 0 and dead):
                log(logf, f"cell {cid}: DROP/DEAD rc={rc} drops={len(drops)} dead={dead}")
                ledger({"kind": "drop", "plan": name, "cell": cid, "rc": rc, "server_dead": dead,
                        "link_down": drops[:2], "cell_s": round(time.time() - t0)})
                ck["tries"][cid] = tries + 1; json.dump(ck, open(ck_path, "w"))
                sh(plan.get("kill", "true"), 120)
                continue
            ck["done"][cid] = {"rc": rc, "s": round(time.time() - t0), "tail": out[-1500:]}
            json.dump(ck, open(ck_path, "w"))
            log(logf, f"cell {cid} done rc={rc} in {round(time.time() - t0)}s\n{out[-1200:]}")
            break
    log(logf, f"plan {name} complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
