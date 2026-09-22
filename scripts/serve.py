"""
Launch vLLM on Intel XPU from a model config (models/<id>/model.yaml).

  python3 scripts/serve.py models/qwen3.8-27b-exl3-4.00bpw --gpu 0            # one replica
  python3 scripts/serve.py models/qwen3.8-27b-exl3-4.00bpw --dp                # replica per GPU + LB
  python3 scripts/serve.py models/... --gpu 0 --print                           # show the command only
  python3 scripts/serve.py models/... --gpu 0 -- --enforce-eager                # extra vllm args

--model-path overrides where the weights are (default: $MODELS_DIR/<repo name> or the HF repo id).
"""
from __future__ import annotations
import argparse, json, os, shlex, signal, subprocess, sys, time
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "model.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def model_path(cfg, override):
    if override:
        return override
    repo = cfg["source"]["repo"]
    d = os.environ.get("MODELS_DIR")
    if d:
        for cand in (cfg["id"], repo.split("/")[-1], repo.replace("/", "-")):
            p = os.path.join(d, cand)
            if os.path.isdir(p):
                return p
    return repo


def vllm_argv(cfg, path, port, extra):
    v = cfg["vllm"]
    argv = ["vllm", "serve", path, "--served-model-name", cfg["served_model_name"],
            "--host", "0.0.0.0", "--port", str(port)]
    if not os.path.isdir(path):
        argv += ["--revision", cfg["source"]["revision"]]
    for k, val in v.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(val, bool):
            if val:
                argv.append(flag)
        elif isinstance(val, (dict, list)):
            argv += [flag, json.dumps(val, separators=(",", ":"))]
        else:
            argv += [flag, str(val)]
    return argv + list(extra)


def env_for(cfg, gpu):
    env = dict(os.environ)
    env.update({k: str(v) for k, v in (cfg.get("env") or {}).items()})
    env["ZE_AFFINITY_MASK"] = str(gpu)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port", type=int)
    ap.add_argument("--dp", action="store_true", help="one replica per GPU in parallel.gpus + load balancer")
    ap.add_argument("--model-path")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--log-dir", default=os.path.join(ROOT, "logs"))
    args, extra = ap.parse_known_args()
    extra = [e for e in extra if e != "--"]
    cfg = load(args.config)
    path = model_path(cfg, args.model_path)

    if not args.dp:
        port = args.port or cfg.get("parallel", {}).get("base_port", 8100)
        argv = vllm_argv(cfg, path, port, extra)
        if args.print:
            print(f"ZE_AFFINITY_MASK={args.gpu} " + shlex.join(argv))
            return
        os.execvpe(argv[0], argv, env_for(cfg, args.gpu))

    par = cfg["parallel"]
    os.makedirs(args.log_dir, exist_ok=True)
    procs, backends = [], []
    for i, gpu in enumerate(par["gpus"]):
        port = par["base_port"] + i
        argv = vllm_argv(cfg, path, port, extra)
        if args.print:
            print(f"ZE_AFFINITY_MASK={gpu} " + shlex.join(argv))
            continue
        log = open(os.path.join(args.log_dir, f"serve{gpu}.log"), "w")
        procs.append(subprocess.Popen(argv, env=env_for(cfg, gpu), stdout=log, stderr=subprocess.STDOUT))
        backends.append(f"http://localhost:{port}")
    lb = [sys.executable, os.path.join(ROOT, "scripts", "lb.py"), "--port", str(par["lb_port"]),
          "--backends", ",".join(backends or [f"http://localhost:{par['base_port'] + i}" for i in range(len(par["gpus"]))])]
    if args.print:
        print(shlex.join(lb))
        return
    procs.append(subprocess.Popen(lb, stdout=open(os.path.join(args.log_dir, "lb.log"), "w"), stderr=subprocess.STDOUT))

    def stop(*_):
        for p in procs:
            p.send_signal(signal.SIGTERM)
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while all(p.poll() is None for p in procs):
        time.sleep(2)
    stop()


if __name__ == "__main__":
    main()
