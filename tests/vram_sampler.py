"""Sample resident VRAM of every process holding the card (DRM fdinfo drm-resident-vram0 / drm-total-vram0) every
50 ms; print the running peak per process each second. Run inside the container (root) while reproducing a load."""
import glob, os, re, sys, time
dur = float(sys.argv[1]) if len(sys.argv) > 1 else 120
peak, t0, last = {}, time.time(), 0
while time.time() - t0 < dur:
    for fd in glob.glob("/proc/[0-9]*/fdinfo/*"):
        try:
            s = open(fd).read()
        except Exception:
            continue
        m = re.search(r"drm-(?:resident|total)-vram0:\s+(\d+)\s*(KiB|MiB)?", s)
        if not m:
            continue
        v = int(m.group(1)) * (1024 if m.group(2) == "MiB" else 1) / 2**20   # GiB
        pid = fd.split("/")[2]
        peak[pid] = max(peak.get(pid, 0), v)
    if time.time() - last > 1:
        last = time.time()
        print(f"t={time.time()-t0:5.1f}s " + " ".join(f"{p}:{v:.2f}GiB" for p, v in sorted(peak.items())), flush=True)
    time.sleep(0.05)
