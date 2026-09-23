"""Video smoke test: a red ball moving left -> right on white; the model must name colour and direction."""
import base64, json, os, subprocess, sys, tempfile, urllib.request
import cv2, numpy as np
base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8100"
tmp = tempfile.mkdtemp()
raw, mp4 = os.path.join(tmp, "raw.avi"), os.path.join(tmp, "ball.mp4")
w = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"MJPG"), 8, (320, 240))
for i in range(24):
    f = np.full((240, 320, 3), 255, np.uint8)
    cv2.circle(f, (30 + i * 11, 120), 22, (0, 0, 220), -1)   # BGR red
    w.write(f)
w.release()
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4], check=True)
url = "data:video/mp4;base64," + base64.b64encode(open(mp4, "rb").read()).decode()
body = {"model": "qwen3.8-27b-exl3", "temperature": 0, "max_tokens": 200,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": url}},
            {"type": "text", "text": "What colour is the moving object, what shape is it, and which direction does it move? Be brief."}]}]}
req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
out = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"]
print(out)
low = out.lower()
ok = "red" in low and ("right" in low) and ("circle" in low or "ball" in low or "round" in low)
print("VIDEO_PASS" if ok else "VIDEO_FAIL")
