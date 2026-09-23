"""Vision smoke test: synthetic image with known content -> the model must name it."""
import base64, io, json, sys, urllib.request
from PIL import Image, ImageDraw, ImageFont
base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8100"
img = Image.new("RGB", (512, 384), "white")
d = ImageDraw.Draw(img)
d.ellipse((40, 60, 220, 240), fill=(220, 20, 20))            # red circle, left
d.rectangle((290, 60, 470, 240), fill=(20, 40, 220))         # blue square, right
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 64)
except Exception:
    font = ImageFont.load_default()
d.text((190, 280), "B70", fill=(0, 0, 0), font=font)
buf = io.BytesIO(); img.save(buf, format="PNG")
url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
body = {"model": "qwen3.8-27b-exl3", "temperature": 0, "max_tokens": 300,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": "Describe the shapes, their colors and positions, and the text in this image. Be brief."}]}]}
req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
out = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"]
print(out)
low = out.lower()
ok = all(w in low for w in ["red", "blue", "circle", "square"]) and "b70" in low.replace(" ", "")
print("VISION_PASS" if ok else "VISION_FAIL")
