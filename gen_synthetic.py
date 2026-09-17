"""
Stage 1 of the two-stage routine from the IIIT-H paper: render printed
Devanagari words from fonts, so the decoder learns character-sequence mapping
on clean text before it has to cope with erratic handwriting.

Words come from the dataset's own 11k vocabulary plus the training labels, so
the synthetic set covers the same lexical space as the real data.
"""
import json, random, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

OUT = Path("data/synthetic"); (OUT / "images").mkdir(parents=True, exist_ok=True)
N = 200_000
random.seed(42)

fonts = [l.split(":")[0] for l in subprocess.run(
    ["fc-list"], capture_output=True, text=True).stdout.splitlines()
    if "deva" in l.lower() and l.split(":")[0].endswith((".ttf", ".otf"))]
fonts = sorted(set(fonts))
print(f"{len(fonts)} Devanagari fonts:")
for f in fonts: print("  ", f)
assert fonts, "no Devanagari fonts found"

words = set()
for line in Path("data/raw/IIIT-HW-Hindi_v1/hindi_vocab.txt").read_text(
        encoding="utf-8", errors="replace").split():
    if any(0x0900 <= ord(c) <= 0x097F for c in line):
        words.add(line)
for r in map(json.loads, open("data/processed/train.jsonl", encoding="utf-8")):
    words.add(r["text"])
words = sorted(words)
print(f"\n{len(words)} unique words available")

def render(word, font_path):
    size = random.randint(48, 84)
    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        return None
    tmp = Image.new("L", (2000, 400), 255)
    d = ImageDraw.Draw(tmp)
    d.text((40, 60), word, font=font, fill=random.randint(0, 70))
    bbox = tmp.point(lambda p: 255 if p < 200 else 0).getbbox()
    if bbox is None:
        return None
    pad = random.randint(6, 20)
    img = tmp.crop((max(0, bbox[0]-pad), max(0, bbox[1]-pad),
                    bbox[2]+pad, bbox[3]+pad))

    a = np.array(img).astype(np.float32)
    # Mimic scanning artefacts so Stage 2 is not a distribution shock.
    if random.random() < 0.5:
        a += np.random.normal(0, random.uniform(3, 12), a.shape)
    if random.random() < 0.4:
        g = np.linspace(1.0, random.uniform(0.8, 1.0), a.shape[1])[None, :]
        a *= g
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.0)))
    if random.random() < 0.4:
        # Slight rotation only: beyond a few degrees the shirorekha stops being
        # horizontal, which is the strongest cue the encoder has.
        img = img.rotate(random.uniform(-2.5, 2.5), expand=True, fillcolor=255)
    return img.convert("RGB")

rows, i = [], 0
while len(rows) < N:
    w = random.choice(words)
    img = render(w, random.choice(fonts))
    if img is None:
        continue
    p = OUT / "images" / f"{i:07d}.jpg"
    img.save(p, quality=random.randint(75, 95))
    rows.append({"image": str(p.resolve()), "text": w})
    i += 1
    if len(rows) % 10000 == 0:
        print(f"  {len(rows)}/{N}", end="\r")

with (OUT / "train.jsonl").open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\nwrote {len(rows)} synthetic samples -> {OUT}/train.jsonl")
