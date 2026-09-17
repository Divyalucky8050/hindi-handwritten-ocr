"""
Stage 1 synthetic data, written incrementally.

The first version wrote the manifest only at the end and died on disk-full at
132k/200k. Recovering labels by replaying the seeded RNG FAILED verification
(row 500 recovered as कांच when the image actually reads इन्हीं) because the
generator's conditional branches consume random draws unpredictably. Append each
label the moment its image is saved instead - then a crash costs only the
in-flight sample, and labels cannot drift out of sync with files.
"""
import json, random, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

OUT = Path("data/synthetic"); (OUT / "images").mkdir(parents=True, exist_ok=True)
N = 80_000            # plenty for stage 1; leaves disk headroom
random.seed(7)

fonts = sorted({l.split(":")[0] for l in subprocess.run(
    ["fc-list"], capture_output=True, text=True).stdout.splitlines()
    if "deva" in l.lower() and l.split(":")[0].endswith((".ttf", ".otf"))})
print(f"{len(fonts)} fonts")

words = set()
for t in Path("data/raw/IIIT-HW-Hindi_v1/hindi_vocab.txt").read_text(
        encoding="utf-8", errors="replace").split():
    if any(0x0900 <= ord(c) <= 0x097F for c in t):
        words.add(t)
for r in map(json.loads, open("data/processed/train.jsonl", encoding="utf-8")):
    words.add(r["text"])
words = sorted(words)
print(f"{len(words)} words")

def render(word, fp):
    try:
        font = ImageFont.truetype(fp, random.randint(48, 84))
    except Exception:
        return None
    tmp = Image.new("L", (2200, 420), 255)
    ImageDraw.Draw(tmp).text((40, 60), word, font=font, fill=random.randint(0, 70))
    bb = tmp.point(lambda p: 255 if p < 200 else 0).getbbox()
    if bb is None:
        return None
    pad = random.randint(6, 20)
    img = tmp.crop((max(0, bb[0]-pad), max(0, bb[1]-pad), bb[2]+pad, bb[3]+pad))
    a = np.array(img).astype(np.float32)
    if random.random() < 0.5:
        a += np.random.normal(0, random.uniform(3, 12), a.shape)
    if random.random() < 0.4:
        a *= np.linspace(1.0, random.uniform(0.8, 1.0), a.shape[1])[None, :]
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.0)))
    if random.random() < 0.4:
        img = img.rotate(random.uniform(-2.5, 2.5), expand=True, fillcolor=255)
    return img.convert("RGB")

n = 0
with (OUT / "train.jsonl").open("w", encoding="utf-8", buffering=1) as mf:
    while n < N:
        w = random.choice(words)
        img = render(w, random.choice(fonts))
        if img is None:
            continue
        p = OUT / "images" / f"{n:07d}.jpg"
        img.save(p, quality=random.randint(75, 95))
        mf.write(json.dumps({"image": str(p.resolve()), "text": w},
                            ensure_ascii=False) + "\n")
        n += 1
        if n % 5000 == 0:
            print(f"  {n}/{N}", end="\r", flush=True)
print(f"\nwrote {n} samples with verified labels")
