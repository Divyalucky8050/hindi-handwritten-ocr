"""
Larger synthetic set for the long training run.

Changes from the first pass:
  * printed words 80k -> 250k
  * characters/conjuncts 60 -> 150 samples per class
  * elastic distortion added - warps strokes locally to mimic hand tremor,
    which plain shear/rotation cannot do. This is the augmentation that most
    closely bridges printed fonts to real handwriting.

Manifests are written incrementally: the first attempt wrote them only at the
end, died on disk-full at 132k, and the labels could not be recovered by
replaying the RNG (row 500 came back as कांच for an image reading इन्हीं).
"""
import json, random, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import cv2

random.seed(2027)
CONS = list("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह")
VOWELS = list("अआइईउऊऋएऐओऔ")
MATRAS = ["", "ा", "ि", "ी", "ु", "ू", "ृ", "े", "ै", "ो", "ौ", "ं", "ः"]
CONJ = ["क्ष","ज्ञ","त्र","श्र","स्त","स्थ","न्द","न्ध","न्त","न्ह","क्क",
        "क्त","ग्र","द्ध","द्व","द्य","ट्ट","ड्ड","प्र","ब्र","म्ह","ल्ल",
        "व्य","श्च","ष्ट","स्व","ह्म","त्त","च्छ","ङ्क","द्र","ह्य","न्न"]

fonts = sorted({l.split(":")[0] for l in subprocess.run(
    ["fc-list"], capture_output=True, text=True).stdout.splitlines()
    if "deva" in l.lower() and l.split(":")[0].endswith((".ttf", ".otf"))})
print(f"{len(fonts)} fonts")

def elastic(a, alpha=34, sigma=5):
    """Local random warping - simulates hand tremor and stroke wobble."""
    h, w = a.shape
    dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    return cv2.remap(a, (xx + dx).astype(np.float32), (yy + dy).astype(np.float32),
                     cv2.INTER_LINEAR, borderValue=255)

def render(txt, fp):
    try:
        f = ImageFont.truetype(fp, random.randint(50, 92))
    except Exception:
        return None
    t = Image.new("L", (2400, 460), 255)
    ImageDraw.Draw(t).text((50, 70), txt, font=f, fill=random.randint(0, 70))
    bb = t.point(lambda p: 255 if p < 200 else 0).getbbox()
    if bb is None:
        return None
    pd = random.randint(8, 22)
    a = np.array(t.crop((max(0, bb[0]-pd), max(0, bb[1]-pd), bb[2]+pd, bb[3]+pd)))
    if random.random() < 0.45:
        a = elastic(a, alpha=random.uniform(18, 45), sigma=random.uniform(4, 7))
    if random.random() < 0.35:                       # stroke thickness
        k = np.ones((2, 2), np.uint8)
        a = cv2.erode(a, k) if random.random() < 0.5 else cv2.dilate(a, k)
    a = a.astype(np.float32)
    if random.random() < 0.5:
        a += np.random.normal(0, random.uniform(3, 13), a.shape)
    if random.random() < 0.4:
        a *= np.linspace(1.0, random.uniform(0.78, 1.0), a.shape[1])[None, :]
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if random.random() < 0.3:
        im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.1)))
    if random.random() < 0.45:
        im = im.rotate(random.uniform(-3, 3), expand=True, fillcolor=255)
    return im.convert("RGB")

def build(outdir, items, per_item, tag):
    out = Path(outdir); (out / "images").mkdir(parents=True, exist_ok=True)
    n = 0
    with (out / "train.jsonl").open("w", encoding="utf-8", buffering=1) as mf:
        for it in items:
            for _ in range(per_item):
                im = render(it, random.choice(fonts))
                if im is None:
                    continue
                p = out / "images" / f"{n:07d}.jpg"
                im.save(p, quality=random.randint(76, 95))
                mf.write(json.dumps({"image": str(p.resolve()), "text": it},
                                    ensure_ascii=False) + "\n")
                n += 1
            if n % 10000 < per_item:
                print(f"  {tag} {n}", end="\r", flush=True)
    print(f"\n{tag}: {n} images")
    return n

# characters and conjuncts
units = set(CONS) | set(VOWELS) | set(CONJ)
for c in CONS:
    for m in MATRAS: units.add(c + m)
for j in CONJ:
    for m in MATRAS[:7]: units.add(j + m)
units = sorted(units)
build("data/chars2", units, 150, "chars")

# printed words
words = set()
for t in Path("data/raw/IIIT-HW-Hindi_v1/hindi_vocab.txt").read_text(
        encoding="utf-8", errors="replace").split():
    if any(0x0900 <= ord(c) <= 0x097F for c in t):
        words.add(t)
for r in map(json.loads, open("data/processed/train.jsonl", encoding="utf-8")):
    words.add(r["text"])
words = sorted(words)
per = max(1, 250_000 // len(words))
build("data/synthetic2", words, per, "words")
print("done")
