"""
Printed Devanagari characters and conjuncts, to be mixed with the 95k
handwritten words during training.

Conjuncts (samyuktakshara - the Devanagari analogue of Kannada's ottakshara)
are rendered as whole units, not as separate consonants, because they are fused
glyphs: क्ष has no visual boundary between क and ष.
"""
import json, random, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

OUT = Path("data/chars"); (OUT / "images").mkdir(parents=True, exist_ok=True)
random.seed(11)
PER_UNIT = 60

CONS = list("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह")
VOWELS = list("अआइईउऊऋएऐओऔ")
MATRAS = ["", "ा", "ि", "ी", "ु", "ू", "ृ", "े", "ै", "ो", "ौ", "ं", "ः"]
CONJ = ["क्ष","ज्ञ","त्र","श्र","स्त","स्थ","न्द","न्ध","न्त","न्ह","क्क",
        "क्त","ग्र","द्ध","द्व","द्य","ट्ट","ड्ड","प्र","ब्र","म्ह","ल्ल",
        "व्य","श्च","ष्ट","स्व","ह्म","त्त","च्छ","ङ्क","द्र","ह्य","न्न"]

units = set(CONS) | set(VOWELS) | set(CONJ)
for c in CONS:
    for m in MATRAS: units.add(c + m)
for j in CONJ:
    for m in MATRAS[:7]: units.add(j + m)
units = sorted(units)

fonts = sorted({l.split(":")[0] for l in subprocess.run(
    ["fc-list"], capture_output=True, text=True).stdout.splitlines()
    if "deva" in l.lower() and l.split(":")[0].endswith((".ttf", ".otf"))})
print(f"{len(units)} units x {PER_UNIT} = ~{len(units)*PER_UNIT} images, {len(fonts)} fonts")

def render(txt, fp):
    try:
        f = ImageFont.truetype(fp, random.randint(56, 96))
    except Exception: return None
    t = Image.new("L", (700, 420), 255)
    ImageDraw.Draw(t).text((60, 60), txt, font=f, fill=random.randint(0, 60))
    bb = t.point(lambda p: 255 if p < 200 else 0).getbbox()
    if bb is None: return None
    pd = random.randint(8, 22)
    im = t.crop((max(0,bb[0]-pd), max(0,bb[1]-pd), bb[2]+pd, bb[3]+pd))
    a = np.array(im).astype(np.float32)
    if random.random() < 0.5:
        a += np.random.normal(0, random.uniform(3, 12), a.shape)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if random.random() < 0.3:
        im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.9)))
    if random.random() < 0.4:
        im = im.rotate(random.uniform(-3, 3), expand=True, fillcolor=255)
    return im.convert("RGB")

n = 0
with (OUT / "train.jsonl").open("w", encoding="utf-8", buffering=1) as mf:
    for u in units:
        for _ in range(PER_UNIT):
            im = render(u, random.choice(fonts))
            if im is None: continue
            p = OUT / "images" / f"{n:07d}.jpg"
            im.save(p, quality=random.randint(78, 95))
            mf.write(json.dumps({"image": str(p.resolve()), "text": u},
                                ensure_ascii=False) + "\n")
            n += 1
        if n % 5000 < PER_UNIT: print(f"  {n}", end="\r", flush=True)
print(f"\nwrote {n} images over {len(units)} classes")
