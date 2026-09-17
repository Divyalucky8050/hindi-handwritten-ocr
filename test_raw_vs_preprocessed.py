import sys
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

from src.script import DevanagariCharTokenizer
from src.segmentation.quality import prepare


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

CKPT = Path("outputs/long/best")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("Loading model...")
print(f"Checkpoint: {CKPT}")
print(f"Device: {DEVICE}")

tok = DevanagariCharTokenizer.load(CKPT / "charset.json")
proc = TrOCRProcessor.from_pretrained(CKPT)

model = VisionEncoderDecoderModel.from_pretrained(CKPT)
model = model.to(DEVICE).eval()

print(f"Tokenizer: {len(tok)} tokens")
print()


# ------------------------------------------------------------
# RECOGNITION
# ------------------------------------------------------------

@torch.no_grad()
def recognize(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)

    pixel_values = proc(
        images=pil,
        return_tensors="pt"
    ).pixel_values.to(DEVICE)

    output = model.generate(
        pixel_values,
        max_length=24,
        num_beams=4,
        length_penalty=0.7,
        no_repeat_ngram_size=0,
        early_stopping=True,
    )

    text = tok.decode(
        output[0].cpu().tolist(),
        skip_special_tokens=True
    )

    return text.strip()


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if len(sys.argv) != 2:
    print("Usage:")
    print(r"python test_raw_vs_preprocessed.py C:\path\to\image.jpg")
    sys.exit(1)

image_path = Path(sys.argv[1])

if not image_path.exists():
    print(f"ERROR: Image not found:")
    print(image_path)
    sys.exit(1)

img = cv2.imread(str(image_path))

if img is None:
    print("ERROR: Could not read image.")
    sys.exit(1)

print("Image:")
print(f"  {image_path}")
print(f"  size = {img.shape[1]} x {img.shape[0]}")
print()


# ------------------------------------------------------------
# TEST 1: RAW IMAGE
# ------------------------------------------------------------

print("=" * 60)
print("TEST 1: RAW IMAGE")
print("=" * 60)

raw_result = recognize(img)

print(f"RAW RESULT: {raw_result!r}")
print()


# ------------------------------------------------------------
# TEST 2: PREPROCESSED IMAGE
# ------------------------------------------------------------

print("=" * 60)
print("TEST 2: CURRENT PREPROCESSING")
print("=" * 60)

try:
    processed, report = prepare(img.copy())

    print(f"Preprocessing applied: {report.get('applied', [])}")
    print(f"Final size: {report.get('final_size')}")
    print()

    processed_result = recognize(processed)

    print(f"PROCESSED RESULT: {processed_result!r}")

except Exception as e:
    print("PREPROCESSING FAILED:")
    print(type(e).__name__, str(e))
    processed_result = None

print()


# ------------------------------------------------------------
# COMPARISON
# ------------------------------------------------------------

print("=" * 60)
print("COMPARISON")
print("=" * 60)

print(f"Raw:        {raw_result!r}")
print(f"Processed:  {processed_result!r}")
print()

if processed_result is not None:
    if raw_result == processed_result:
        print("RESULT: preprocessing does not change the prediction.")
    else:
        print("RESULT: preprocessing CHANGES the prediction.")
        print("We need to determine which version is better.")