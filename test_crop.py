import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

from src.script import DevanagariCharTokenizer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CKPT = Path("outputs/long/best")

if len(sys.argv) != 2:
    print("Usage:")
    print(r'python test_crop.py "C:\path\to\image.jpg"')
    sys.exit(1)

IMAGE_PATH = Path(sys.argv[1])

if not IMAGE_PATH.exists():
    print(f"Image not found: {IMAGE_PATH}")
    sys.exit(1)


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

print("Loading model...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

tok = DevanagariCharTokenizer.load(CKPT / "charset.json")
proc = TrOCRProcessor.from_pretrained(CKPT)
model = VisionEncoderDecoderModel.from_pretrained(CKPT)
model.to(device)
model.eval()

print(f"Tokenizer: {len(tok)} tokens")


# ---------------------------------------------------------
# Recognition
# ---------------------------------------------------------

@torch.no_grad()
def recognize(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    inputs = proc(
        images=image,
        return_tensors="pt"
    )

    pixel_values = inputs.pixel_values.to(device)

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


# ---------------------------------------------------------
# Ink detection
# ---------------------------------------------------------

def ink_mask(gray):
    """
    Find dark handwritten strokes while ignoring most paper background.
    """

    filtered = cv2.bilateralFilter(
        gray,
        7,
        50,
        50
    )

    mask = cv2.adaptiveThreshold(
        filtered,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        12
    )

    # Remove tiny noise components
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    cleaned = np.zeros_like(mask)

    min_area = max(
        12,
        int(0.00002 * mask.size)
    )

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


# ---------------------------------------------------------
# Forced tight crop
# ---------------------------------------------------------

def tight_crop(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    mask = ink_mask(gray)

    ys, xs = np.where(mask > 0)

    if len(xs) < 20:
        print("Could not find enough ink pixels.")
        return image

    # Percentile crop prevents a few noise pixels from
    # stretching the bounding box.
    x0 = int(np.percentile(xs, 1))
    x1 = int(np.percentile(xs, 99))

    y0 = int(np.percentile(ys, 1))
    y1 = int(np.percentile(ys, 99))

    h, w = gray.shape

    # Add 10% padding around detected handwriting.
    word_w = max(1, x1 - x0 + 1)
    word_h = max(1, y1 - y0 + 1)

    px = int(word_w * 0.20)
    py = int(word_h * 0.25)

    x0 = max(0, x0 - px)
    x1 = min(w - 1, x1 + px)

    y0 = max(0, y0 - py)
    y1 = min(h - 1, y1 + py)

    cropped = image[y0:y1 + 1, x0:x1 + 1]

    print()
    print("TIGHT CROP")
    print(f"Original size : {w} x {h}")
    print(f"Crop box      : ({x0}, {y0}) -> ({x1}, {y1})")
    print(
        f"Cropped size  : "
        f"{cropped.shape[1]} x {cropped.shape[0]}"
    )

    return cropped


# ---------------------------------------------------------
# Main test
# ---------------------------------------------------------

print()
print("=" * 60)
print("IMAGE")
print("=" * 60)
print(IMAGE_PATH)

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    print("Could not read image.")
    sys.exit(1)

print(
    f"Original image: "
    f"{image.shape[1]} x {image.shape[0]}"
)


# ---------------------------------------------------------
# Test 1: Raw
# ---------------------------------------------------------

print()
print("=" * 60)
print("TEST 1: RAW IMAGE")
print("=" * 60)

raw_result = recognize(image)

print(f"RAW RESULT: {raw_result!r}")


# ---------------------------------------------------------
# Test 2: Forced tight crop
# ---------------------------------------------------------

print()
print("=" * 60)
print("TEST 2: FORCED TIGHT CROP")
print("=" * 60)

cropped = tight_crop(image)

crop_result = recognize(cropped)

print(f"CROP RESULT: {crop_result!r}")


# ---------------------------------------------------------
# Save crop for visual inspection
# ---------------------------------------------------------

crop_path = Path("outputs") / "test_tight_crop.jpg"
crop_path.parent.mkdir(parents=True, exist_ok=True)

cv2.imwrite(
    str(crop_path),
    cropped
)

print()
print(f"Saved crop: {crop_path}")


# ---------------------------------------------------------
# Comparison
# ---------------------------------------------------------

print()
print("=" * 60)
print("COMPARISON")
print("=" * 60)

print(f"Raw:   {raw_result!r}")
print(f"Crop:  {crop_result!r}")

if raw_result == crop_result:
    print()
    print("RESULT: Tight cropping does NOT change the prediction.")
    print("The problem is probably model recognition, not image framing.")
else:
    print()
    print("RESULT: Tight cropping CHANGES the prediction.")
    print("This means image framing/preprocessing affects recognition.")

print("=" * 60)
