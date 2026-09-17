import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

from src.script import DevanagariCharTokenizer, to_aksharas, edit_distance


CKPT = "outputs/long/best"
TEST_FILE = "data/processed/test.jsonl"
OUTPUT = "outputs/reports/error_analysis.json"

BATCH_SIZE = 48
MAX_LENGTH = 24

print("Loading model...")

tok = DevanagariCharTokenizer.load(f"{CKPT}/charset.json")
proc = TrOCRProcessor.from_pretrained(CKPT)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VisionEncoderDecoderModel.from_pretrained(CKPT).to(device).eval()

rows = [
    json.loads(line)
    for line in open(TEST_FILE, encoding="utf-8")
]

print(f"Device: {device}")
print(f"Test images: {len(rows)}")

preds = []

t0 = time.time()

for i in range(0, len(rows), BATCH_SIZE):
    batch = rows[i:i + BATCH_SIZE]

    images = [
        Image.open(r["image"]).convert("RGB")
        for r in batch
    ]

    pixel_values = proc(
        images=images,
        return_tensors="pt"
    ).pixel_values.to(device)

    with torch.no_grad():
        output = model.generate(
            pixel_values,
            max_length=MAX_LENGTH,
            num_beams=4,
            length_penalty=0.7,
            no_repeat_ngram_size=0,
        )

    batch_preds = tok.batch_decode(
        output.cpu().tolist()
    )

    preds.extend([p.strip() for p in batch_preds])

    done = min(i + BATCH_SIZE, len(rows))
    print(f"\rProcessed {done}/{len(rows)}", end="")

print()
print(f"Recognition time: {time.time() - t0:.1f}s")


errors = []
correct = 0

confusions = Counter()
length_errors = Counter()
edit_distance_counts = Counter()

for row, pred in zip(rows, preds):
    truth = row["text"].strip()

    ref_a = to_aksharas(truth)
    pred_a = to_aksharas(pred)

    distance = edit_distance(ref_a, pred_a)

    is_correct = ref_a == pred_a

    if is_correct:
        correct += 1
        continue

    length_errors[
        (len(ref_a), len(pred_a))
    ] += 1

    edit_distance_counts[distance] += 1

    # Character/akshara substitutions at aligned positions.
    for r, p in zip(ref_a, pred_a):
        if r != p:
            confusions[(r, p)] += 1

    errors.append({
        "image": row["image"],
        "truth": truth,
        "prediction": pred,
        "reference_aksharas": ref_a,
        "prediction_aksharas": pred_a,
        "edit_distance": distance,
        "reference_length": len(ref_a),
        "prediction_length": len(pred_a),
    })


n = len(rows)
accuracy = correct / max(n, 1)

report = {
    "checkpoint": CKPT,
    "test_images": n,
    "correct": correct,
    "incorrect": n - correct,
    "word_accuracy": accuracy,

    "edit_distance_distribution": {
        str(k): v
        for k, v in sorted(edit_distance_counts.items())
    },

    "length_errors": [
        {
            "reference_length": r,
            "prediction_length": p,
            "count": c,
        }
        for (r, p), c in length_errors.most_common()
    ],

    "top_confusions": [
        {
            "expected": r,
            "predicted": p,
            "count": c,
        }
        for (r, p), c in confusions.most_common(50)
    ],

    "errors": errors,
}

Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(
        report,
        f,
        ensure_ascii=False,
        indent=2
    )

print()
print("=" * 60)
print(f"WORD ACCURACY : {accuracy * 100:.2f}%")
print(f"CORRECT       : {correct}")
print(f"INCORRECT     : {n - correct}")
print("=" * 60)

print("\nTOP CONFUSIONS")
for x in report["top_confusions"][:20]:
    print(
        f"{x['expected']} -> {x['predicted']}"
        f"   {x['count']} times"
    )

print("\nEDIT DISTANCE")
for d, c in report["edit_distance_distribution"].items():
    print(f"{d} edits : {c}")

print(f"\nSaved: {OUTPUT}")