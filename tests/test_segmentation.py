"""Validate segmentation on a synthetic page with Devanagari-like structure.

No Devanagari font is required: we synthesise words as a shirorekha (solid
horizontal bar) with vertical stems hanging below and occasional marks above,
which reproduces the exact structural properties the segmenter relies on.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.segmentation.segment import binarize, find_lines, find_words, segment_page


def synth_word(n_aksharas: int, rng, akshara_w: int = 22, height: int = 46) -> np.ndarray:
    """One word: shirorekha bar on top, stems below, some marks above."""
    w = n_aksharas * akshara_w
    canvas = np.full((height, w), 255, dtype=np.uint8)
    bar_y = int(height * 0.30)

    # shirorekha: continuous bar across the whole word
    canvas[bar_y:bar_y + 3, 1:w - 1] = 0

    for i in range(n_aksharas):
        x0 = i * akshara_w + 3
        stem_x = x0 + akshara_w // 2
        # vertical stem below the bar
        canvas[bar_y:int(height * 0.82), stem_x:stem_x + 3] = 0
        # a curved element to the left of the stem
        canvas[bar_y + 4:bar_y + 14, x0:x0 + 3] = 0
        canvas[bar_y + 12:bar_y + 15, x0:stem_x] = 0
        # matra above the bar, sometimes
        if rng.random() < 0.4:
            canvas[max(0, bar_y - 9):bar_y - 4, stem_x - 2:stem_x + 4] = 0
        # descender below, sometimes
        if rng.random() < 0.25:
            canvas[int(height * 0.80):height - 2, stem_x - 4:stem_x - 1] = 0
    return canvas


def synth_page(n_lines=4, seed=0):
    rng = np.random.default_rng(seed)
    W, H = 900, 420
    page = np.full((H, W), 250, dtype=np.uint8)
    truth = []
    y = 40
    line_gap = 88
    for li in range(n_lines):
        x = 50
        n_words = int(rng.integers(3, 6))
        counts = []
        for _ in range(n_words):
            na = int(rng.integers(2, 5))
            word = synth_word(na, rng)
            wh, ww = word.shape
            if x + ww > W - 40:
                break
            region = page[y:y + wh, x:x + ww]
            page[y:y + wh, x:x + ww] = np.minimum(region, word)
            counts.append(na)
            x += ww + int(rng.integers(30, 44))   # inter-word gap
        truth.append(counts)
        y += line_gap
    # mild illumination gradient, like a phone photo
    grad = np.linspace(1.0, 0.82, W)[None, :]
    page = np.clip(page * grad, 0, 255).astype(np.uint8)
    page = np.stack([page] * 3, axis=-1)
    return page, truth


def main() -> int:
    page, truth = synth_page(n_lines=4, seed=7)
    expected_lines = len(truth)
    expected_words = sum(len(c) for c in truth)

    seg = segment_page(page, do_deskew=True)
    got_lines = len(seg["lines"])
    got_words = seg["n_words"]

    print(f"skew estimate      : {seg['angle']:.2f} deg (truth 0.00)")
    print(f"lines  expected {expected_lines}  got {got_lines}")
    print(f"words  expected {expected_words}  got {got_words}")
    for i, line in enumerate(seg["lines"]):
        print(f"  line {i}: {len(line['words'])} words "
              f"(expected {len(truth[i]) if i < len(truth) else '?'})")

    ok = True
    if got_lines != expected_lines:
        print("FAIL: line count mismatch")
        ok = False
    if abs(got_words - expected_words) > 1:
        print("FAIL: word count off by more than 1")
        ok = False

    # every crop must be non-empty and contain ink
    for line in seg["lines"]:
        for wd in line["words"]:
            img = wd["image"]
            if img.size == 0:
                print("FAIL: empty crop")
                ok = False
            elif binarize(img).sum() == 0:
                print("FAIL: crop contains no ink")
                ok = False

    # rotated input should be recovered
    import cv2
    h, w = page.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 3.0, 1.0)
    rot = cv2.warpAffine(page, M, (w, h), borderValue=(250, 250, 250))
    seg_r = segment_page(rot, do_deskew=True)
    print(f"\nrotated +3.0 deg -> estimated {seg_r['angle']:.2f}, "
          f"lines {len(seg_r['lines'])}, words {seg_r['n_words']}")
    if abs(seg_r["angle"] + 3.0) > 1.0:
        print("WARN: skew estimate off by more than 1 degree")
    if len(seg_r["lines"]) != expected_lines:
        print("FAIL: line count wrong after rotation")
        ok = False

    print("\nPASS" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())