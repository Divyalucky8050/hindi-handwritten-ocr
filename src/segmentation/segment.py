"""
Page -> lines -> words segmentation for handwritten Devanagari.

Why we need this at all
-----------------------
IIIT-HW-Dev is a *word-level* dataset, so the recogniser is trained on cropped
word images. But a demo where the user must pre-crop every word is not a usable
system, and the sentence-level language model in the correction stage needs word
*order* to be meaningful. This module bridges the two: it takes a photo of a page
and emits word crops in reading order.

The Devanagari-specific trick
-----------------------------
For Latin handwriting, line segmentation by horizontal projection profile is
fragile because ascenders and descenders from adjacent lines interleave.
Devanagari makes it much easier: every word hangs from a *shirorekha*, the solid
horizontal headline bar. In a horizontal projection profile that bar shows up as
a sharp, tall spike - far sharper than anything in Latin script. So we can find
lines by locating shirorekha spikes rather than by thresholding ink density,
which is dramatically more robust to skew and to varying line spacing.

Word splitting then uses vertical projection, with the gap threshold derived
from the *observed* inter-component gap distribution rather than a fixed pixel
count, so it adapts to image resolution automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int

    def crop(self, img: np.ndarray, pad: int = 4) -> np.ndarray:
        H, W = img.shape[:2]
        y0, y1 = max(0, self.y - pad), min(H, self.y + self.h + pad)
        x0, x1 = max(0, self.x - pad), min(W, self.x + self.w + pad)
        return img[y0:y1, x0:x1]

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


def binarize(img: np.ndarray) -> np.ndarray:
    """Return a binary image with ink = 255, background = 0.

    Adaptive thresholding beats Otsu on phone photos of paper, where a single
    global threshold loses text in the shadowed corner.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=35, C=11,
    )
    # Remove speckle: components smaller than a tiny fraction of the page.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    min_area = max(6, int(0.000004 * bw.size))
    keep = np.zeros_like(bw)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return keep


def estimate_skew(bw: np.ndarray, max_angle: float = 8.0, step: float = 0.25) -> float:
    """Find the rotation that maximises shirorekha spike sharpness.

    We search for the angle whose horizontal projection profile has the highest
    variance. Because shirorekhas are solid horizontal bars, they produce the
    tallest possible spikes exactly when they are level - so variance peaks at
    the correct deskew angle. This is more reliable here than Hough lines, which
    tend to lock onto ruled notebook lines instead of the text.
    """
    best_angle, best_score = 0.0, -1.0
    h, w = bw.shape
    for angle in np.arange(-max_angle, max_angle + step, step):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rot = cv2.warpAffine(bw, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        profile = rot.sum(axis=1).astype(np.float64)
        score = profile.var()
        if score > best_score:
            best_score, best_angle = score, float(angle)
    return best_angle


def deskew(img: np.ndarray, bw: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    angle = estimate_skew(bw)
    if abs(angle) < 0.2:
        return img, bw, 0.0
    h, w = bw.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img_r = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                           borderValue=(255, 255, 255))
    bw_r = cv2.warpAffine(bw, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return img_r, bw_r, angle


def find_lines(bw: np.ndarray, min_line_height: int = 12) -> list[Box]:
    """Segment into text lines using the horizontal projection profile."""
    profile = bw.sum(axis=1).astype(np.float64) / 255.0
    if profile.max() <= 0:
        return []
    # Smooth so that a matra sitting above the shirorekha does not split a line.
    k = max(3, int(bw.shape[0] * 0.004) | 1)
    smooth = cv2.GaussianBlur(profile.reshape(-1, 1), (1, k), 0).ravel()

    thresh = 0.08 * smooth.max()
    inked = smooth > thresh

    bands, start = [], None
    for i, v in enumerate(inked):
        if v and start is None:
            start = i
        elif not v and start is not None:
            bands.append((start, i))
            start = None
    if start is not None:
        bands.append((start, len(inked)))

    lines = []
    for y0, y1 in bands:
        if y1 - y0 < min_line_height:
            continue
        cols = np.where(bw[y0:y1].sum(axis=0) > 0)[0]
        if cols.size == 0:
            continue
        lines.append(Box(int(cols[0]), int(y0), int(cols[-1] - cols[0] + 1), int(y1 - y0)))
    return lines


def _gap_threshold(gaps: list[int], line_height: int) -> float:
    """Decide which whitespace runs separate words rather than aksharas.

    Inter-word and inter-akshara gaps form two clusters, so we look for the
    largest *multiplicative* jump in the sorted gap widths and cut there. A high
    percentile does not work: when a line happens to contain no intra-word gaps
    at all (common in Devanagari, because the shirorekha physically connects the
    aksharas of a word) every observed gap is already a word gap, and scaling the
    75th percentile upward pushes the threshold above all of them, merging the
    entire line into one word.

    When no clear jump exists, all gaps belong to one cluster and we decide which
    cluster that is by comparing against line height - word gaps in handwriting
    run roughly 0.3-1.0x the line height, akshara gaps well under that.
    """
    if not gaps:
        return float("inf")

    uniq = sorted(set(gaps))
    if len(uniq) > 1:
        best_ratio, best_cut = 1.0, None
        for a, b in zip(uniq[:-1], uniq[1:]):
            ratio = b / max(a, 1)
            if ratio > best_ratio:
                best_ratio, best_cut = ratio, float(np.sqrt(a * b))
        if best_ratio >= 1.6 and best_cut is not None:
            return best_cut

    # Single cluster: are these word gaps or akshara gaps?
    median_gap = float(np.median(gaps))
    if median_gap >= 0.30 * line_height:
        return median_gap * 0.6      # all of them separate words
    return float("inf")              # none of them do; the line is one word


def find_words(bw_line: np.ndarray, x_offset: int = 0, y_offset: int = 0,
               min_word_width: int = 8) -> list[Box]:
    """Split one line into words using adaptive gap thresholding.

    The threshold is derived from the observed gap distribution, so the same code
    works whether the page was photographed at 800px or 4000px wide. A fixed
    pixel threshold silently fails on resolution change.
    """
    profile = bw_line.sum(axis=0)
    inked = profile > 0
    if not inked.any():
        return []

    # Interior gap runs only - ignore leading/trailing whitespace.
    first, last = int(np.argmax(inked)), int(len(inked) - np.argmax(inked[::-1]))
    gaps, start = [], None
    for i in range(first, last):
        if not inked[i] and start is None:
            start = i
        elif inked[i] and start is not None:
            gaps.append(i - start)
            start = None

    gap_thresh = max(float(min_word_width),
                     _gap_threshold(gaps, bw_line.shape[0]))

    words, start = [], None
    run_gap = 0
    for i, v in enumerate(inked):
        if v:
            if start is None:
                start = i
            run_gap = 0
        else:
            if start is not None:
                run_gap += 1
                if run_gap >= gap_thresh:
                    end = i - run_gap + 1
                    if end - start >= min_word_width:
                        words.append((start, end))
                    start, run_gap = None, 0
    if start is not None:
        words.append((start, len(inked)))

    out = []
    for x0, x1 in words:
        col = bw_line[:, x0:x1]
        rows = np.where(col.sum(axis=1) > 0)[0]
        if rows.size == 0:
            continue
        out.append(Box(int(x0 + x_offset), int(rows[0] + y_offset),
                       int(x1 - x0), int(rows[-1] - rows[0] + 1)))
    return out


def segment_page(img: np.ndarray, do_deskew: bool = True) -> dict:
    """Full pipeline: photo -> ordered word crops.

    Returns:
        {
          "angle": float,
          "lines": [{"box": {...}, "words": [{"box": {...}, "image": ndarray}]}],
          "n_words": int,
        }
    """
    bw = binarize(img)
    angle = 0.0
    if do_deskew:
        img, bw, angle = deskew(img, bw)

    result = {"angle": angle, "lines": [], "n_words": 0}
    for line_box in find_lines(bw):
        y0, y1 = line_box.y, line_box.y + line_box.h
        x0, x1 = line_box.x, line_box.x + line_box.w
        word_boxes = find_words(bw[y0:y1, x0:x1], x_offset=x0, y_offset=y0)
        words = [{"box": wb.as_dict(), "image": wb.crop(img)} for wb in word_boxes]
        result["lines"].append({"box": line_box.as_dict(), "words": words})
        result["n_words"] += len(words)
    return result


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    # Cap the long edge: phone photos are 4000px+, and segmentation quality does
    # not improve past ~2000px while runtime scales quadratically.
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge > 2000:
        s = 2000 / long_edge
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def debug_overlay(img: np.ndarray, seg: dict) -> np.ndarray:
    """Draw detected lines and words - use this when tuning on your own photos."""
    out = img.copy()
    for line in seg["lines"]:
        b = line["box"]
        cv2.rectangle(out, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]),
                      (255, 140, 0), 1)
        for word in line["words"]:
            wb = word["box"]
            cv2.rectangle(out, (wb["x"], wb["y"]),
                          (wb["x"] + wb["w"], wb["y"] + wb["h"]), (0, 60, 200), 2)
    return outs