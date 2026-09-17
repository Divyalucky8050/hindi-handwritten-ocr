"""
Pre-flight image checks and preprocessing for recognition-only Hindi OCR.

Pipeline:

    image
      -> quality checks
      -> tight automatic crop
      -> lighting normalization when needed
      -> recognition model

Important:
    Auto-rotation and auto-deskew are intentionally NOT used.

The recognition model performs substantially better when the handwritten
word occupies most of the input image. A test image demonstrated:

    full image  -> अस्वेखर
    tight crop  -> भुवनेश्वरी

Therefore cropping is now attempted for every valid image.
"""

from __future__ import annotations

import cv2
import numpy as np


# ============================================================
# Thresholds
# ============================================================

MIN_ASPECT = 1.25

SHADOW_WARN = 25.0

MIN_CONTRAST = 60.0

BLUR_HEIGHT = 128
MIN_SHARPNESS = 12.0

LOOSE_MARGIN = 0.28


# ============================================================
# Ink detection
# ============================================================

def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """
    Detect dark handwriting while reducing paper/shadow noise.
    """

    g = cv2.bilateralFilter(
        gray,
        7,
        50,
        50,
    )

    bw = cv2.adaptiveThreshold(
        g,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        12,
    )

    # Remove very small connected components.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        bw,
        connectivity=8,
    )

    keep = np.zeros_like(bw)

    min_area = max(
        12,
        int(0.00002 * bw.size),
    )

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            keep[labels == i] = 255

    return keep


# ============================================================
# Sharpness
# ============================================================

def _sharpness(gray: np.ndarray) -> float:
    """
    Measure sharpness after resizing to a fixed height.
    """

    h, w = gray.shape

    if h < 8:
        return 0.0

    scale = BLUR_HEIGHT / h

    small = cv2.resize(
        gray,
        (
            max(8, int(w * scale)),
            BLUR_HEIGHT,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return float(
        cv2.Laplacian(
            small,
            cv2.CV_64F,
        ).var()
    )


# ============================================================
# Image inspection
# ============================================================

def inspect(img: np.ndarray) -> dict:
    """
    Measure the image without changing it.
    """

    if img.ndim == 3:
        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = img

    h, w = gray.shape

    mask = _ink_mask(gray)

    ys, xs = np.where(mask > 0)

    metrics = {
        "width": int(w),
        "height": int(h),

        "aspect": round(
            w / max(h, 1),
            3,
        ),

        "sharpness": round(
            _sharpness(gray),
            2,
        ),

        "contrast": round(
            float(gray.max()) - float(gray.min()),
            1,
        ),

        "shadow": round(
            abs(
                float(
                    gray[
                        :,
                        :max(1, w // 3)
                    ].mean()
                )
                -
                float(
                    gray[
                        :,
                        -max(1, w // 3):
                    ].mean()
                )
            ),
            1,
        ),

        "ink_pixels": int(ys.size),
    }

    if ys.size >= 20:

        metrics["margin"] = round(
            float(
                max(
                    min(
                        ys.min(),
                        h - 1 - ys.max(),
                    ) / h,

                    min(
                        xs.min(),
                        w - 1 - xs.max(),
                    ) / w,
                )
            ),
            3,
        )

    else:
        metrics["margin"] = 0.0

    return metrics


# ============================================================
# Word count
# ============================================================

def count_words(img: np.ndarray) -> int:
    """
    Kept for compatibility.

    The segmenter is NOT trusted for recognition because it can
    over-split a single handwritten word.
    """

    try:

        from src.segmentation.segment import segment_page

        seg = segment_page(
            img,
            do_deskew=False,
        )

        n = int(
            seg.get(
                "n_words",
                1,
            )
        )

        if n <= 1:
            return n

        w = img.shape[1]

        widths = [
            word["box"]["w"]
            for line in seg["lines"]
            for word in line["words"]
        ]

        meaningful = [
            x
            for x in widths
            if x > 0.18 * w
        ]

        return len(meaningful) or 1

    except Exception:
        return 1


# ============================================================
# Pre-flight quality check
# ============================================================

def preflight(
    img: np.ndarray,
    check_word_count: bool = True,
) -> dict:
    """
    Decide whether the image is suitable for recognition.

    Bad images are rejected with a specific reason.
    """

    m = inspect(img)

    errors = []
    warnings = []

    # --------------------------------------------------------
    # No writing
    # --------------------------------------------------------

    if m["ink_pixels"] < 40:

        errors.append(
            "No clear writing found. Check the photo is of text "
            "and that the ink is darker than the paper."
        )

        return {
            "ok": False,
            "errors": errors,
            "warnings": warnings,
            "metrics": m,
        }

    # --------------------------------------------------------
    # Aspect
    # --------------------------------------------------------

    if m["aspect"] < MIN_ASPECT:

        errors.append(
            f"This image is taller than it is wide "
            f"(ratio {m['aspect']:.2f}), which usually means "
            f"the photo is rotated. Rotate it so the text reads "
            f"left to right, then upload again."
        )

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    if m["contrast"] < MIN_CONTRAST:

        errors.append(
            f"Not enough contrast between ink and paper "
            f"(measured {m['contrast']:.0f}). Use a darker pen "
            f"or brighter, more even light."
        )

    # --------------------------------------------------------
    # Sharpness
    # --------------------------------------------------------

    if m["sharpness"] < MIN_SHARPNESS:

        errors.append(
            f"The image is too blurry to read reliably "
            f"(sharpness {m['sharpness']:.1f}). Hold the camera "
            f"steady, tap to focus on the text, and retake it."
        )

    # --------------------------------------------------------
    # Word-count check deliberately disabled
    # --------------------------------------------------------

    # We do not reject an image based on segmentation because
    # the segmenter can split one handwritten word into several
    # fragments, especially on ruled paper.

    # --------------------------------------------------------
    # Lighting warning
    # --------------------------------------------------------

    if m["shadow"] > SHADOW_WARN:

        warnings.append(
            f"Uneven lighting detected "
            f"(difference {m['shadow']:.0f}) across the frame. "
            f"Flattening it, but evenly lit photos will read better."
        )

    # --------------------------------------------------------
    # Loose framing warning
    # --------------------------------------------------------

    if m["margin"] > LOOSE_MARGIN:

        warnings.append(
            "Lots of blank paper around the word. "
            "Cropping in automatically; a tighter photo "
            "will read better."
        )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "metrics": m,
    }


# ============================================================
# Tight automatic crop
# ============================================================

def auto_crop(
    img: np.ndarray,
    pad_frac: float = 0.20,
) -> tuple[np.ndarray, bool]:
    """
    Crop tightly around detected handwriting.

    Unlike the previous implementation, this function is intended
    to be attempted on every valid recognition image.

    A small proportional margin is retained around the writing.
    """

    if img.ndim == 3:

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY,
        )

    else:
        gray = img

    mask = _ink_mask(gray)

    ys, xs = np.where(mask > 0)

    if ys.size < 20:
        return img, False

    # --------------------------------------------------------
    # Percentile bounds
    # --------------------------------------------------------

    y0 = int(
        np.percentile(
            ys,
            1.0,
        )
    )

    y1 = int(
        np.percentile(
            ys,
            99.0,
        )
    )

    x0 = int(
        np.percentile(
            xs,
            1.0,
        )
    )

    x1 = int(
        np.percentile(
            xs,
            99.0,
        )
    )

    bh = max(
        1,
        y1 - y0 + 1,
    )

    bw = max(
        1,
        x1 - x0 + 1,
    )

    H, W = mask.shape

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    # If the detected ink covers almost the entire image,
    # cropping provides little benefit.
    if bh * bw > 0.95 * mask.size:
        return img, False

    # --------------------------------------------------------
    # Padding
    # --------------------------------------------------------

    py = int(
        bh * pad_frac
    ) + 4

    px = int(
        bw * pad_frac
    ) + 4

    x0 = max(
        0,
        x0 - px,
    )

    x1 = min(
        W - 1,
        x1 + px,
    )

    y0 = max(
        0,
        y0 - py,
    )

    y1 = min(
        H - 1,
        y1 + py,
    )

    cropped = img[
        y0:y1 + 1,
        x0:x1 + 1,
    ]

    # Don't accept a pathological crop.
    if cropped.size == 0:
        return img, False

    return cropped, True


# ============================================================
# Lighting normalization
# ============================================================

def normalise_lighting(
    img: np.ndarray,
) -> np.ndarray:
    """
    Flatten shadows while preserving grayscale stroke information.

    The image is NOT binarized because the recognizer was trained
    on grayscale information.
    """

    if img.ndim == 3:

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY,
        )

    else:
        gray = img

    bg = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=max(gray.shape) / 12,
    )

    flat = cv2.divide(
        gray,
        bg,
        scale=200,
    )

    flat = cv2.normalize(
        flat,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    return cv2.cvtColor(
        flat.astype(np.uint8),
        cv2.COLOR_GRAY2BGR,
    )


# ============================================================
# Preparation
# ============================================================

def prepare(
    img: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Prepare an image for recognition.

    IMPORTANT:
    Tight cropping is now attempted on every valid image.

    This matches the successful experiment where:

        full image -> अस्वेखर
        tight crop -> भुवनेश्वरी
    """

    m = inspect(img)

    report = {
        "applied": [],
        "original_size": [
            int(img.shape[1]),
            int(img.shape[0]),
        ],
    }

    # --------------------------------------------------------
    # 1. Crop FIRST
    # --------------------------------------------------------

    #cropped, was_cropped = auto_crop(
     #   img,
      #  pad_frac=0.20,
    #)

    #if was_cropped:

     #   img = cropped

      #  report["applied"].append(
       #     "tight crop"
        #)

    # --------------------------------------------------------
    # 2. Lighting normalization
    # --------------------------------------------------------

    if m["shadow"] > 12.0:

        img = normalise_lighting(
            img
        )

        report["applied"].append(
            "lighting flattened"
        )

    # --------------------------------------------------------
    # Final size
    # --------------------------------------------------------

    report["final_size"] = [
        int(img.shape[1]),
        int(img.shape[0]),
    ]

    return img, report
