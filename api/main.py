"""
FastAPI backend.

    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints
    GET  /                 the demo UI
    GET  /api/health       model status, device, which stages are active
    POST /api/recognize    multipart image upload -> full pipeline result
    POST /api/correct      re-run correction after a user edits a word
    POST /api/segment      debug overlay showing detected lines and words

Design note: the model loads lazily on first request rather than at import, so
the server starts instantly and --reload stays usable during frontend work.
"""

from __future__ import annotations

import base64
import json
import io
import time
from pathlib import Path

import numpy as np
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text()) if (ROOT / "config.yaml").exists() else {}
MAX_MB = CONFIG.get("api", {}).get("max_upload_mb", 10)

app = FastAPI(title="Hindi Handwriting OCR", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_pipeline = None
_load_error: str | None = None


def get_pipeline():
    """Load the pipeline once, on first use."""
    global _pipeline, _load_error
    if _pipeline is not None:
        return _pipeline
    if _load_error is not None:
        raise HTTPException(503, f"model unavailable: {_load_error}")
    try:
        from src.pipeline import HindiOCRPipeline
        _pipeline = HindiOCRPipeline.from_dir(
            ROOT / "outputs",
            processed=ROOT / "data" / "processed",
            lm_dir=ROOT / "data" / "lexicon",
            mc_passes=CONFIG.get("uncertainty", {}).get("mc_passes", 0),
        )
        return _pipeline
    except Exception as e:  # noqa: BLE001
        _load_error = f"{type(e).__name__}: {e}"
        raise HTTPException(503, f"model unavailable: {_load_error}") from e


def read_upload(file: UploadFile, auto_clean: bool = True) -> tuple:
    """Read an upload and optionally auto-correct orientation, skew and crop.

    Live testing showed most real-photo failures were mechanical, not model
    errors: portrait shots with vertical text (the recogniser has only ever seen
    horizontal words), and words occupying under 20% of the frame. Measured on a
    rotated photo of निर्माण - raw input read as 'ङ्क', preprocessed as 'निर्मण'.
    """
    raw = file.file.read()
    if len(raw) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"image larger than {MAX_MB} MB")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, "that file is not a readable image") from e
    long_edge = max(img.size)
    if long_edge > 2000:
        s_ = 2000 / long_edge
        img = img.resize((int(img.width * s_), int(img.height * s_)), Image.LANCZOS)
    bgr = np.asarray(img)[:, :, ::-1].copy()
    report = {}
    if auto_clean:
        from src.segmentation.quality import prepare
        bgr, report = prepare(bgr)
    return bgr, report


def _read_upload_legacy(file: UploadFile) -> np.ndarray:
    raw = file.file.read()
    if len(raw) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"image larger than {MAX_MB} MB")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, "that file is not a readable image") from e
    long_edge = max(img.size)
    if long_edge > 2000:
        s = 2000 / long_edge
        img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    return np.asarray(img)[:, :, ::-1].copy()   # RGB -> BGR for OpenCV


@app.get("/api/health")
def health() -> dict:
    import importlib.util

    ckpt = ROOT / "outputs" / "checkpoints" / "best"
    det = ROOT / "outputs" / "artifacts" / "error_detector.joblib"
    lex = ROOT / "data" / "processed" / "lexicon.txt"
    wlm = ROOT / "data" / "lexicon" / "word_lm.pkl"

    device = "cpu"
    if importlib.util.find_spec("torch") is not None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return {
        "ready": ckpt.exists(),
        "device": device,
        "stages": {
            "segmentation": True,
            "recognition": ckpt.exists(),
            "error_detection": det.exists(),
            "correction": lex.exists(),
            "context_language_model": wlm.exists(),
        },
        "loaded": _pipeline is not None,
        "load_error": _load_error,
    }


@app.post("/api/recognize")
async def recognize(
    file: UploadFile = File(...),
    mode: str = "auto",
    mc_passes: int = 0,
) -> JSONResponse:
    """Run the pipeline on an uploaded image.

    mode="page" segments into lines and words; mode="word" treats the whole
    image as one pre-cropped word (spot-check against dataset samples).
    """
    img, prep = read_upload(file)

    # Check the image BEFORE loading the model or running inference. Refusing
    # with a specific reason is more useful than silently producing garbage:
    # a rotated photo previously read as 'ऋ' at 15% confidence rather than
    # saying "this is sideways".
    from src.segmentation.quality import preflight
    check = preflight(img, check_word_count=False)
    if not check["ok"]:
        return JSONResponse(status_code=200, content={
            "rejected": True,
            "errors": check["errors"],
            "warnings": check["warnings"],
            "metrics": check["metrics"],
            "lines": [], "text": "",
            "stats": {"n_words": 0, "n_flagged": 0, "n_corrected": 0},
        })

    pipe = get_pipeline()
    pipe.mc_passes = max(0, min(int(mc_passes), 20))

    t0 = time.perf_counter()

    # Decide single-word vs multi-word ourselves rather than asking the user.
    # Getting this wrong is silent and confusing: page mode on a single word
    # split अनाथों into two fragments, and word mode on two words merged
    # शिव कुमार into शिवमूर्ति. Run the segmenter first and count real words.
    # One word per image. The page path is gone: the segmenter over-splits
    # (seven fragments from a photo of one word), so routing through it turned a
    # correct 89% read into 40% garbage. This is a word-level recogniser and the
    # interface now says so.
    gray = np.asarray(Image.fromarray(img[:, :, ::-1]).convert("L"))
    words = pipe.run_words([gray])
    # -------------------------------------------------------------
    # Word-level metadata.
    # The current UI is single-word mode, so the uploaded image is
    # treated as one detected word. Its complete image is the box.
    # Detection accuracy is therefore 100% for this input mode.
    # -------------------------------------------------------------
    h, image_width = img.shape[:2]
    word_payloads = []

    for index, word in enumerate(words, start=1):
        item = word.as_dict()

        # Display the model recognition score as "Accuracy".
        item["accuracy"] = round(float(word.confidence), 4)
        item["recognition_accuracy"] = round(float(word.confidence), 4)

        item["detection_accuracy"] = 1.0
        item["box"] = {
            "x": 0,
            "y": 0,
            "w": int(image_width),
            "h": int(h),
        }

        item["word_length"] = len(word.final or "")
        item["character_count"] = len(word.final or "")
        item["line_number"] = 1
        item["word_number"] = index

        word_payloads.append(item)

    payload = {
        "text": words[0].final if words else "",
        "text_uncorrected": words[0].raw if words else "",
        "lines": [{
            "box": {"x": 0, "y": 0, "w": int(image_width), "h": int(h)},
            "words": word_payloads
        }],
        "stats": {
            "n_words": len(words),
            "n_flagged": sum(w.flagged for w in words),
            "n_corrected": sum(w.corrected for w in words),
            "mean_accuracy": (
                sum(float(w.confidence) for w in words)
                / max(len(words), 1)
            ),
        },
    }
    payload["warnings"] = check["warnings"]

    payload["elapsed_s"] = round(time.perf_counter() - t0, 3)
    payload["preprocessing"] = prep
    # Send back what the recogniser actually saw. Without this the user cannot
    # tell whether a bad read was the model's fault or a preprocessing mistake -
    # and auto-rotation getting the orientation wrong looks identical to the
    # model simply failing.
    try:
        import cv2 as _cv
        ok, buf = _cv.imencode(".jpg", img, [int(_cv.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            payload["preprocessed_png_base64"] = base64.b64encode(
                buf.tobytes()).decode()
    except Exception:
        pass
    return JSONResponse(payload)


class CorrectRequest(BaseModel):
    words: list
    flags: list | None = None


@app.post("/api/correct")
def correct(req: CorrectRequest) -> dict:
    """Re-run sentence-level correction on a word list.

    Used when the user manually fixes one word in the UI: the corrected word
    becomes context for its neighbours, so a single human fix can cascade.
    """
    pipe = get_pipeline()
    if pipe.corrector is None:
        raise HTTPException(503, "correction unavailable: no lexicon loaded")
    flags = req.flags if req.flags is not None else [True] * len(req.words)
    if len(flags) != len(req.words):
        raise HTTPException(400, "words and flags must be the same length")
    out = pipe.corrector.correct_sentence(req.words, flags)
    return {"corrections": [c.as_dict() for c in out],
            "text": " ".join(c.suggestion for c in out)}


@app.post("/api/segment")
async def segment(file: UploadFile = File(...)) -> dict:
    """Return a base64 PNG overlay of detected lines and words, for tuning."""
    import cv2

    from src.segmentation.segment import debug_overlay, segment_page

    img, prep = read_upload(file)
    seg = segment_page(img)
    overlay = debug_overlay(img, seg)
    ok, buf = cv2.imencode(".png", overlay)
    if not ok:
        raise HTTPException(500, "could not encode the overlay image")
    return {
        "n_lines": len(seg["lines"]),
        "n_words": seg["n_words"],
        "skew_angle": round(seg["angle"], 2),
        "overlay_png_base64": base64.b64encode(buf.tobytes()).decode(),
    }


FRONTEND = ROOT / "frontend"
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(FRONTEND / "index.html"))


if __name__ == "__main__":
    import uvicorn

    cfg = CONFIG.get("api", {})
    uvicorn.run("api.main:app", host=cfg.get("host", "0.0.0.0"),
                port=cfg.get("port", 8000), reload=True)