"""Dataset and collator for word-image -> text training."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.script import DevanagariCharTokenizer


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class HandwritingAugment:
    """Geometric + photometric augmentation tuned for handwritten word images.

    Deliberately excluded:
      * Horizontal flips - mirror-image Devanagari is not a thing a writer
        produces, so this only teaches the encoder to waste capacity.
      * Aggressive rotation - beyond ~4 degrees the shirorekha (the horizontal
        headline bar every word hangs from) stops being horizontal, and that bar
        is the single strongest visual cue the encoder has for word baseline.
      * Vertical shear - distorts matra positions above/below the bar, which is
        exactly the distinction between similar-looking words.
    """

    def __init__(self, p: float = 0.5, max_rotate: float = 3.0,
                 max_shear: float = 0.15, seed: int | None = None):
        self.p = p
        self.max_rotate = max_rotate
        self.max_shear = max_shear
        self.rng = random.Random(seed)

    def __call__(self, img: Image.Image) -> Image.Image:
        import cv2

        if self.rng.random() > self.p:
            return img
        a = np.array(img.convert("L"))
        h, w = a.shape

        # Horizontal shear only - mimics natural slant variation between writers.
        if self.rng.random() < 0.6:
            s = self.rng.uniform(-self.max_shear, self.max_shear)
            M = np.float32([[1, s, -s * h / 2], [0, 1, 0]])
            a = cv2.warpAffine(a, M, (w, h), borderValue=255,
                               flags=cv2.INTER_LINEAR)

        if self.rng.random() < 0.5:
            ang = self.rng.uniform(-self.max_rotate, self.max_rotate)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
            a = cv2.warpAffine(a, M, (w, h), borderValue=255,
                               flags=cv2.INTER_LINEAR)

        # Stroke thickness: erode thickens dark ink, dilate thins it.
        if self.rng.random() < 0.4:
            k = np.ones((2, 2), np.uint8)
            a = cv2.erode(a, k) if self.rng.random() < 0.5 else cv2.dilate(a, k)

        if self.rng.random() < 0.4:
            g = self.rng.uniform(0.7, 1.4)
            a = np.clip(255.0 * (a / 255.0) ** g, 0, 255).astype(np.uint8)

        if self.rng.random() < 0.3:
            a = np.clip(a.astype(np.int16) +
                        self.rng.gauss(0, 6) * np.ones_like(a, dtype=np.int16),
                        0, 255).astype(np.uint8)

        # Grid distortion: warps on a coarse grid, giving uneven baselines and
        # locally varying letter widths - a writer whose line drifts across the
        # page. Distinct from elastic distortion, which is fine stroke wobble.
        if self.rng.random() < 0.25:
            a = self._grid_distort(a)

        # Perspective warp: photographing a page at an angle, the commonest
        # real-world distortion when a user snaps a picture instead of scanning.
        # Kept mild so the shirorekha stays near-horizontal.
        if self.rng.random() < 0.2:
            a = self._perspective(a)

        return Image.fromarray(a).convert("RGB")

    def _grid_distort(self, a, steps: int = 5, limit: float = 0.12):
        import cv2
        h, w = a.shape
        xs = np.linspace(0, w, steps + 1)
        ys = np.linspace(0, h, steps + 1)
        map_x = np.zeros((h, w), np.float32)
        map_y = np.zeros((h, w), np.float32)
        for i in range(steps):
            for j in range(steps):
                x0, x1 = int(xs[j]), int(xs[j + 1])
                y0, y1 = int(ys[i]), int(ys[i + 1])
                if x1 <= x0 or y1 <= y0:
                    continue
                dx = self.rng.uniform(-limit, limit) * (x1 - x0)
                dy = self.rng.uniform(-limit, limit) * (y1 - y0)
                gx, gy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
                map_x[y0:y1, x0:x1] = gx + dx
                map_y[y0:y1, x0:x1] = gy + dy
        return cv2.remap(a, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    def _perspective(self, a, limit: float = 0.06):
        import cv2
        h, w = a.shape
        d = lambda n: self.rng.uniform(-limit, limit) * n
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[d(w), d(h)], [w + d(w), d(h)],
                          [w + d(w), h + d(h)], [d(w), h + d(h)]])
        return cv2.warpPerspective(a, cv2.getPerspectiveTransform(src, dst),
                                   (w, h), borderValue=255)


class WordImageDataset(Dataset):
    """Yields (pixel_values, labels) pairs.

    `processor` is a transformers TrOCRProcessor / ViTImageProcessor - we use it
    only for image normalisation, never for text, because its tokenizer has no
    Devanagari coverage. Text goes through DevanagariCharTokenizer instead.
    """

    def __init__(
        self,
        rows: list[dict],
        processor,
        tokenizer: DevanagariCharTokenizer,
        max_target_length: int = 48,
        augment: HandwritingAugment | None = None,
    ):
        self.rows = rows
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        try:
            img = Image.open(row["image"]).convert("RGB")
        except Exception:
            # A handful of corrupt files in a 95k-image dataset should not kill a
            # 6-hour training run. Substitute blank and let the loss absorb it.
            img = Image.new("RGB", (384, 384), color="white")
        if self.augment is not None:
            img = self.augment(img)

        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values[0]
        # HuggingFace tokenizer interface: it adds the model's own special tokens
        # (cls/sep) itself. Do NOT hand-roll BOS/EOS here - getting that wrong is
        # what broke the earlier character-level design.
        # Two tokenizer conventions coexist: DevanagariCharTokenizer (stage 1,
        # TrOCRForCausalLM decoder, plain labels) and the HF tokenizer for the
        # aayushpuri checkpoint (RobertaForCausalLM, needs explicit
        # decoder_input_ids because it shifts labels internally). Detect rather
        # than hard-code, or one path silently trains on misaligned labels.
        if hasattr(self.tokenizer, "stoi"):
            ids = self.tokenizer.encode(row["text"],
                                        max_length=self.max_target_length)
        else:
            ids = self.tokenizer(row["text"],
                                 max_length=self.max_target_length,
                                 truncation=True).input_ids
        return {"pixel_values": pixel_values, "labels": ids, "text": row["text"]}


class Collator:
    """Pads label sequences and masks pad positions with -100.

    -100 is the ignore_index torch's cross-entropy uses by default; if you pad
    with the pad token id instead, the model is rewarded for predicting padding
    and short words get systematically truncated at inference.
    """

    def __init__(self, pad_token_id: int, needs_decoder_input_ids: bool = False,
                 decoder_start_token_id: int | None = None):
        self.pad_token_id = pad_token_id
        # transformers 4.57's VisionEncoderDecoderModel does NOT build
        # decoder_input_ids from labels for the TrOCR path - it raises
        # "You have to specify either decoder_input_ids or decoder_inputs_embeds".
        # So we shift them ourselves. RobertaForCausalLM shifts internally and
        # needs the UNSHIFTED sequence instead, hence two modes.
        self.needs_decoder_input_ids = needs_decoder_input_ids
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, batch: list[dict]) -> dict:
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        maxlen = max(len(b["labels"]) for b in batch)
        labels = torch.full((len(batch), maxlen), -100, dtype=torch.long)
        decoder_input_ids = torch.full((len(batch), maxlen),
                                       self.pad_token_id, dtype=torch.long)
        for i, b in enumerate(batch):
            ids = torch.tensor(b["labels"], dtype=torch.long)
            labels[i, : len(ids)] = ids
            decoder_input_ids[i, : len(ids)] = ids

        if self.decoder_start_token_id is not None:
            # DevanagariCharTokenizer.encode already yields
            # [BOS, ...chars..., EOS], so that IS the decoder input. Score
            # against the same sequence shifted left by one: standard
            # next-token prediction. Measured lowest loss of three variants on
            # the untrained model (9.43 vs 9.72 and 9.93). Note ~9.5 is the
            # correct starting loss here - the decoder embeddings are randomly
            # initialised, so it knows no Devanagari at all yet.
            lab = torch.full_like(labels, -100)
            lab[:, :-1] = labels[:, 1:]
            return {"pixel_values": pixel_values, "labels": lab,
                    "decoder_input_ids": decoder_input_ids}

        out = {"pixel_values": pixel_values, "labels": labels}
        # Two decoder conventions in play. TrOCRForCausalLM (our from-scratch
        # runs) expects labels alone and builds decoder_input_ids itself via
        # shift_tokens_right. RobertaForCausalLM (the aayushpuri checkpoint)
        # shifts internally, so passing labels alone double-shifts them -
        # measured loss 15.68 vs 2.90 on the same model. Pass explicit
        # decoder_input_ids only for that second case.
        if self.needs_decoder_input_ids:
            out["decoder_input_ids"] = decoder_input_ids
        return out
        return out