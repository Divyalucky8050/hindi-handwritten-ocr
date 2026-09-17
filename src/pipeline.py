"""
Recognition-only Hindi handwritten OCR pipeline.

Pipeline:
    Image
      -> quality check
      -> crop / lighting normalization
      -> trained TrOCR model
      -> raw Hindi recognition
      -> word confidence
      -> character confidence

Disabled:
    - Lexicon
    - Correction
    - Error detector
    - Language model

Model:
    outputs/long/best
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.inference.uncertainty import predict_with_uncertainty
from src.models.build import load_trained
from src.segmentation.quality import preflight, prepare


@dataclass
class WordResult:
    """Recognition-only result."""

    final: str
    raw: str

    confidence: float = 0.0

    # Character/token confidence
    char_probs: list = field(default_factory=list)
    entropy_per_char: list = field(default_factory=list)

    # Compatibility fields for existing API
    flagged: bool = False
    corrected: bool = False
    alternatives: list = field(default_factory=list)
    reasons: list = field(default_factory=list)

    rejected: bool = False
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    weakest_position: int = -1

    # Optional evaluation metric
    error_rate: float | None = None

    @property
    def word_confidence(self) -> float:
        return self.confidence

    @property
    def character_confidence(self) -> list:
        return self.char_probs

    def as_dict(self) -> dict:
        return {
            "text": self.final,
            "raw": self.raw,

            "word_confidence": round(
                float(self.confidence),
                4,
            ),

            "accuracy": round(
                float(self.confidence),
                4,
            ),

            "recognition_accuracy": round(
                float(self.confidence),
                4,
            ),

            "characters": [
                {
                    "char": str(char),
                    "accuracy": round(
                        float(prob),
                        4,
                    ),
                }
                for char, prob in self.char_probs
            ],

            "char_probs": [
                [
                    str(char),
                    round(float(prob), 4),
                ]
                for char, prob in self.char_probs
            ],

            "weakest_position": self.weakest_position,

            "character_error_rate": (
                round(
                    float(self.error_rate),
                    4,
                )
                if self.error_rate is not None
                else None
            ),

            # Recognition-only compatibility values
            "flagged": False,
            "corrected": False,
            "alternatives": [],
            "reasons": [],
            "rejected": self.rejected,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class HindiOCRPipeline:

    def __init__(
        self,
        model,
        processor,
        tokenizer,
        device: str | None = None,
        max_length: int = 32,
        mc_passes: int = 0,
    ):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.max_length = max_length
        self.mc_passes = mc_passes

        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Device: {self.device}")
        print("Recognition-only pipeline")
        print("Correction: disabled")
        print("Lexicon: disabled")
        print("Error detection: disabled")
        print("Language model: disabled")

    # =========================================================
    # LOAD 85.43% MODEL
    # =========================================================

    @classmethod
    def from_dir(
        cls,
        root: str | Path = "outputs",
        device: str | None = None,
        max_length: int = 32,
        mc_passes: int = 0,
        **kwargs,
    ):
        """
        Load the established recognition checkpoint.

        IMPORTANT:
            outputs/long/best

        This is the checkpoint associated with the
        established 85.43% raw recognition result.
        """

        root = Path(root)

        checkpoint = (
            root
            / "long"
            / "best"
        )

        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Recognition checkpoint not found:\n"
                f"{checkpoint}"
            )

        print(
            f"Loading model: {checkpoint}"
        )

        model, processor, tokenizer = load_trained(
            checkpoint
        )

        return cls(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length,
            mc_passes=mc_passes,
        )

    # =========================================================
    # IMAGE CONVERSION
    # =========================================================

    @staticmethod
    def _to_pil(image):

        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, np.ndarray):

            if image.ndim == 2:
                return (
                    Image
                    .fromarray(image)
                    .convert("RGB")
                )

            if image.ndim == 3:
                return (
                    Image
                    .fromarray(image)
                    .convert("RGB")
                )

        raise TypeError(
            f"Unsupported image type: "
            f"{type(image)}"
        )

    # =========================================================
    # ORIGINAL PREPROCESSING
    # =========================================================

    def _prepare_image(self, image):
        """
        Use the same quality/preprocessing path as
        the original OCR pipeline.

        PIL RGB
            ->
        OpenCV BGR
            ->
        preflight
            ->
        prepare
            ->
        RGB PIL
        """

        pil = self._to_pil(image)

        # PIL RGB -> OpenCV BGR
        rgb = np.asarray(pil)

        bgr = rgb[
            :,
            :,
            ::-1
        ].copy()

        # -----------------------------------------------------
        # Quality check
        # -----------------------------------------------------

        quality = preflight(
            bgr,
            check_word_count=False,
        )

        # -----------------------------------------------------
        # If quality fails, preserve the original image.
        # Do not silently destroy the input.
        # -----------------------------------------------------

        if not quality["ok"]:
            return (
                pil,
                quality,
            )

        # -----------------------------------------------------
        # Original crop + lighting normalization
        # -----------------------------------------------------

        processed, prep_report = prepare(
            bgr
        )

        # BGR -> RGB
        processed_rgb = processed[
            :,
            :,
            ::-1
        ]

        processed_image = (
            Image
            .fromarray(processed_rgb)
            .convert("RGB")
        )

        return (
            processed_image,
            quality,
        )

    # =========================================================
    # RAW RECOGNITION
    # =========================================================

    @torch.no_grad()
    def _recognise(
        self,
        images: list,
    ) -> tuple[list, list]:

        if not images:
            return [], []

        processed_images = []
        quality_reports = []

        for image in images:

            processed_image, quality = (
                self._prepare_image(image)
            )

            processed_images.append(
                processed_image
            )

            quality_reports.append(
                quality
            )

        # -----------------------------------------------------
        # Processor
        # -----------------------------------------------------

        pixel_values = self.processor(
            images=processed_images,
            return_tensors="pt",
        ).pixel_values.to(
            self.device
        )

        # -----------------------------------------------------
        # Recognition
        #
        # num_beams=4 is deliberately used here to match
        # the established evaluation/training generation
        # configuration rather than the previous greedy
        # confidence-only path.
        # -----------------------------------------------------

        predictions = (
            predict_with_uncertainty(
                self.model,
                self.tokenizer,
                pixel_values,

                max_length=self.max_length,

                num_beams=4,

                mc_passes=self.mc_passes,

                device=self.device,
            )
        )

        return (
            predictions,
            quality_reports,
        )

    # =========================================================
    # AER
    # =========================================================

    @staticmethod
    def _calculate_aer(
        reference: str,
        hypothesis: str,
    ) -> float:

        from src.script import (
            akshara_error_rate
        )

        return float(
            akshara_error_rate(
                reference,
                hypothesis,
            )
        )

    # =========================================================
    # BUILD RESULT
    # =========================================================

    def _make_result(
        self,
        prediction,
        quality_report=None,
        reference: str | None = None,
    ) -> WordResult:

        text = (
            prediction.text
            or ""
        )

        token_probs = list(
            prediction.token_probs
            or []
        )

        # -----------------------------------------------------
        # Character confidence
        # -----------------------------------------------------

        chars = list(text)

        n = min(
            len(chars),
            len(token_probs),
        )

        char_probs = []

        for i in range(n):

            char_probs.append(
                (
                    chars[i],
                    float(
                        token_probs[i]
                    ),
                )
            )

        # If decoder/tokenizer lengths differ,
        # don't silently delete visible characters.
        if len(chars) > n:

            for i in range(
                n,
                len(chars),
            ):

                char_probs.append(
                    (
                        chars[i],
                        0.0,
                    )
                )

        # -----------------------------------------------------
        # Word confidence
        # -----------------------------------------------------

        if hasattr(
            prediction,
            "mean_prob",
        ):

            confidence = float(
                prediction.mean_prob
            )

        elif token_probs:

            confidence = float(
                sum(token_probs)
                / len(token_probs)
            )

        else:

            confidence = 0.0

        # -----------------------------------------------------
        # Weakest character
        # -----------------------------------------------------

        weakest_position = -1

        if token_probs:

            weakest_position = int(
                min(
                    range(
                        len(token_probs)
                    ),
                    key=lambda i:
                        token_probs[i],
                )
            )

        # -----------------------------------------------------
        # Entropy
        # -----------------------------------------------------

        entropy_per_char = list(
            getattr(
                prediction,
                "token_entropies",
                [],
            )
            or []
        )

        # -----------------------------------------------------
        # AER
        #
        # Only possible when ground truth is supplied.
        # -----------------------------------------------------

        error_rate = None

        if reference is not None:

            error_rate = (
                self._calculate_aer(
                    reference,
                    text,
                )
            )

        # -----------------------------------------------------
        # Warnings
        # -----------------------------------------------------

        warnings = []

        if quality_report:

            warnings.extend(
                quality_report.get(
                    "warnings",
                    [],
                )
                or []
            )

        errors = []

        if quality_report:

            errors.extend(
                quality_report.get(
                    "errors",
                    [],
                )
                or []
            )

        rejected = False

        if quality_report:

            rejected = not bool(
                quality_report.get(
                    "ok",
                    True,
                )
            )

        return WordResult(

            final=text,

            raw=text,

            confidence=confidence,

            char_probs=char_probs,

            entropy_per_char=(
                entropy_per_char
            ),

            flagged=False,

            corrected=False,

            alternatives=[],

            reasons=[],

            rejected=rejected,

            errors=errors,

            warnings=warnings,

            weakest_position=(
                weakest_position
            ),

            error_rate=error_rate,
        )

    # =========================================================
    # MULTIPLE WORDS
    # =========================================================

    def run_words(
        self,
        images: list,
        boxes: list | None = None,
        references: list[str] | None = None,
    ) -> list:

        predictions, quality_reports = (
            self._recognise(images)
        )

        results = []

        for i, prediction in enumerate(
            predictions
        ):

            reference = None

            if references is not None:

                if i < len(references):

                    reference = (
                        references[i]
                    )

            quality = None

            if i < len(
                quality_reports
            ):

                quality = (
                    quality_reports[i]
                )

            result = self._make_result(
                prediction,
                quality_report=quality,
                reference=reference,
            )

            results.append(
                result
            )

        return results

    # =========================================================
    # SINGLE WORD
    # =========================================================

    def run_word(
        self,
        image,
        reference: str | None = None,
    ) -> WordResult:

        results = self.run_words(
            [image],
            references=(
                [reference]
                if reference is not None
                else None
            ),
        )

        if not results:
            return WordResult(
                final="",
                raw="",
                confidence=0.0,
            )

        return results[0]

    # =========================================================
    # IMAGE FILE
    # =========================================================

    def run_word_image(
        self,
        image_path,
        reference: str | None = None,
    ) -> WordResult:

        image = (
            Image
            .open(image_path)
            .convert("RGB")
        )

        return self.run_word(
            image,
            reference=reference,
        )

    # =========================================================
    # PAGE
    # =========================================================

    def run_page(
        self,
        image_path: str | Path,
        do_deskew: bool = True,
    ):

        from src.segmentation.segment import (
            load_image,
            segment_page,
        )

        img = load_image(
            str(image_path)
        )

        seg = segment_page(
            img,
            do_deskew=do_deskew,
        )

        lines = []

        for line in seg["lines"]:

            images = [
                word["image"]
                for word
                in line["words"]
            ]

            boxes = [
                word["box"]
                for word
                in line["words"]
            ]

            words = self.run_words(
                images,
                boxes=boxes,
            )

            lines.append(
                {
                    "words": words,
                    "box": line["box"],
                }
            )

        return lines


# =============================================================
# COMMAND-LINE TEST
# =============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Recognition-only Hindi "
            "handwritten OCR"
        )
    )

    parser.add_argument(
        "image",
        help=(
            "Path to handwritten "
            "Hindi word image"
        ),
    )

    parser.add_argument(
        "--reference",
        default=None,
        help=(
            "Optional ground-truth word "
            "for AER calculation"
        ),
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--mc-passes",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    pipeline = HindiOCRPipeline.from_dir(
        max_length=args.max_length,
        mc_passes=args.mc_passes,
    )

    result = pipeline.run_word_image(
        args.image,
        reference=args.reference,
    )

    print()
    print("=" * 60)
    print("RECOGNITION RESULT")
    print("=" * 60)

    print(
        f"Recognized word: "
        f"{result.final}"
    )

    print(
        f"Word confidence: "
        f"{result.word_confidence * 100:.2f}%"
    )

    print()
    print("Character confidence:")

    for char, confidence in (
        result.char_probs
    ):

        print(
            f"  {char} : "
            f"{confidence * 100:.2f}%"
        )

    if result.error_rate is not None:

        print()

        print(
            "Character/Akshara Error Rate: "
            f"{result.error_rate * 100:.2f}%"
        )

    if result.warnings:

        print()
        print("Warnings:")

        for warning in result.warnings:
            print(
                f"  - {warning}"
            )

    if result.errors:

        print()
        print("Errors:")

        for error in result.errors:
            print(
                f"  - {error}"
            )

    print()