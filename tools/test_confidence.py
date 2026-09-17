from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import HindiOCRPipeline


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print(
            r'python tools\test_confidence.py "IMAGE_PATH"'
        )
        raise SystemExit(1)

    image_path = sys.argv[1]

    pipeline = HindiOCRPipeline.from_dir()

    result = pipeline.run_word_image(
        image_path
    )

    print()
    print("=" * 60)
    print("CONFIDENCE TEST")
    print("=" * 60)

    print(f"Image: {image_path}")
    print(f"Recognized word: {result.final}")
    print(
        f"Word confidence: "
        f"{result.confidence * 100:.2f}%"
    )

    print()
    print("Character confidence:")

    for char, confidence in result.char_probs:
        print(
            f"  {char} : "
            f"{confidence * 100:.2f}%"
        )

    if result.char_probs:
        weakest = min(
            result.char_probs,
            key=lambda item: item[1],
        )

        print()
        print(
            "Lowest-confidence character: "
            f"{weakest[0]} "
            f"({weakest[1] * 100:.2f}%)"
        )


if __name__ == "__main__":
    main()