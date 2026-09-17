from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import HindiOCRPipeline
from src.script import akshara_error_rate


TEST_FILE = ROOT / "data" / "processed" / "test.jsonl"


def load_test_records():
    records = []

    with TEST_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


def main():

    print("=" * 70)
    print("HINDI OCR AER EVALUATION")
    print("=" * 70)

    print(f"Test file: {TEST_FILE}")

    records = load_test_records()

    print(
        f"Test samples: {len(records)}"
    )

    pipeline = HindiOCRPipeline.from_dir()

    total_aer = 0.0
    total_words = 0
    exact_matches = 0

    print()
    print("Starting evaluation...")
    print()

    for index, record in enumerate(
        records,
        start=1,
    ):

        image_path = record["image"]
        reference = record["text"]

        result = pipeline.run_word_image(
            image_path
        )

        prediction = result.final

        aer = akshara_error_rate(
            reference,
            prediction,
        )

        total_aer += aer
        total_words += 1

        if prediction == reference:
            exact_matches += 1

        if index <= 10:
            print(
                f"[{index}/{len(records)}]"
            )
            print(
                f"Truth : {reference}"
            )
            print(
                f"Pred  : {prediction}"
            )
            print(
                f"AER   : {aer * 100:.2f}%"
            )
            print()

        if index % 500 == 0:
            current_aer = (
                total_aer
                / total_words
            )

            current_accuracy = (
                exact_matches
                / total_words
            )

            print(
                f"Progress: "
                f"{index}/{len(records)} | "
                f"Accuracy: "
                f"{current_accuracy * 100:.2f}% | "
                f"AER: "
                f"{current_aer * 100:.2f}%"
            )

    if total_words == 0:
        print("No test samples found.")
        return

    average_aer = (
        total_aer
        / total_words
    )

    accuracy = (
        exact_matches
        / total_words
    )

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"Test samples       : {total_words}"
    )

    print(
        f"Exact word matches : {exact_matches}"
    )

    print(
        f"Word accuracy      : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Average AER        : "
        f"{average_aer * 100:.2f}%"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()