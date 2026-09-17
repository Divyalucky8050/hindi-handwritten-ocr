"""
Turn a raw IIIT-HW-Dev download into clean manifests + a character vocabulary.

IIIT-HW-Dev ships as word-level images plus plain-text annotation files, one
line per sample, in the form:

    <relative/image/path> <label>

The exact directory names and split filenames have varied between releases, so
this script *discovers* the annotation files rather than hard-coding them. Run
it once; everything downstream reads the manifests it writes.

Usage
-----
    python -m src.data.prepare --raw data/raw/iiit-hw-dev --out data/processed

Outputs
-------
    data/processed/train.jsonl      {"image": abs_path, "text": label}
    data/processed/val.jsonl
    data/processed/test.jsonl
    data/processed/charset.json     decoder vocabulary
    data/processed/lexicon.txt      unique words seen in train (for correction)
    data/processed/stats.json       label lengths, charset size, split sizes
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.script import DevanagariCharTokenizer, akshara_count, is_devanagari, normalize

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
SPLIT_HINTS = {
    "train": ("train",),
    "val": ("val", "valid", "dev"),
    "test": ("test", "eval"),
}


def find_annotation_files(raw: Path) -> dict[str, list[Path]]:
    """Group *.txt annotation files under `raw` by the split their name implies."""
    found: dict[str, list[Path]] = {"train": [], "val": [], "test": [], "unknown": []}
    for p in sorted(raw.rglob("*.txt")):
        if p.stat().st_size == 0:
            continue
        name = p.name.lower()
        for split, hints in SPLIT_HINTS.items():
            if any(h in name for h in hints):
                found[split].append(p)
                break
        else:
            found["unknown"].append(p)
    return found


def parse_annotation(path: Path, raw: Path) -> list[dict]:
    """Parse `<image path> <label>` lines, tolerating tabs, multiple spaces and BOM.

    Labels containing spaces are handled by splitting only on the FIRST
    whitespace run - the first field is the path, everything after is the label.
    """
    rows, skipped = [], 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip().lstrip("\ufeff")
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            skipped += 1
            continue
        rel, label = parts[0], normalize(parts[1])
        if not label or not is_devanagari(label):
            skipped += 1
            continue
        img = resolve_image(Path(rel), path.parent, raw)
        if img is None:
            skipped += 1
            continue
        rows.append({"image": str(img), "text": label})
    if skipped:
        print(f"  [{path.name}] kept {len(rows)}, skipped {skipped}")
    return rows


_INDEX: dict[str, Path] | None = None


def resolve_image(rel: Path, ann_dir: Path, raw: Path) -> Path | None:
    """Locate an image referenced by an annotation line.

    Tries the obvious relative locations first, then falls back to a one-time
    basename index of every image under `raw`. The fallback matters because some
    releases store annotations with paths relative to the archive root and
    others relative to the annotation file.
    """
    global _INDEX
    for cand in (ann_dir / rel, raw / rel, raw / rel.name):
        if cand.is_file():
            return cand.resolve()
    if _INDEX is None:
        print("  building image index (one time)...")
        _INDEX = {}
        for p in raw.rglob("*"):
            if p.suffix.lower() in IMAGE_EXTS:
                _INDEX.setdefault(p.name, p.resolve())
        print(f"  indexed {len(_INDEX)} images")
    return _INDEX.get(rel.name)


def make_splits(rows: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    """Fallback 80/10/10 split when the release has no split files.

    Splits by *label* rather than by row so the same word never appears in both
    train and test. Without this you measure memorisation, not recognition.
    """
    by_label: dict[str, list[dict]] = {}
    for r in rows:
        by_label.setdefault(r["text"], []).append(r)
    labels = sorted(by_label)
    random.Random(seed).shuffle(labels)
    n = len(labels)
    cuts = {"train": labels[: int(0.8 * n)],
            "val": labels[int(0.8 * n): int(0.9 * n)],
            "test": labels[int(0.9 * n):]}
    return {k: [r for lab in v for r in by_label[lab]] for k, v in cuts.items()}


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", default=Path("data/processed"), type=Path)
    ap.add_argument("--min-char-freq", type=int, default=2,
                    help="drop characters rarer than this from the decoder vocab")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = args.raw.resolve()
    if not raw.is_dir():
        sys.exit(f"error: {raw} is not a directory")

    print(f"scanning {raw}")
    ann = find_annotation_files(raw)
    for k, v in ann.items():
        if v:
            print(f"  {k}: {[p.name for p in v]}")

    splits: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        rows = [r for p in ann[split] for r in parse_annotation(p, raw)]
        if rows:
            splits[split] = rows

    if not splits:
        print("no split-named annotations found; pooling all .txt files and "
              "splitting 80/10/10 by label")
        pooled = [r for p in ann["unknown"] for r in parse_annotation(p, raw)]
        if not pooled:
            sys.exit("error: no usable annotation lines found. Check --raw path.")
        splits = make_splits(pooled, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        write_jsonl(rows, args.out / f"{split}.jsonl")
        print(f"wrote {len(rows):>7} -> {split}.jsonl")

    # Decoder vocabulary is built from TRAIN ONLY. Building it from all splits
    # leaks test information and hides genuine unseen-character failures.
    train_texts = [r["text"] for r in splits.get("train", [])]
    tok = DevanagariCharTokenizer.from_texts(train_texts, min_freq=args.min_char_freq)
    tok.save(args.out / "charset.json")
    print(f"wrote charset.json ({len(tok)} tokens incl. 4 special)")

    lexicon = sorted({r["text"] for r in splits.get("train", [])})
    (args.out / "lexicon.txt").write_text("\n".join(lexicon), encoding="utf-8")
    print(f"wrote lexicon.txt ({len(lexicon)} unique words)")

    lengths = [akshara_count(t) for t in train_texts]
    unseen = Counter()
    for split in ("val", "test"):
        for r in splits.get(split, []):
            for c in r["text"]:
                if c not in tok.stoi:
                    unseen[c] += 1
    stats = {
        "splits": {k: len(v) for k, v in splits.items()},
        "unique_train_labels": len(lexicon),
        "charset_size": len(tok),
        "akshara_len": {
            "min": min(lengths, default=0),
            "max": max(lengths, default=0),
            "mean": round(sum(lengths) / max(len(lengths), 1), 2),
            "p99": sorted(lengths)[int(0.99 * len(lengths))] if lengths else 0,
        },
        "chars_in_eval_not_in_train": dict(unseen.most_common(20)),
    }
    (args.out / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps(stats, ensure_ascii=False, indent=2))
    print("\nSet max_target_length in config.yaml to at least "
          f"{stats['akshara_len']['p99'] * 2 + 4} (p99 aksharas x ~2 codepoints + specials)")


if __name__ == "__main__":
    main()