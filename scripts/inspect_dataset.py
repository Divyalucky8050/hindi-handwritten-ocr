"""
Inspect an extracted IIIT-HW-Dev download and report its real structure.

Run this BEFORE src.data.prepare. It tells you exactly what you got, so you can
point --raw at the right directory instead of guessing.

    python scripts/inspect_dataset.py data/raw/IIIT-HW-Dev_v1

It prints:
  * the directory tree, two levels deep
  * every .txt file, with its first three lines shown verbatim
  * whether the first field of each line resolves to an image that exists
  * a decode of one label so you can confirm the encoding is really UTF-8
  * the image count and a sample image's dimensions
"""

from __future__ import annotations

import sys
import unicodedata
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def tree(root: Path, max_depth: int = 2) -> None:
    print(f"\n=== directory tree (depth {max_depth}) ===")
    for p in sorted(root.rglob("*")):
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        depth = len(rel.parts)
        if depth > max_depth:
            continue
        if p.is_dir():
            n_img = sum(1 for c in p.iterdir()
                        if c.is_file() and c.suffix.lower() in IMAGE_EXTS)
            n_sub = sum(1 for c in p.iterdir() if c.is_dir())
            extra = []
            if n_img:
                extra.append(f"{n_img} images")
            if n_sub:
                extra.append(f"{n_sub} subdirs")
            print(f"{'  ' * (depth - 1)}{rel.name}/  "
                  f"{'(' + ', '.join(extra) + ')' if extra else ''}")
        elif depth <= max_depth and p.suffix.lower() not in IMAGE_EXTS:
            print(f"{'  ' * (depth - 1)}{rel.name}  ({p.stat().st_size:,} bytes)")


def show_readme(root: Path) -> None:
    for p in root.rglob("*"):
        if p.is_file() and p.name.lower().startswith("readme"):
            print(f"\n=== {p.relative_to(root)} ===")
            txt = p.read_text(encoding="utf-8", errors="replace")
            print(txt[:2000])
            if len(txt) > 2000:
                print(f"... ({len(txt)} chars total)")
            return
    print("\n(no Readme found)")


def find_images(root: Path) -> dict[str, Path]:
    print("\n=== indexing images (this takes a minute on 95k files) ===")
    index: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS:
            index.setdefault(p.name, p)
    print(f"found {len(index):,} uniquely-named images")
    return index


def inspect_annotations(root: Path, index: dict[str, Path]) -> None:
    txts = [p for p in sorted(root.rglob("*.txt"))
            if p.stat().st_size > 0 and not p.name.lower().startswith("readme")]
    if not txts:
        print("\n!! no .txt annotation files found - check you extracted the right archive")
        return

    print(f"\n=== {len(txts)} annotation file(s) ===")
    for p in txts:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        lines = [l for l in lines if l.strip()]
        print(f"\n--- {p.relative_to(root)}  ({len(lines):,} lines) ---")
        for l in lines[:3]:
            print(f"    {l!r}")

        field_counts = Counter(len(l.split()) for l in lines[:500])
        print(f"    fields per line (first 500): {dict(field_counts)}")

        sample = lines[0].split(maxsplit=1)
        if len(sample) < 2:
            print("    !! lines do not split into path + label")
            continue

        rel, label = sample[0], sample[1]
        print(f"    field 0 : {rel!r}")
        print(f"    field 1 : {label!r}")

        for desc, cand in (("as given from dataset root", root / rel),
                           ("relative to the txt file", p.parent / rel),
                           ("by basename lookup", index.get(Path(rel).name))):
            if cand is not None and Path(cand).is_file():
                print(f"    RESOLVES {desc}: {cand}")
                break
        else:
            print("    !! field 0 does not resolve to an existing image")

        sampled = lines[:200]
        hits = 0
        for l in sampled:
            parts = l.split(maxsplit=1)
            if not parts:
                continue
            name = Path(parts[0]).name
            if (root / parts[0]).is_file() or name in index:
                hits += 1
        print(f"    {hits}/{len(sampled)} sampled paths resolve")

        codepoints = [f"U+{ord(c):04X} {unicodedata.name(c, '?')}" for c in label[:6]]
        print(f"    label codepoints: {codepoints}")
        devan = sum(1 for c in label if 0x0900 <= ord(c) <= 0x097F)
        verdict = "  (looks correct)" if devan else \
            "  !! NOT Devanagari - this file may be a word-ID list, not text labels"
        print(f"    Devanagari chars in label: {devan}/{len(label)}{verdict}")

        uniq = len({l.split(maxsplit=1)[1] for l in lines
                    if len(l.split(maxsplit=1)) == 2})
        print(f"    unique labels: {uniq:,}")


def sample_image(index: dict[str, Path]) -> None:
    if not index:
        return
    p = next(iter(index.values()))
    print(f"\n=== sample image: {p.name} ===")
    try:
        from PIL import Image
        with Image.open(p) as im:
            print(f"    size {im.size}, mode {im.mode}, format {im.format}")
    except Exception as e:  # noqa: BLE001
        print(f"    could not open ({e}). Is Pillow installed?")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    root = Path(sys.argv[1]).expanduser().resolve()
    if not root.is_dir():
        print(f"error: {root} is not a directory")
        return 1

    print(f"inspecting {root}")
    tree(root)
    show_readme(root)
    index = find_images(root)
    inspect_annotations(root, index)
    sample_image(index)

    print("\n" + "=" * 60)
    print("If paths resolve and labels are Devanagari, run:")
    print(f"  python -m src.data.prepare --raw {root} --out data/processed")
    print("If nothing resolved, point --raw at the folder that directly")
    print("contains the annotation .txt files and the image directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())