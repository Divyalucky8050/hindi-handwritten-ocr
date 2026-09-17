"""
Devanagari script utilities.

Why this module exists
----------------------
Devanagari is an abugida, not an alphabet. A single visual unit that a human
reader perceives as "one letter" (an *akshara*) is often several Unicode
codepoints:

    क्ष  =  क + ् + ष        (3 codepoints, 1 akshara)
    कि   =  क + ि           (2 codepoints, 1 akshara; the matra is drawn BEFORE
                              the consonant even though it is stored after it)
    र्क   =  र + ् + क        (3 codepoints, 1 akshara, reph drawn above)

Consequences for OCR that most projects get wrong:

1. Character Error Rate computed over raw codepoints is not comparable to CER
   for Latin script and systematically flatters the model, because getting the
   consonant right while dropping the matra counts as 1/2 correct rather than a
   wrong akshara.
2. Edit distance for spelling correction over codepoints produces nonsense
   candidates, because it will happily insert a matra with no base consonant.
3. A "character-level" decoder vocabulary should be built over codepoints (that
   is what the model must emit), but every *evaluation* and *correction* step
   should operate over aksharas.

So: tokenize over codepoints, measure and correct over aksharas.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

# --------------------------------------------------------------------------
# Unicode ranges (Devanagari block U+0900-U+097F)
# --------------------------------------------------------------------------
CONSONANT = r"[\u0915-\u0939\u0958-\u095F\u0978-\u097F]"
VOWEL = r"[\u0904-\u0914\u0960\u0961\u0972-\u0977]"
MATRA = r"[\u093A\u093B\u093E-\u094C\u094E\u094F\u0955-\u0957\u0962\u0963]"
HALANT = "\u094D"          # virama, forms conjuncts
NUKTA = "\u093C"
SIGNS = r"[\u0900-\u0903]"  # chandrabindu, anusvara, visarga
DIGITS = r"[\u0966-\u096F]"
PUNCT = r"[\u0964\u0965\u0970\u0971]"  # danda, double danda
AVAGRAHA = "\u093D"

# An akshara: zero or more (consonant + optional nukta + halant) conjunct
# prefixes, then a base (consonant or independent vowel), then any matras,
# then any signs, then an optional trailing halant (word-final half form).
AKSHARA_PATTERN = (
    f"(?:{CONSONANT}{NUKTA}?{HALANT})*"
    f"(?:{CONSONANT}{NUKTA}?|{VOWEL}|{AVAGRAHA})"
    f"{MATRA}*{NUKTA}?{SIGNS}*{HALANT}?"
)

_AKSHARA_RE = re.compile(f"{AKSHARA_PATTERN}|{DIGITS}|{PUNCT}|.", re.UNICODE)

DEVANAGARI_RANGE = (0x0900, 0x097F)


def normalize(text: str) -> str:
    """Canonical form for Devanagari text.

    Applies NFC, then folds the precomposed nukta consonants onto their
    decomposed base+nukta form. IIIT-HW-Dev annotations are inconsistent about
    these, and if you skip this step the same word appears as two distinct
    labels and your vocabulary silently grows.
    """
    text = unicodedata.normalize("NFC", text)
    folds = {
        "\u0929": "\u0928\u093C",  # nnna
        "\u0931": "\u0930\u093C",  # rra
        "\u0934": "\u0933\u093C",  # llla
        "\u0958": "\u0915\u093C",  # qa
        "\u0959": "\u0916\u093C",  # khha
        "\u095A": "\u0917\u093C",  # ghha
        "\u095B": "\u091C\u093C",  # za
        "\u095C": "\u0921\u093C",  # dddha
        "\u095D": "\u0922\u093C",  # rha
        "\u095E": "\u092B\u093C",  # fa
        "\u095F": "\u092F\u093C",  # yya
    }
    for pre, dec in folds.items():
        text = text.replace(pre, dec)
    return text.strip()


def to_aksharas(text: str) -> list[str]:
    """Split text into akshara (grapheme cluster) units."""
    text = normalize(text)
    return [m.group(0) for m in _AKSHARA_RE.finditer(text) if m.group(0)]


def akshara_count(text: str) -> int:
    return len(to_aksharas(text))


def is_devanagari(text: str, threshold: float = 0.5) -> bool:
    """True if at least `threshold` of the non-space characters are Devanagari."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    lo, hi = DEVANAGARI_RANGE
    hits = sum(1 for c in chars if lo <= ord(c) <= hi)
    return hits / len(chars) >= threshold


def edit_distance(a: list[str] | str, b: list[str] | str) -> int:
    """Levenshtein distance over arbitrary token sequences.

    Pass lists of aksharas (not raw strings) when comparing Devanagari.
    """
    a = list(a)
    b = list(b)
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def akshara_error_rate(reference: str, hypothesis: str) -> float:
    """Akshara Error Rate - the correct analogue of CER for Devanagari."""
    ref = to_aksharas(reference)
    if not ref:
        return 0.0 if not to_aksharas(hypothesis) else 1.0
    return edit_distance(ref, to_aksharas(hypothesis)) / len(ref)


# --------------------------------------------------------------------------
# Character-level tokenizer for the decoder
# --------------------------------------------------------------------------
class DevanagariCharTokenizer:
    """Codepoint-level tokenizer with a vocabulary built from the corpus.

    We use a character vocabulary rather than the pretrained TrOCR (RoBERTa BPE)
    vocabulary because that vocabulary has essentially no Devanagari coverage -
    Hindi text falls back to byte-level fragments, which wastes decoder capacity
    and lengthens sequences by 3-4x. IIIT-HW-Dev has a closed character
    inventory of roughly 100 symbols, so a character vocabulary is both smaller
    and a better fit.

    Special tokens follow the TrOCR/BART convention so the decoder config can be
    reused: <s> is both BOS and the decoder_start token, </s> is EOS.
    """

    PAD, BOS, EOS, UNK = "<pad>", "<s>", "</s>", "<unk>"
    SPECIALS = [PAD, BOS, EOS, UNK]

    def __init__(self, vocab: list[str]):
        for s in reversed(self.SPECIALS):
            if s in vocab:
                vocab.remove(s)
        self.itos = self.SPECIALS + vocab
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_token_id = self.stoi[self.PAD]
        self.bos_token_id = self.stoi[self.BOS]
        self.eos_token_id = self.stoi[self.EOS]
        self.unk_token_id = self.stoi[self.UNK]

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def from_texts(cls, texts, min_freq: int = 1) -> "DevanagariCharTokenizer":
        from collections import Counter

        counts = Counter()
        for t in texts:
            counts.update(normalize(t))
        vocab = sorted(c for c, n in counts.items() if n >= min_freq)
        return cls(vocab)

    def encode(self, text: str, max_length: int = 64, add_special: bool = True) -> list[int]:
        ids = [self.stoi.get(c, self.unk_token_id) for c in normalize(text)]
        if add_special:
            ids = [self.bos_token_id] + ids[: max_length - 2] + [self.eos_token_id]
        return ids[:max_length]

    def decode(self, ids, skip_special: bool = True,
               skip_special_tokens: bool | None = None) -> str:
        # Accept the HuggingFace kwarg name too: call sites are shared between
        # this tokenizer and HF tokenizers, and mismatching the name raises
        # TypeError at inference time only.
        if skip_special_tokens is not None:
            skip_special = skip_special
        out = []
        for i in ids:
            i = int(i)
            if i < 0 or i >= len(self.itos):
                continue
            tok = self.itos[i]
            if tok in self.SPECIALS:
                if tok == self.EOS:
                    break
                if skip_special:
                    continue
            out.append(tok)
        return normalize("".join(out))

    def batch_decode(self, batch, skip_special: bool = True,
                     skip_special_tokens: bool | None = None) -> list[str]:
        if skip_special_tokens is not None:
            skip_special = skip_special_tokens
        return [self.decode(row, skip_special) for row in batch]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"itos": self.itos}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "DevanagariCharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        itos = data["itos"]
        return cls([t for t in itos if t not in cls.SPECIALS])


if __name__ == "__main__":
    samples = ["क्षमा", "हिन्दी", "विद्यार्थी", "श्रीमान्", "क़िताब"]
    for s in samples:
        print(f"{s:<12} -> {to_aksharas(s)}  ({akshara_count(s)} aksharas, {len(s)} codepoints)")
    tok = DevanagariCharTokenizer.from_texts(samples)
    print("\nvocab size:", len(tok))
    ids = tok.encode("हिन्दी")
    print("encode/decode roundtrip:", ids, "->", repr(tok.decode(ids)))
    print("AER('हिन्दी','हिंदी') =", round(akshara_error_rate("हिन्दी", "हिंदी"), 3))