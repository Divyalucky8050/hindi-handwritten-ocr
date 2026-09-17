"""Exercise the lexicon, LMs and corrector on a small hand-written Hindi corpus."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.correction.corrector import Corrector, confusability, visual_similarity
from src.correction.lexicon import Lexicon
from src.correction.lm import NgramLM
from src.script import to_aksharas

CORPUS = [
    "राम स्कूल जाता है",
    "सीता किताब पढ़ती है",
    "बच्चे स्कूल में पढ़ते हैं",
    "मोहन बाजार जाता है",
    "वह अपनी किताब पढ़ता है",
    "मेरा नाम राम है",
    "यह मेरा घर है",
    "हम सब मिलकर काम करते हैं",
    "पानी बहुत ठंडा है",
    "वह बाजार से फल लाता है",
    "किताब मेज पर रखी है",
    "बच्चा दूध पीता है",
    "सूरज पूरब से निकलता है",
    "गाय घास खाती है",
    "मैं हिन्दी पढ़ता हूँ",
    "वह स्कूल में पढ़ाता है",
    "राम और मोहन दोस्त हैं",
    "यह किताब बहुत अच्छी है",
    "घर के बाहर पेड़ है",
    "वह पानी पीता है",
]


def main() -> int:
    ok = True

    # ---------------- lexicon ----------------
    lex = Lexicon.from_corpus("\n".join(CORPUS))
    print(f"lexicon: {len(lex)} unique words")

    assert lex.contains("किताब"), "lexicon lookup failed"
    assert not lex.contains("कीताब"), "lexicon accepted a non-word"

    near = lex.nearest("कीताब", max_distance=2, limit=5)
    print(f"nearest('कीताब') = {near}")
    if not near or near[0][0] != "किताब":
        print("FAIL: BK-tree did not recover किताब")
        ok = False

    # BK-tree must agree with brute force, or the pruning is wrong.
    from src.script import edit_distance
    q = to_aksharas("पनी")
    brute = sorted(
        [(w, edit_distance(to_aksharas(w), q)) for w in lex.freq
         if edit_distance(to_aksharas(w), q) <= 2],
        key=lambda x: (x[1], -lex.freq[x[0]]),
    )
    tree = lex.nearest("पनी", max_distance=2, limit=100)
    if {w for w, _ in brute} != {w for w, _ in tree}:
        print(f"FAIL: BK-tree disagrees with brute force\n  brute={brute}\n  tree={tree}")
        ok = False
    else:
        print(f"BK-tree == brute force on 'पनी' ({len(tree)} matches)")

    # ---------------- language models ----------------
    akshara_lm = NgramLM(order=4, level="akshara").fit(set(lex.freq))
    word_lm = NgramLM(order=3, level="word").fit(CORPUS)
    print(f"akshara LM vocab {len(akshara_lm.vocab)}, word LM vocab {len(word_lm.vocab)}")

    real = akshara_lm.word_logprob("किताब")
    junk = akshara_lm.word_logprob("ािकबत")
    print(f"akshara LM: real={real:.2f}  junk={junk:.2f}")
    if not real > junk:
        print("FAIL: akshara LM ranks junk above a real word")
        ok = False

    good = word_lm.sentence_logprob(["राम", "स्कूल", "जाता", "है"])
    bad = word_lm.sentence_logprob(["है", "जाता", "राम", "स्कूल"])
    print(f"word LM: good={good:.2f}  scrambled={bad:.2f}")
    if not good > bad:
        print("FAIL: word LM does not prefer correct word order")
        ok = False

    # ---------------- visual confusability ----------------
    print(f"\nvisual_similarity('ब','व') = {visual_similarity('ब', 'व')}")
    print(f"visual_similarity('क','घ') = {visual_similarity('क', 'घ')}")
    if not visual_similarity("ब", "व") > visual_similarity("क", "घ"):
        print("FAIL: confusable pair not ranked above unrelated pair")
        ok = False

    # ---------------- corrector ----------------
    corr = Corrector(lex, word_lm=word_lm, akshara_lm=akshara_lm)

    # Case 1: matra corruption inside a sentence -> should be fixed.
    c = corr.correct_word("कीताब", left=["सीता"], right=["पढ़ती", "है"])
    print(f"\ncorrect('कीताब' | सीता _ पढ़ती है) -> {c.suggestion!r} "
          f"(accepted={c.accepted}, d={c.distance})")
    if c.suggestion != "किताब":
        print("FAIL: did not correct कीताब -> किताब")
        ok = False

    # Case 2: a word that is already correct must be left alone.
    c2 = corr.correct_word("किताब", left=["सीता"], right=["पढ़ती", "है"])
    print(f"correct('किताब') -> {c2.suggestion!r} (accepted={c2.accepted})")
    if c2.accepted or c2.suggestion != "किताब":
        print("FAIL: corrector altered a correct word")
        ok = False

    # Case 3: sentence-level, only flagged words touched.
    words = ["सीता", "कीताब", "पढ़ती", "है"]
    flags = [False, True, False, False]
    results = corr.correct_sentence(words, flags)
    out = [r.suggestion for r in results]
    print(f"sentence {words} -> {out}")
    if out != ["सीता", "किताब", "पढ़ती", "है"]:
        print("FAIL: sentence correction wrong")
        ok = False

    # ---------------- BK-tree speed ----------------
    big = Lexicon({f"{w}{i}": 1 for w in lex.freq for i in range(40)})
    t0 = time.perf_counter()
    for _ in range(50):
        big.nearest("कीताब", max_distance=2, limit=5)
    dt = (time.perf_counter() - t0) / 50 * 1000
    print(f"\nBK-tree query on {len(big)} words: {dt:.2f} ms/query")

    print("\nPASS" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())