"""
Full test-set evaluation of the from-scratch model trained on
characters + conjuncts + printed words + handwritten words.

Recognition first, then the lexicon corrector on low-confidence words - the same
protocol used for the pretrained baseline, so the numbers are comparable.
"""
import json, numpy as np, torch, time
from pathlib import Path
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from src.script import DevanagariCharTokenizer, to_aksharas, normalize
from src.correction.lexicon import Lexicon
from src.correction.lm import NgramLM
from src.correction.corrector import Corrector
from src.evaluate.metrics import recognition_metrics, correction_metrics, akshara_confusion
from src.evaluate.calibration import expected_calibration_error, risk_coverage_curve

CKPT, N = "outputs/long/best", 12869
t0 = time.time()

tok = DevanagariCharTokenizer.load(f"{CKPT}/charset.json")
proc = TrOCRProcessor.from_pretrained(CKPT)
model = VisionEncoderDecoderModel.from_pretrained(CKPT).cuda().eval()
print(f"charset {len(tok)}")

# Same combined lexicon used for the baseline: dataset vocab + train labels.
words = {}
for t in Path("data/raw/IIIT-HW-Hindi_v1/hindi_vocab.txt").read_text(
        encoding="utf-8", errors="replace").split():
    w = normalize(t)
    if w and any(0x0900 <= ord(c) <= 0x097F for c in w):
        words[w] = words.get(w, 0) + 1
for r in map(json.loads, open("data/processed/train.jsonl", encoding="utf-8")):
    w = normalize(r["text"]); words[w] = words.get(w, 0) + 1
lex = Lexicon(words)
ak = NgramLM(order=4, level="akshara").fit(set(lex.freq))
corr = Corrector(lex, akshara_lm=ak, accept_margin=0.2, max_distance=2,
                 max_candidates=12)
print(f"lexicon {len(lex)}")

rows = [json.loads(l) for l in open("data/processed/test.jsonl", encoding="utf-8")][:N]
refs = [r["text"].strip() for r in rows]

preds, confs = [], []
for i in range(0, len(rows), 48):
    b = rows[i:i+48]
    pv = proc(images=[Image.open(r["image"]).convert("RGB") for r in b],
              return_tensors="pt").pixel_values.cuda()
    with torch.no_grad():
        o = model.generate(pv, max_length=24, num_beams=4, length_penalty=0.7,
                           no_repeat_ngram_size=0, output_scores=True,
                           return_dict_in_generate=True)
    preds += [p.strip() for p in tok.batch_decode(o.sequences.cpu().tolist())]
    confs += (torch.exp(o.sequences_scores).clamp(0, 1).cpu().tolist()
              if getattr(o, "sequences_scores", None) is not None
              else [0.5] * len(b))
    print(f"  {min(i+48, len(rows))}/{len(rows)}", end="\r")
print()

thr = float(np.percentile(confs, 80))
final = [corr.correct_word(p).suggestion if c < thr else p
         for p, c in zip(preds, confs)]

ok0 = [to_aksharas(a) == to_aksharas(b) for a, b in zip(refs, preds)]
rec, rec2 = recognition_metrics(refs, preds), recognition_metrics(refs, final)
cm = correction_metrics(refs, preds, final)
cov, risk = risk_coverage_curve(confs, ok0)
rat = lambda x: round(float(risk[int(np.argmin(np.abs(cov - x)))]), 4)

print("=" * 62)
print(f"FROM-SCRATCH MODEL on {len(rows)} test images")
print(f"  recognition word accuracy : {rec['word_accuracy']*100:.2f}%")
print(f"  FINAL word accuracy       : {rec2['word_accuracy']*100:.2f}%")
print(f"  AER  {rec['aer_corpus']:.4f} -> {rec2['aer_corpus']:.4f}")
print(f"  fixed {cm['fixed']}  damaged {cm['damaged']}  net {cm['net_change']*100:+.2f} pp")
print(f"  ECE {expected_calibration_error(confs, ok0):.4f}")
print(f"  risk @70% coverage {rat(0.7)}   @100% {rat(1.0)}")
print("\n  BASELINE (pretrained + correction): 78.40% -> 82.87%")
print("\ntop confusions:")
for c in akshara_confusion(refs, final, top=6):
    print(f"  {c['expected']} -> {c['read_as']} ({c['count']}x)")
print("\nexamples:")
for r, p, f in list(zip(refs, preds, final))[:10]:
    print(f"  {'OK ' if r == f else '   '} truth={r!r:16} pred={p!r:16} final={f!r}")
print("=" * 62)
print(f"elapsed {time.time()-t0:.0f}s")

Path("outputs/reports").mkdir(parents=True, exist_ok=True)
json.dump({"model": "long_run_332k", "n": len(rows),
           "recognition": rec, "final": rec2, "correction": cm},
          open("outputs/reports/long_report.json", "w"),
          ensure_ascii=False, indent=2)
print("saved -> outputs/reports/long_report.json")
