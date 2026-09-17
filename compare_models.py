"""Test every available Devanagari OCR checkpoint on the same 300 test images."""
import json, torch, traceback
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, AutoTokenizer
from src.script import to_aksharas, akshara_error_rate

CANDIDATES = [
    "aayushpuri01/TrOCR-Devanagari",
    "syubraj/TrOCR_Nepali",
    "paudelanil/trocr-devanagari-2",
]

rows = [json.loads(l) for l in open("data/processed/test.jsonl", encoding="utf-8")][:300]
refs = [r["text"].strip() for r in rows]
imgs = [Image.open(r["image"]).convert("RGB") for r in rows]

for name in CANDIDATES:
    print(f"\n{'='*60}\n{name}")
    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = VisionEncoderDecoderModel.from_pretrained(name).cuda().eval()
        try:
            proc = TrOCRProcessor.from_pretrained(name)
        except Exception:
            from transformers import ViTImageProcessor
            proc = TrOCRProcessor(
                image_processor=ViTImageProcessor.from_pretrained(
                    "google/vit-base-patch16-224-in21k"),
                tokenizer=tok)

        preds = []
        for i in range(0, len(rows), 32):
            pv = proc(images=imgs[i:i+32], return_tensors="pt").pixel_values.cuda()
            with torch.no_grad():
                o = model.generate(pv, max_length=32, num_beams=4,
                                   no_repeat_ngram_size=0)
            preds += [p.strip() for p in tok.batch_decode(o, skip_special_tokens=True)]

        exact = sum(to_aksharas(r) == to_aksharas(p) for r, p in zip(refs, preds))
        aer = sum(akshara_error_rate(r, p) for r, p in zip(refs, preds)) / len(refs)
        print(f"  WORD ACCURACY : {100*exact/len(refs):.2f}%   AER: {aer:.4f}")
        for r, p in list(zip(refs, preds))[:5]:
            print(f"    truth={r!r:18} pred={p!r}")
        del model; torch.cuda.empty_cache()
    except Exception:
        print("  FAILED:")
        traceback.print_exc(limit=2)
