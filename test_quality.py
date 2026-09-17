"""Check the pre-flight thresholds against real images before wiring them in."""
import json, glob, cv2
from src.segmentation.quality import preflight

print("=== DATASET images: all should PASS ===")
rows = [json.loads(l) for l in open("data/processed/test.jsonl", encoding="utf-8")][:60]
fails = 0
for r in rows:
    res = preflight(cv2.imread(r["image"]))
    if not res["ok"]:
        fails += 1
        if fails <= 5:
            print(f"  FAIL {r['text']}: {res['errors'][0][:70]}")
print(f"  {60 - fails}/60 passed")

print("\n=== IMAGES IN C:\\hindi-ocr ===")
for p in sorted(glob.glob("*.jpg") + glob.glob("*.jpeg")):
    img = cv2.imread(p)
    if img is None:
        continue
    res = preflight(img)
    m = res["metrics"]
    tag = "PASS" if res["ok"] else "STOP"
    print(f"  {tag} {p:24} aspect {m['aspect']:5.2f} sharp {m['sharpness']:7.1f} "
          f"contrast {m['contrast']:5.0f} margin {m['margin']:.2f}")
    for e in res["errors"]:
        print(f"       -> {e[:78]}")
    for w in res["warnings"]:
        print(f"       ~  {w[:78]}")