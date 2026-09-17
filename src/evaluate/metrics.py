"""
Evaluation for the whole system.

Report these four blocks in your results section - they map one-to-one onto the
four contributions claimed in the abstract, which is what a reviewer checks.

1. Recognition   AER, WER, exact-match word accuracy
2. Calibration   ECE, Brier score, risk-coverage curve
3. Detection     precision / recall / PR-AUC vs a confidence-threshold baseline
4. Correction    net accuracy change, fixes vs damages

Block 4 is the one most projects omit and the one that decides whether the system
is actually useful. Correction that fixes 40 words and breaks 45 is a regression,
and only a fix/damage breakdown reveals that. Reporting post-correction accuracy
alone hides it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.script import akshara_error_rate, to_aksharas


# ---------------------------------------------------------------- recognition
def recognition_metrics(refs: list[str], hyps: list[str]) -> dict:
    if not refs:
        return {}
    aers = [akshara_error_rate(r, h) for r, h in zip(refs, hyps)]
    exact = [to_aksharas(r) == to_aksharas(h) for r, h in zip(refs, hyps)]
    from src.script import edit_distance
    tot_d = sum(edit_distance(to_aksharas(r), to_aksharas(h)) for r, h in zip(refs, hyps))
    tot_n = sum(len(to_aksharas(r)) for r in refs)
    return {
        "n": len(refs),
        "aer_corpus": round(tot_d / max(tot_n, 1), 4),
        "aer_mean_per_word": round(float(np.mean(aers)), 4),
        "word_accuracy": round(float(np.mean(exact)), 4),
        "wer": round(1.0 - float(np.mean(exact)), 4),
    }


# ---------------------------------------------------------------- calibration
def calibration_metrics(confidences, correct, n_bins: int = 10) -> dict:
    from src.evaluate.calibration import expected_calibration_error, risk_coverage_curve

    conf = np.asarray(confidences, float)
    corr = np.asarray(correct, float)
    if conf.size == 0:
        return {}
    ece = expected_calibration_error(conf, corr, n_bins)
    brier = float(np.mean((conf - corr) ** 2))
    cov, risk = risk_coverage_curve(conf, corr)

    edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        bins.append({"range": [round(lo, 2), round(hi, 2)], "n": int(m.sum()),
                     "mean_confidence": round(float(conf[m].mean()), 4),
                     "actual_accuracy": round(float(corr[m].mean()), 4)})

    def risk_at(c: float) -> float:
        i = int(np.argmin(np.abs(cov - c)))
        return round(float(risk[i]), 4)

    return {
        "ece": round(ece, 4),
        "brier": round(brier, 4),
        "mean_confidence": round(float(conf.mean()), 4),
        "actual_accuracy": round(float(corr.mean()), 4),
        "overconfidence_gap": round(float(conf.mean() - corr.mean()), 4),
        "reliability_bins": bins,
        "risk_at_coverage": {"50%": risk_at(0.5), "70%": risk_at(0.7),
                             "90%": risk_at(0.9), "100%": risk_at(1.0)},
    }


# ---------------------------------------------------------------- detection
def detection_metrics(y_true, y_score, threshold: float, beta: float = 2.0) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y_true, int)
    s = np.asarray(y_score, float)
    pred = (s >= threshold).astype(int)
    tp = int((pred & y).sum())
    fp = int((pred & (1 - y)).sum())
    fn = int(((1 - pred) & y).sum())
    tn = int(((1 - pred) & (1 - y)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    b2 = beta ** 2
    fb = (1 + b2) * prec * rec / max(b2 * prec + rec, 1e-12)
    out = {
        "threshold": round(threshold, 4),
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(2 * prec * rec / max(prec + rec, 1e-12), 4),
        f"f{beta:g}": round(fb, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }
    if 0 < y.sum() < len(y):
        out["roc_auc"] = round(float(roc_auc_score(y, s)), 4)
        out["pr_auc"] = round(float(average_precision_score(y, s)), 4)
    return out


# ---------------------------------------------------------------- correction
@dataclass
class CorrectionAudit:
    fixed: int = 0          # was wrong, now right
    damaged: int = 0        # was right, now wrong
    improved: int = 0       # still wrong but closer
    worsened: int = 0       # still wrong and further away
    unchanged: int = 0
    examples_fixed: list = None
    examples_damaged: list = None


def correction_metrics(refs: list[str], before: list[str], after: list[str]) -> dict:
    """Fix/damage breakdown plus net accuracy change."""
    from src.script import edit_distance

    a = CorrectionAudit(examples_fixed=[], examples_damaged=[])
    for r, b, c in zip(refs, before, after):
        rb, bb, cc = to_aksharas(r), to_aksharas(b), to_aksharas(c)
        if bb == cc:
            a.unchanged += 1
            continue
        was_ok, now_ok = bb == rb, cc == rb
        if not was_ok and now_ok:
            a.fixed += 1
            if len(a.examples_fixed) < 10:
                a.examples_fixed.append({"truth": r, "before": b, "after": c})
        elif was_ok and not now_ok:
            a.damaged += 1
            if len(a.examples_damaged) < 10:
                a.examples_damaged.append({"truth": r, "before": b, "after": c})
        else:
            d0, d1 = edit_distance(bb, rb), edit_distance(cc, rb)
            if d1 < d0:
                a.improved += 1
            elif d1 > d0:
                a.worsened += 1

    acc_before = float(np.mean([to_aksharas(r) == to_aksharas(b)
                                for r, b in zip(refs, before)])) if refs else 0.0
    acc_after = float(np.mean([to_aksharas(r) == to_aksharas(c)
                               for r, c in zip(refs, after)])) if refs else 0.0
    n_changed = a.fixed + a.damaged + a.improved + a.worsened
    return {
        "word_accuracy_before": round(acc_before, 4),
        "word_accuracy_after": round(acc_after, 4),
        "net_change": round(acc_after - acc_before, 4),
        "aer_before": round(float(np.mean([akshara_error_rate(r, b)
                                           for r, b in zip(refs, before)])), 4),
        "aer_after": round(float(np.mean([akshara_error_rate(r, c)
                                          for r, c in zip(refs, after)])), 4),
        "fixed": a.fixed, "damaged": a.damaged,
        "improved_not_fixed": a.improved, "worsened_still_wrong": a.worsened,
        "unchanged": a.unchanged,
        "n_changed": n_changed,
        "precision_of_changes": round(a.fixed / max(n_changed, 1), 4),
        "fix_damage_ratio": round(a.fixed / max(a.damaged, 1), 2),
        "examples_fixed": a.examples_fixed,
        "examples_damaged": a.examples_damaged,
    }


# ---------------------------------------------------------------- confusion
def akshara_confusion(refs: list[str], hyps: list[str], top: int = 25) -> list[dict]:
    """Most frequent akshara substitutions - feeds CONFUSABLE_GROUPS in the corrector.

    Run this after your first evaluation and paste the top pairs into
    src/correction/corrector.py. That closes the loop: the corrector's notion of
    "visually similar" becomes your own model's measured confusions rather than
    a hand-written guess, which is a genuine, defensible improvement to report.
    """
    from collections import Counter

    counts: Counter = Counter()
    for r, h in zip(refs, hyps):
        ra, ha = to_aksharas(r), to_aksharas(h)
        for i in range(min(len(ra), len(ha))):
            if ra[i] != ha[i]:
                counts[(ra[i], ha[i])] += 1
    return [{"expected": e, "read_as": g, "count": c}
            for (e, g), c in counts.most_common(top)]


def build_report(refs, hyps, confidences, error_probs=None,
                 detector_threshold=0.5, corrected=None,
                 out_path: str | Path | None = None) -> dict:
    """Assemble the full four-block report."""
    correct = [to_aksharas(r) == to_aksharas(h) for r, h in zip(refs, hyps)]
    report = {
        "recognition": recognition_metrics(refs, hyps),
        "calibration": calibration_metrics(confidences, correct),
        "confusion_pairs": akshara_confusion(refs, hyps),
    }
    if error_probs is not None:
        report["detection"] = detection_metrics(
            [0 if c else 1 for c in correct], error_probs, detector_threshold)
    if corrected is not None:
        report["correction"] = correction_metrics(refs, hyps, corrected)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    return report