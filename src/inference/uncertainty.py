"""
Confidence and uncertainty estimation for sequence recognition.

Why the distinction matters
--------------------------
Confidence and uncertainty are not the same quantity, and conflating them is the
gap your project targets. A model can be *confident and wrong* (it puts 0.97 on
a single wrong akshara because the handwriting genuinely resembles a different
letter) and it can be *uncertain and right* (0.4/0.35 split between two matras,
picking the correct one). A single softmax number cannot separate these.

So we measure three different things per prediction:

1. **Confidence** - how much probability mass the model put on what it chose.
   Cheap, comes free with generation, but poorly calibrated in seq2seq models
   because the decoder is trained with teacher forcing and never sees its own
   mistakes.

2. **Uncertainty (aleatoric-ish)** - predictive entropy of the full output
   distribution at each step. High entropy means the model is genuinely torn
   across many options, which is different from being narrowly torn between two.

3. **Uncertainty (epistemic-ish)** - disagreement across stochastic forward
   passes with dropout active (MC dropout). This is the signal that catches
   "confident and wrong": if the answer flips when you perturb the network, the
   confidence was an artefact.

The features below are what feed the error-detection classifier. None of them is
reliable alone; the classifier's job is to learn their joint decision boundary,
which is exactly why this is framed as a learned module rather than a threshold.

Implementation note on `scores` vs `logits`
------------------------------------------
`generate(..., output_scores=True)` returns logits AFTER logits processors have
run, and under beam search they are cumulative beam scores rather than per-step
distributions. Entropy over those is meaningless. We request `output_logits=True`
instead, which gives raw per-step logits, and default to greedy decoding so each
step has exactly one well-defined distribution.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

# Calibration metrics live in src.evaluate.calibration so that they can be used
# without importing torch (reports, notebooks, CI). Re-exported here for
# convenience because they are conceptually part of uncertainty estimation.
from src.evaluate.calibration import (  # noqa: E402
    expected_calibration_error,
    risk_coverage_curve,
)


@dataclass
class Prediction:
    """One recognised word plus everything we know about how sure we are."""

    text: str
    tokens: list = field(default_factory=list)
    token_probs: list = field(default_factory=list)
    token_entropies: list = field(default_factory=list)
    token_margins: list = field(default_factory=list)
    mc_variants: list = field(default_factory=list)

    # ---- aggregate confidence ----
    @property
    def mean_prob(self) -> float:
        return float(np.mean(self.token_probs)) if self.token_probs else 0.0

    @property
    def min_prob(self) -> float:
        """Weakest link. Usually the single most predictive feature - a word is
        wrong if ANY akshara is wrong, so the minimum matters more than the mean."""
        return float(np.min(self.token_probs)) if self.token_probs else 0.0

    @property
    def norm_logprob(self) -> float:
        """Length-normalised sequence log-probability."""
        if not self.token_probs:
            return -20.0
        lp = np.log(np.clip(self.token_probs, 1e-12, 1.0))
        return float(lp.sum() / len(lp))

    # ---- aggregate uncertainty ----
    @property
    def mean_entropy(self) -> float:
        return float(np.mean(self.token_entropies)) if self.token_entropies else 0.0

    @property
    def max_entropy(self) -> float:
        return float(np.max(self.token_entropies)) if self.token_entropies else 0.0

    @property
    def min_margin(self) -> float:
        """Smallest top1-top2 gap across steps: the closest call the model made."""
        return float(np.min(self.token_margins)) if self.token_margins else 0.0

    @property
    def mc_disagreement(self) -> float:
        """Fraction of dropout passes that disagreed with the deterministic output."""
        if not self.mc_variants:
            return 0.0
        return float(np.mean([v != self.text for v in self.mc_variants]))

    @property
    def n_tokens(self) -> int:
        return len(self.token_probs)

    def summary(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.mean_prob, 4),
            "min_token_confidence": round(self.min_prob, 4),
            "mean_entropy": round(self.mean_entropy, 4),
            "max_entropy": round(self.max_entropy, 4),
            "min_margin": round(self.min_margin, 4),
            "mc_disagreement": round(self.mc_disagreement, 4),
            "n_tokens": self.n_tokens,
        }

    def weakest_position(self) -> int:
        """Index of the least confident token - used to highlight in the UI."""
        if not self.token_probs:
            return -1
        return int(np.argmin(self.token_probs))


def _entropy(logits: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Shannon entropy of softmax(logits), optionally scaled to [0, 1].

    Normalising by log(V) makes entropy comparable across models with different
    vocabulary sizes, which matters if you later swap the character vocabulary
    for a subword one - otherwise your trained error detector silently breaks.
    """
    logp = F.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    h = -(p * logp).sum(dim=-1)
    if normalize:
        h = h / float(np.log(logits.shape[-1]))
    return h


@contextlib.contextmanager
def dropout_active(model: torch.nn.Module):
    """Turn dropout on while leaving normalisation layers in eval mode.

    A plain `model.train()` would also unfreeze BatchNorm/LayerNorm running
    statistics and corrupt them during inference. We flip only Dropout modules.
    """
    changed = []
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d)):
            if not m.training:
                m.train()
                changed.append(m)
    try:
        yield model
    finally:
        for m in changed:
            m.eval()


@torch.no_grad()
def predict_with_uncertainty(
    model,
    tokenizer,
    pixel_values: torch.Tensor,
    max_length: int = 48,
    num_beams: int = 1,
    mc_passes: int = 0,
    device: str | None = None,
) -> list:
    """Recognise a batch of word images and quantify uncertainty for each.

    Args:
        pixel_values: (B, 3, H, W) already normalised by the image processor.
        num_beams: keep at 1 for clean per-step distributions. Beam search
            improves text slightly but makes the uncertainty features
            inconsistent with what the classifier was trained on.
        mc_passes: number of MC-dropout passes. 0 disables (fast path).
            5-10 is plenty; the marginal information past 10 is small and the
            cost is linear.

    Returns:
        One Prediction per input image.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    pixel_values = pixel_values.to(device)

    out = model.generate(
        pixel_values,
        max_length=max_length,
        num_beams=num_beams,
        no_repeat_ngram_size=0,
        do_sample=False,
        output_logits=True,
        return_dict_in_generate=True,
    )

    seqs = out.sequences.cpu()                     # (B, T) incl. decoder_start
    step_logits = out.logits                       # tuple of T-1 x (B, V)
    B = seqs.shape[0]

    preds: list = []
    for b in range(B):
        probs, ents, margins, toks = [], [], [], []
        for t, logits_t in enumerate(step_logits):
            tok = int(seqs[b, t + 1])              # step t produced position t+1
            if tok in (tokenizer.eos_token_id, tokenizer.pad_token_id):
                break
            lg = logits_t[b]
            p = F.softmax(lg.float(), dim=-1)
            top2 = torch.topk(p, k=min(2, p.shape[-1])).values
            probs.append(float(p[tok]))
            ents.append(float(_entropy(lg.unsqueeze(0))[0]))
            margins.append(float(top2[0] - top2[1]) if top2.numel() > 1 else 1.0)
            toks.append(tok)

        preds.append(Prediction(
            text=tokenizer.decode(seqs[b].tolist(), skip_special_tokens=True).strip(),
            tokens=toks,
            token_probs=probs,
            token_entropies=ents,
            token_margins=margins,
        ))

    if mc_passes > 0:
        variants = _mc_dropout_passes(model, tokenizer, pixel_values,
                                      max_length, mc_passes)
        for b, pred in enumerate(preds):
            pred.mc_variants = [v[b] for v in variants]

    return preds


@torch.no_grad()
def _mc_dropout_passes(model, tokenizer, pixel_values, max_length, n) -> list:
    """n stochastic decodes of the same batch, dropout active."""
    results = []
    with dropout_active(model):
        for _ in range(n):
            out = model.generate(pixel_values, max_length=max_length,
                                 num_beams=1, do_sample=False)
            results.append([t.strip() for t in tokenizer.batch_decode(out.cpu().tolist(), skip_special_tokens=True)])
    return results