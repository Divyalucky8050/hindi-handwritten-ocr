"""
Fine-tune the recognition model.

    python -m src.models.train --config config.yaml

Everything tunable lives in config.yaml. Resume with --resume <checkpoint-dir>.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.data.dataset import Collator, HandwritingAugment, WordImageDataset, load_jsonl
from src.models.build import build_model
from src.script import akshara_error_rate


def build_compute_metrics(tokenizer):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred.predictions, eval_pred.label_ids
        if isinstance(preds, tuple):
            preds = preds[0]
        if preds.ndim == 3:            # logits, not generated ids
            preds = preds.argmax(-1)
        # Trainer pads BOTH labels and generated predictions with -100, which is
        # not a valid token id: batch_decode raises OverflowError on it. Clean
        # both arrays, not just labels. Getting this wrong makes every metric
        # meaningless while training itself proceeds correctly - which is exactly
        # what happened during development (identical eval_loss across runs).
        labels = np.where(labels < 0, tokenizer.pad_token_id, labels)
        preds = np.where(preds < 0, tokenizer.pad_token_id, preds)

        print(f"\n[DEBUG] preds shape={preds.shape} dtype={preds.dtype} "
              f"min={preds.min()} max={preds.max()}")
        print(f"[DEBUG] labels shape={labels.shape} min={labels.min()} max={labels.max()}")
        print(f"[DEBUG] preds[0]={preds[0][:15].tolist()}")
        print(f"[DEBUG] labels[0]={labels[0][:15].tolist()}")
        hyp = tokenizer.batch_decode(preds, skip_special_tokens=True)
        ref = tokenizer.batch_decode(labels, skip_special_tokens=True)

        aer = float(np.mean([akshara_error_rate(r, h) for r, h in zip(ref, hyp)]))
        wer = float(np.mean([r != h for r, h in zip(ref, hyp)]))
        exact = 1.0 - wer
        return {"aer": aer, "wer": wer, "word_accuracy": exact}

    return compute_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml", type=Path)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    dcfg, mcfg, tcfg = cfg["data"], cfg["model"], cfg["train"]

    proc_dir = Path(dcfg["processed_dir"])
    model, processor, tokenizer = build_model(
        None,
        backbone=mcfg["backbone"],
        max_target_length=mcfg["max_target_length"],
        freeze_encoder_layers=mcfg.get("freeze_encoder_layers", 0),
    )

    train_rows = load_jsonl(proc_dir / "train.jsonl")
    val_rows = load_jsonl(proc_dir / "val.jsonl")
    if tcfg.get("max_train_samples"):
        # Smoke-test path: prove the plumbing works on a laptop before paying
        # for GPU hours. Accuracy from this is meaningless by design.
        train_rows = train_rows[: tcfg["max_train_samples"]]
    if tcfg.get("val_subset"):
        # Generative eval is slow. Evaluating on a fixed subset each epoch
        # instead of the full val set is the difference between a 6-hour and
        # a 14-hour run, and the metric ranking is unchanged.
        rng = np.random.default_rng(0)
        idx = rng.choice(len(val_rows), min(tcfg["val_subset"], len(val_rows)),
                         replace=False)
        val_rows = [val_rows[i] for i in idx]
    print(f"train: {len(train_rows)}  val: {len(val_rows)}")

    aug = HandwritingAugment(p=tcfg.get("augment_prob", 0.5), seed=tcfg.get("seed", 42))
    train_ds = WordImageDataset(train_rows, processor, tokenizer,
                                mcfg["max_target_length"], augment=aug)
    val_ds = WordImageDataset(val_rows, processor, tokenizer,
                              mcfg["max_target_length"], augment=None)

    out_dir = Path(tcfg["output_dir"])
    targs = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=tcfg["batch_size"],
        per_device_eval_batch_size=tcfg.get("eval_batch_size", tcfg["batch_size"]),
        gradient_accumulation_steps=tcfg.get("grad_accum", 1),
        learning_rate=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 0.01)),
        warmup_ratio=float(tcfg.get("warmup_ratio", 0.05)),
        num_train_epochs=tcfg["epochs"],
        lr_scheduler_type=tcfg.get("scheduler", "cosine"),
        fp16=tcfg.get("fp16", True) and torch.cuda.is_available(),
        bf16=tcfg.get("bf16", False),
        gradient_checkpointing=tcfg.get("gradient_checkpointing", False),
        dataloader_num_workers=tcfg.get("num_workers", 4),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=tcfg.get("logging_steps", 100),
        save_total_limit=tcfg.get("save_total_limit", 2),
        predict_with_generate=True,
        generation_max_length=mcfg["max_target_length"],
        generation_num_beams=4,
        load_best_model_at_end=True,
        metric_for_best_model="aer",
        greater_is_better=False,
        report_to=tcfg.get("report_to", ["none"]),
        seed=tcfg.get("seed", 42),
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=Collator(tokenizer.pad_token_id),
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=tcfg.get("patience", 3))],
    )

    trainer.train(resume_from_checkpoint=args.resume)

    final = out_dir / "best"
    trainer.save_model(str(final))
    processor.save_pretrained(str(final))
    tokenizer.save_pretrained(str(final))

    metrics = trainer.evaluate()
    (out_dir / "final_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nfinal:", json.dumps({k: round(v, 4) for k, v in metrics.items()
                                 if isinstance(v, float)}, indent=2))
    print(f"saved -> {final}")


if __name__ == "__main__":
    main()