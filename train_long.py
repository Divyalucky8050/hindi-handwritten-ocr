"""
Long run: 12 epochs on characters + printed words + handwritten words.

Data is larger than the 90.85% run (100,650 chars vs 40,260; 161,638 printed
words vs 80,000) and now includes elastic distortion in generation plus grid
distortion and perspective warp in augmentation.

Fresh start rather than resuming the 8-epoch checkpoint: the data distribution
changed substantially, so a clean cosine schedule over the new mix is a better
fit than adapting a model that converged on the old one. The previous run showed
no plateau by epoch 8, so a longer schedule on more data should exceed it.

Batch 32 kept unchanged - that exact setting produced 90.85% end-to-end, and a
larger batch would mean fewer weight updates plus an LR needing retuning.
"""
import json, numpy as np, torch
from pathlib import Path
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          EarlyStoppingCallback, TrOCRProcessor,
                          VisionEncoderDecoderModel)
from src.data.dataset import Collator, HandwritingAugment, WordImageDataset, load_jsonl
from src.script import DevanagariCharTokenizer, akshara_error_rate

chars = load_jsonl("data/chars2/train.jsonl")
printed = load_jsonl("data/synthetic2/train.jsonl")
hand = load_jsonl("data/processed/train.jsonl")
val = load_jsonl("data/processed/val.jsonl")
print(f"chars {len(chars)}  printed {len(printed)}  handwritten {len(hand)}")

train_rows = chars + printed + hand
rng = np.random.default_rng(0)
rng.shuffle(train_rows)
val_rows = [val[i] for i in rng.choice(len(val), 1500, replace=False)]
print(f"combined {len(train_rows)}  val {len(val_rows)} (handwritten only)")

tok = DevanagariCharTokenizer.from_texts([r["text"] for r in train_rows], min_freq=2)
tok.save("data/processed/charset_long.json")
print(f"charset {len(tok)}")

B = "microsoft/trocr-base-handwritten"
proc = TrOCRProcessor.from_pretrained(B)
model = VisionEncoderDecoderModel.from_pretrained(B)
model.decoder.resize_token_embeddings(len(tok))
model.config.decoder.vocab_size = model.decoder.config.vocab_size = len(tok)
with torch.no_grad():
    emb = model.decoder.get_input_embeddings()
    emb.weight.normal_(0.0, 0.02); emb.weight[tok.pad_token_id].zero_()
for cfg in (model.config, model.generation_config):
    for a, v in (("decoder_start_token_id", tok.bos_token_id),
                 ("bos_token_id", tok.bos_token_id),
                 ("pad_token_id", tok.pad_token_id),
                 ("eos_token_id", tok.eos_token_id),
                 ("max_length", 24), ("num_beams", 4),
                 ("no_repeat_ngram_size", 0), ("length_penalty", 0.7),
                 ("early_stopping", True), ("vocab_size", len(tok))):
        if hasattr(cfg, a): setattr(cfg, a, v)

def metrics(ep):
    p, l = ep.predictions, ep.label_ids
    if isinstance(p, tuple): p = p[0]
    if p.ndim == 3: p = p.argmax(-1)
    p = np.where(p < 0, tok.pad_token_id, p)
    l = np.where(l < 0, tok.pad_token_id, l)
    h, r = tok.batch_decode(p), tok.batch_decode(l)
    return {"aer": float(np.mean([akshara_error_rate(a, b) for a, b in zip(r, h)])),
            "word_accuracy": float(np.mean([a == b for a, b in zip(r, h)]))}

Seq2SeqTrainer(
    model=model,
    args=Seq2SeqTrainingArguments(
        output_dir="outputs/long", num_train_epochs=12,
        per_device_train_batch_size=32, gradient_accumulation_steps=2,
        per_device_eval_batch_size=64, learning_rate=4e-5,
        warmup_ratio=0.02, weight_decay=0.01, lr_scheduler_type="cosine",
        label_smoothing_factor=0.1, fp16=True, dataloader_num_workers=8,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        predict_with_generate=True, generation_max_length=24,
        generation_num_beams=4, load_best_model_at_end=True,
        metric_for_best_model="word_accuracy", greater_is_better=True,
        logging_steps=200, report_to=["none"], remove_unused_columns=False),
    train_dataset=WordImageDataset(train_rows, proc, tok, 24,
                                   augment=HandwritingAugment(p=0.3, seed=42)),
    eval_dataset=WordImageDataset(val_rows, proc, tok, 24, augment=None),
    data_collator=Collator(tok.pad_token_id, decoder_start_token_id=tok.bos_token_id),
    compute_metrics=metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
).train()

out = Path("outputs/long/best"); out.mkdir(parents=True, exist_ok=True)
model.save_pretrained(out); proc.save_pretrained(out); tok.save(out / "charset.json")
print(f"saved -> {out}")
