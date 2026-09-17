"""
Build the recognition model from a Devanagari-pretrained TrOCR checkpoint.

Why this replaced the earlier character-vocabulary approach
----------------------------------------------------------
The first design kept microsoft/trocr-base-handwritten's pretrained vision
encoder but replaced its English RoBERTa decoder vocabulary with a fresh
101-symbol Devanagari character vocabulary, randomly initialised.

Measured result: 5.7% word accuracy after 5 epochs (~37 min on an RTX 4090),
with training loss falling to 0.06 while generation collapsed into matra
repetition even on training images. Diagnostics ruled out token-id wiring,
label truncation, encoder freezing, BOS placement and cross-attention: the
encoder demonstrably read the images (blank/noise/real inputs produced clearly
different outputs), but a randomly-initialised decoder could not learn stable
Devanagari orthography from 9,495 unique training words.

Using a decoder already pretrained on Devanagari scored 80% word accuracy on the
same validation images with ZERO fine-tuning. That 14x gap is the justification
for this file's existence, and is worth reporting as an experimental result
rather than hidden as an implementation detail.

The checkpoint below is trained on Nepali, which shares the Devanagari script
with Hindi. Fine-tuning on IIIT-HW-Hindi adapts it to Hindi vocabulary while
keeping the script knowledge it already has.
"""

from __future__ import annotations

from pathlib import Path

from transformers import (
    AutoTokenizer,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

DEFAULT_BACKBONE = "aayushpuri01/TrOCR-Devanagari"


def build_model(
    tokenizer=None,
    backbone: str = DEFAULT_BACKBONE,
    max_target_length: int = 32,
    freeze_encoder_layers: int = 0,
):
    """Load the Devanagari-pretrained model and its native tokenizer.

    Returns (model, processor, tokenizer). The `tokenizer` argument is accepted
    and ignored for backwards compatibility with the old call signature.
    """
    processor = TrOCRProcessor.from_pretrained(backbone)
    model = VisionEncoderDecoderModel.from_pretrained(backbone)
    tok = AutoTokenizer.from_pretrained(backbone)

    # Special-token ids come from the checkpoint's own tokenizer. Overriding
    # these with hand-picked values is what broke the previous design.
    model.config.decoder_start_token_id = (
        tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
    )
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = (
        tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
    )
    model.config.vocab_size = model.config.decoder.vocab_size

    # This checkpoint ships max_length=16 and no_repeat_ngram_size=3 in its
    # model config, which silently override generation_config during Trainer
    # evaluation. max_length=16 truncates longer Hindi words, and forbidding
    # repeated 3-grams is actively wrong for Devanagari. Clear both on the model
    # config as well as the generation config, or the values come back.
    for attr, val in (("max_length", max_target_length),
                      ("no_repeat_ngram_size", 0),
                      ("length_penalty", 1.0),
                      ("num_beams", 4),
                      ("early_stopping", True)):
        if hasattr(model.config, attr):
            setattr(model.config, attr, val)

    gc = model.generation_config
    gc.decoder_start_token_id = model.config.decoder_start_token_id
    gc.pad_token_id = model.config.pad_token_id
    gc.eos_token_id = model.config.eos_token_id
    gc.max_length = max_target_length
    # Beam search measurably outperforms greedy here: on an earlier checkpoint it
    # cut AER from 2.30 to 1.24. Note that src/inference/uncertainty.py needs
    # greedy decoding for clean per-step entropy, so build the error-detection
    # dataset with num_beams=1 and use beams only for the final demo.
    gc.num_beams = 4
    gc.early_stopping = True
    gc.no_repeat_ngram_size = 0
    gc.length_penalty = 1.0

    if freeze_encoder_layers > 0:
        _freeze_encoder(model, freeze_encoder_layers)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"backbone         : {backbone}")
    print(f"decoder vocab    : {model.config.decoder.vocab_size}")
    print(f"special ids      : start={gc.decoder_start_token_id} "
          f"pad={gc.pad_token_id} eos={gc.eos_token_id}")
    print(f"trainable params : {n_train/1e6:.1f}M / {n_total/1e6:.1f}M")
    return model, processor, tok


def _encoder_blocks(enc):
    """Locate encoder transformer blocks across transformers versions.

    4.x nests them at encoder.encoder.layer; 5.x exposes encoder.layers.
    """
    for path in (("encoder", "layer"), ("encoder", "layers"), ("layers",), ("layer",)):
        obj = enc
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and len(obj) > 0:
            return obj
    return None


def _freeze_encoder(model, n_layers: int) -> None:
    enc = model.encoder
    emb = getattr(enc, "embeddings", None)
    if emb is not None:
        for p in emb.parameters():
            p.requires_grad = False
    layers = _encoder_blocks(enc)
    if layers is None:
        print("WARNING: could not locate encoder blocks; skipping freezing.")
        return
    for layer in layers[:n_layers]:
        for p in layer.parameters():
            p.requires_grad = False
    print(f"froze encoder embeddings + first {n_layers}/{len(layers)} blocks")


def load_trained(checkpoint: str | Path, charset=None):
    """Load the fine-tuned Hindi handwritten OCR checkpoint."""

    from pathlib import Path as _P
    from transformers import AutoConfig

    ck = _P(str(checkpoint))

    # The trained model used our custom Devanagari character tokenizer.
    cs = ck / "charset.json"

    if cs.exists():
        from src.script import DevanagariCharTokenizer
        tok = DevanagariCharTokenizer.load(cs)
        vocab_size = len(tok)
        print(f"char tokenizer loaded ({vocab_size} tokens)")
    else:
        tok = AutoTokenizer.from_pretrained(str(ck))
        vocab_size = len(tok)
        print(f"tokenizer loaded ({vocab_size} tokens)")

    # IMPORTANT:
    # The checkpoint weights contain 85 decoder embeddings.
    # Force the model configuration to use the same vocabulary size
    # before loading the safetensors weights.
    config = AutoConfig.from_pretrained(str(ck))

    config.decoder.vocab_size = vocab_size
    config.vocab_size = vocab_size

    model = VisionEncoderDecoderModel.from_pretrained(
        str(ck),
        config=config,
        ignore_mismatched_sizes=False,
    )

    try:
        processor = TrOCRProcessor.from_pretrained(str(ck))
    except Exception:
        processor = TrOCRProcessor.from_pretrained(DEFAULT_BACKBONE)

    model.config.decoder_start_token_id = (
        tok.cls_token_id
        if getattr(tok, "cls_token_id", None) is not None
        else tok.bos_token_id
    )

    model.config.pad_token_id = tok.pad_token_id

    model.config.eos_token_id = (
        tok.sep_token_id
        if getattr(tok, "sep_token_id", None) is not None
        else tok.eos_token_id
    )

    model.eval()

    print(f"trained decoder vocab : {model.config.decoder.vocab_size}")

    return model, processor, tok