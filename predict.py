"""
predict.py
==========
Single-sample inference for the multimodal fake-news classifier.

Changes from v1:
  - CONFIG imported from config.py (no circular dependency)
  - AMP autocast applied during inference

Standalone usage:
    python predict.py \
        --checkpoint checkpoints/best_model.pt \
        --title "Shocking new discovery changes everything" \
        --image_url "https://i.imgur.com/example.jpg"
"""

import argparse
import logging

import numpy as np
import torch
from torch.cuda.amp import autocast
from transformers import BertTokenizer

from config  import CONFIG
from dataset import load_image, EVAL_TRANSFORM
from model   import MultimodalFakeNewsClassifier

logger = logging.getLogger("fakenews")


def predict_single(
    checkpoint_path: str,
    cfg:             dict,
    title:           str,
    image_url:       str,
) -> dict:
    """
    Run inference on one (title, image_url) pair.

    Returns:
        {
          "prediction" : "REAL" | "FAKE",
          "confidence" : float,
          "prob_real"  : float,
          "prob_fake"  : float,
        }
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt      = torch.load(checkpoint_path, map_location=device)
    tokenizer = BertTokenizer.from_pretrained(cfg["bert_model"])

    model = MultimodalFakeNewsClassifier(
        bert_model_name=cfg["bert_model"],
        fusion_dim=cfg["fusion_dim"],
        num_classes=cfg["num_classes"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Tokenise text ─────────────────────────────────────────────────────────
    enc = tokenizer(
        title,
        max_length=cfg["max_text_len"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # ── Load & transform image ────────────────────────────────────────────────
    img        = load_image(image_url, cfg["img_cache_dir"], cfg["img_timeout"])
    img_tensor = EVAL_TRANSFORM(img).unsqueeze(0).to(device)

    # ── Inference with AMP ────────────────────────────────────────────────────
    with torch.no_grad():
        with autocast(enabled=device.type == "cuda"):
            logits = model(input_ids, attention_mask, img_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    pred   = int(np.argmax(probs))
    result = {
        "prediction": "FAKE" if pred == 1 else "REAL",
        "confidence": float(probs[pred]),
        "prob_real":  float(probs[0]),
        "prob_fake":  float(probs[1]),
    }

    _print_result(title, image_url, result)
    return result


def _print_result(title: str, image_url: str, result: dict):
    bar_len  = 30
    fake_bar = int(result["prob_fake"] * bar_len)
    real_bar = bar_len - fake_bar

    print("\n" + "=" * 55)
    print("PREDICTION")
    print("=" * 55)
    print(f"  Title      : {title[:70]}")
    print(f"  Image URL  : {image_url[:70]}")
    print(f"  Result     : {result['prediction']}  ({result['confidence']*100:.1f}% confidence)")
    print(f"  P(real)    : {result['prob_real']:.4f}  {'█' * real_bar}")
    print(f"  P(fake)    : {result['prob_fake']:.4f}  {'█' * fake_bar}")
    print("=" * 55)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--title",      required=True)
    parser.add_argument("--image_url",  required=True)
    args = parser.parse_args()
    predict_single(args.checkpoint, CONFIG, args.title, args.image_url)
