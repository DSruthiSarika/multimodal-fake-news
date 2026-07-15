"""
evaluate.py
===========
Metrics helpers + full test-set evaluation.

Changes from v1:
  - CONFIG imported from config.py (no circular dependency with main.py)
  - AMP autocast applied during inference

Standalone usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import logging

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
    confusion_matrix,
)
from tqdm import tqdm

from config  import CONFIG
from dataset import FakeNewsDataset
from model   import MultimodalFakeNewsClassifier

logger     = logging.getLogger("fakenews")
LABEL_NAMES = ["Real", "Fake"]


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_metrics(
    labels: list | np.ndarray,
    preds:  list | np.ndarray,
    probs:  np.ndarray = None,
) -> dict:
    """Compute standard classification metrics."""
    labels = np.array(labels)
    preds  = np.array(preds)

    metrics = {
        "accuracy":    accuracy_score(labels, preds),
        "f1_macro":    f1_score(labels, preds, average="macro",    zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "f1_fake":     f1_score(labels, preds, pos_label=1,        zero_division=0),
        "precision":   precision_score(labels, preds, average="macro", zero_division=0),
        "recall":      recall_score(labels, preds,    average="macro", zero_division=0),
    }
    if probs is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(labels, probs[:, 1])
        except Exception:
            pass
    return metrics


def print_report(labels, preds, probs=None) -> dict:
    metrics = compute_metrics(labels, preds, probs)
    print("\n" + "=" * 55)
    print("EVALUATION RESULTS")
    print("=" * 55)
    for k, v in metrics.items():
        print(f"  {k:<15}: {v:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=LABEL_NAMES))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(labels, preds)
    print(f"  {'':10} {'Real':>8} {'Fake':>8}")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:<10} {cm[i, 0]:>8} {cm[i, 1]:>8}")
    print("=" * 55)
    return metrics


# ── Full test-set evaluation ──────────────────────────────────────────────────

def evaluate(checkpoint_path: str, cfg: dict) -> dict:
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

    test_ds = FakeNewsDataset(
        cfg["test_path"], tokenizer, cfg["max_text_len"],
        augment=False, limit=cfg.get("test_limit"),
        cache_dir=cfg["img_cache_dir"], img_timeout=cfg["img_timeout"],
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )
    logger.info(f"Test samples : {len(test_ds):,}")

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids, attn_mask, images, labels = [b.to(device) for b in batch]
            with autocast(enabled=device.type == "cuda"):
                logits = model(input_ids, attn_mask, images)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    return print_report(all_labels, all_preds, np.array(all_probs))


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    evaluate(args.checkpoint, CONFIG)
