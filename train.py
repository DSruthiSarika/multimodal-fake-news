"""
train.py
========
Training loop for the multimodal fake-news classifier.

Changes from v1:
  - Full random seed coverage (Python, NumPy, PyTorch, CUDA)
  - Mixed precision training via torch.cuda.amp (AMP)
  - Early stopping based on validation macro-F1
  - CONFIG imported from config.py (no circular dependency)

Called by main.py --mode train, or run directly:
    python train.py
"""

import os
import time
import random
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from config  import CONFIG
from dataset import FakeNewsDataset, compute_class_weights
from model   import MultimodalFakeNewsClassifier, count_parameters
from evaluate import compute_metrics

logger = logging.getLogger("fakenews")


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── One training epoch (AMP-enabled) ─────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler:    GradScaler,
    device:    torch.device,
    grad_clip: float,
) -> tuple[float, dict]:
    model.train()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for batch in tqdm(loader, desc="  Train", leave=False):
        input_ids, attn_mask, images, labels = [b.to(device) for b in batch]

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast(enabled=device.type == "cuda"):
            logits = model(input_ids, attn_mask, images)
            loss   = criterion(logits, labels)

        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)

        total_loss += loss.item() * len(labels)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)

    avg_loss = total_loss / len(all_labels)
    metrics  = compute_metrics(all_labels, all_preds, np.array(all_probs))
    return avg_loss, metrics


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for batch in tqdm(loader, desc="  Val  ", leave=False):
        input_ids, attn_mask, images, labels = [b.to(device) for b in batch]

        with autocast(enabled=device.type == "cuda"):
            logits = model(input_ids, attn_mask, images)
            loss   = criterion(logits, labels)

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        total_loss += loss.item() * len(labels)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)

    avg_loss = total_loss / len(all_labels)
    metrics  = compute_metrics(all_labels, all_preds, np.array(all_probs))
    return avg_loss, metrics


# ── Main training pipeline ────────────────────────────────────────────────────

def train(cfg: dict) -> str:
    """
    Full training pipeline with AMP, early stopping, and full seed control.

    Args:
        cfg : config dict (from config.py, optionally overridden by CLI args)

    Returns:
        Path to the best checkpoint file.
    """
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device     : {device}")
    logger.info(f"AMP active : {device.type == 'cuda'}")

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    tokenizer = BertTokenizer.from_pretrained(cfg["bert_model"])

    # ── Datasets & loaders ────────────────────────────────────────────────────
    logger.info("Loading datasets …")
    train_ds = FakeNewsDataset(
        cfg["train_path"], tokenizer, cfg["max_text_len"],
        augment=True,  limit=cfg.get("train_limit"),
        cache_dir=cfg["img_cache_dir"], img_timeout=cfg["img_timeout"],
    )
    val_ds = FakeNewsDataset(
        cfg["val_path"], tokenizer, cfg["max_text_len"],
        augment=False, limit=cfg.get("val_limit"),
        cache_dir=cfg["img_cache_dir"], img_timeout=cfg["img_timeout"],
    )
    logger.info(f"Train : {len(train_ds):,} | Val : {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MultimodalFakeNewsClassifier(
        bert_model_name=cfg["bert_model"],
        fusion_dim=cfg["fusion_dim"],
        num_classes=cfg["num_classes"],
        dropout=cfg["dropout"],
    ).to(device)

    p = count_parameters(model)
    logger.info(f"Parameters : total={p['total']:,} | trainable={p['trainable']:,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    weights   = compute_class_weights(cfg["train_path"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )
    total_steps  = len(train_loader) * cfg["num_epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # AMP scaler (no-op on CPU)
    scaler = GradScaler(enabled=device.type == "cuda")

    # ── Training loop with early stopping ─────────────────────────────────────
    best_f1      = 0.0
    patience_ctr = 0
    patience     = cfg.get("early_stopping_patience", 3)
    best_path    = Path(cfg["checkpoint_dir"]) / "best_model.pt"

    for epoch in range(1, cfg["num_epochs"] + 1):
        logger.info(f"\nEpoch {epoch}/{cfg['num_epochs']}")
        t0 = time.time()

        train_loss, train_m = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, cfg["grad_clip"]
        )
        val_loss, val_m = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        logger.info(
            f"  [{elapsed:.0f}s] "
            f"Train loss={train_loss:.4f} acc={train_m['accuracy']:.4f} f1={train_m['f1_macro']:.4f} | "
            f"Val   loss={val_loss:.4f}   acc={val_m['accuracy']:.4f}   f1={val_m['f1_macro']:.4f}"
        )

        # ── Checkpoint if best ────────────────────────────────────────────────
        if val_m["f1_macro"] > best_f1:
            best_f1      = val_m["f1_macro"]
            patience_ctr = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_metrics": val_m,
                "config":      cfg,
            }, best_path)
            logger.info(f"  ✓ Best model saved → {best_path}  (val F1={best_f1:.4f})")
        else:
            patience_ctr += 1
            logger.info(f"  No improvement. Early stopping counter: {patience_ctr}/{patience}")
            if patience_ctr >= patience:
                logger.info(f"  Early stopping triggered at epoch {epoch}.")
                break

        # Per-epoch checkpoint
        epoch_path = Path(cfg["checkpoint_dir"]) / f"epoch_{epoch:02d}.pt"
        torch.save({"epoch": epoch, "model_state": model.state_dict()}, epoch_path)

    logger.info(f"\nTraining complete. Best val macro-F1: {best_f1:.4f}")
    return str(best_path)
