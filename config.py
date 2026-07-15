"""
config.py
=========
Single source of truth for all hyperparameters and file paths.
Imported by main.py, train.py, evaluate.py, and predict.py.
Eliminates the circular dependency where evaluate.py imported CONFIG from main.py.
"""

CONFIG = {
    # ── Data paths ────────────────────────────────────────────────────────────
    "train_path":     "multimodal_train.tsv",
    "val_path":       "multimodal_validate.tsv",
    "test_path":      "multimodal_test_public.tsv",

    # ── Image cache ───────────────────────────────────────────────────────────
    "img_cache_dir":  "image_cache",
    "img_timeout":    5,

    # ── Output ────────────────────────────────────────────────────────────────
    "checkpoint_dir": "checkpoints",

    # ── Model ─────────────────────────────────────────────────────────────────
    "bert_model":     "bert-base-uncased",
    "max_text_len":   128,
    "image_size":     224,
    "fusion_dim":     512,
    "dropout":        0.3,
    "num_classes":    2,

    # ── Training ──────────────────────────────────────────────────────────────
    "batch_size":     8,
    "num_epochs":     10,
    "lr":             2e-5,
    "weight_decay":   1e-4,
    "grad_clip":      1.0,
    "warmup_ratio":   0.1,
    "num_workers":    4,
    "seed":           42,

    # ── Early stopping ────────────────────────────────────────────────────────
    "early_stopping_patience": 3,   # stop if val F1 doesn't improve for N epochs

    # ── Sample limits (None = full dataset) ───────────────────────────────────
    "train_limit":    None,
    "val_limit":      None,
    "test_limit":     None,
}
