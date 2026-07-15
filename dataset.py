"""
dataset.py
==========
PyTorch Dataset for multimodal fake-news classification.

Each sample returns:
    input_ids       : [max_len]     BERT token ids
    attention_mask  : [max_len]     BERT attention mask
    image           : [3, H, W]     normalised image tensor
    label           : scalar int    0 = real, 1 = fake
"""

import io
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer


# ── Image transforms ──────────────────────────────────────────────────────────

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Image loading helpers ─────────────────────────────────────────────────────

def url_to_filename(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest() + ".jpg"


def load_image(url: str, cache_dir: str, timeout: int) -> Image.Image:
    """
    Load image from local cache first; fall back to live download.
    Returns a PIL RGB image, or a black placeholder on any error.
    """
    cache_path = Path(cache_dir) / url_to_filename(url)

    # 1. Try cache
    if cache_path.exists():
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception:
            cache_path.unlink(missing_ok=True)

    # 2. Try live download
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return img
    except Exception:
        pass

    # 3. Placeholder
    return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))


# ── Dataset ───────────────────────────────────────────────────────────────────

class FakeNewsDataset(Dataset):
    """
    Args:
        tsv_path    : path to one of the multimodal_*.tsv files
        tokenizer   : BertTokenizer instance
        max_len     : maximum token length for text
        augment     : if True, use training augmentations on images
        limit       : cap the number of rows (None = full dataset)
        cache_dir   : directory where pre-downloaded images are stored
        img_timeout : seconds before giving up on a live download
    """

    def __init__(
        self,
        tsv_path:    str,
        tokenizer:   BertTokenizer,
        max_len:     int  = 128,
        augment:     bool = False,
        limit:       int  = None,
        cache_dir:   str  = "image_cache",
        img_timeout: int  = 5,
    ):
        df = pd.read_csv(tsv_path, sep="\t")
        if limit:
            df = df.head(limit)

        df["text"] = df["clean_title"].fillna(df["title"]).fillna("").astype(str)
        df = df[df["hasImage"] == True]  # only rows with confirmed images
        df = df[["text", "image_url", "2_way_label"]].dropna(subset=["image_url"])
        df["2_way_label"] = df["2_way_label"].astype(int)

        self.texts      = df["text"].tolist()
        self.image_urls = df["image_url"].tolist()
        self.labels     = df["2_way_label"].tolist()

        self.tokenizer  = tokenizer
        self.max_len    = max_len
        self.transform  = TRAIN_TRANSFORM if augment else EVAL_TRANSFORM
        self.cache_dir  = cache_dir
        self.timeout    = img_timeout

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        # ── Text ──────────────────────────────────────────────────────────────
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)       # [max_len]
        attention_mask = enc["attention_mask"].squeeze(0)  # [max_len]

        # ── Image ─────────────────────────────────────────────────────────────
        img        = load_image(self.image_urls[idx], self.cache_dir, self.timeout)
        img_tensor = self.transform(img)                   # [3, 224, 224]

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, attention_mask, img_tensor, label


# ── Class weights (for imbalanced loss) ──────────────────────────────────────

def compute_class_weights(tsv_path: str) -> torch.Tensor:
    """Returns a [2] float tensor with inverse-frequency weights."""
    counts = pd.read_csv(tsv_path, sep="\t", usecols=["2_way_label"])["2_way_label"].value_counts()
    total  = counts.sum()
    return torch.tensor([total / counts[i] for i in range(2)], dtype=torch.float32)
