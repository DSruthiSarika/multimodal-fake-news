"""
download_images.py
==================
Pre-downloads all images from the dataset TSV files into a local cache
directory before training. Run this once so the training loop never
stalls on network I/O.

Usage:
    python download_images.py --splits train val test
    python download_images.py --splits train --workers 16
"""

import os
import io
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from PIL import Image
from tqdm import tqdm

from dataset import url_to_filename

# ── Default paths ─────────────────────────────────────────────────────────────
SPLIT_PATHS = {
    "train": "multimodal_train.tsv",
    "val":   "multimodal_validate.tsv",
    "test":  "multimodal_test_public.tsv",
}
DEFAULT_CACHE_DIR = "image_cache"
TIMEOUT           = 5    # seconds per request
IMG_SIZE          = 224  # resize on save to keep cache small


def download_and_cache(url: str, cache_dir: str) -> bool:
    """
    Download one image and save as JPEG.
    Returns True on success, False on any error.
    """
    fname = url_to_filename(url)
    dest  = Path(cache_dir) / fname
    if dest.exists():
        return True  # already cached

    try:
        resp = requests.get(url, timeout=TIMEOUT, stream=True)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img.save(dest, format="JPEG", quality=85)
        return True
    except Exception:
        return False


def collect_urls(splits: list[str]) -> list[str]:
    """Read TSV files and return a deduplicated list of image URLs."""
    urls = set()
    for split in splits:
        path = SPLIT_PATHS.get(split)
        if not path or not Path(path).exists():
            print(f"[WARN] TSV not found for split '{split}': {path}")
            continue
        df = pd.read_csv(path, sep="\t", usecols=["image_url", "hasImage"])
        df = df[df["hasImage"].astype(bool)]  # only rows with confirmed images
        df = df.dropna(subset=["image_url"])
        urls.update(df["image_url"].tolist())
    return list(urls)


def main():
    parser = argparse.ArgumentParser(description="Pre-download dataset images")
    parser.add_argument("--splits",    nargs="+", default=["train", "val", "test"],
                        choices=["train", "val", "test"])
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--workers",   type=int, default=8,
                        help="Parallel download threads")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    urls = collect_urls(args.splits)
    print(f"Total unique URLs to download: {len(urls):,}")

    success = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_and_cache, url, args.cache_dir): url
                   for url in urls}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            if fut.result():
                success += 1
            else:
                fail += 1

    print(f"\nDone — success: {success:,} | failed: {fail:,}")
    print(f"Images saved to: {args.cache_dir}/")


if __name__ == "__main__":
    main()
