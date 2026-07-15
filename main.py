"""
main.py
=======
Thin CLI entry point. All config lives in config.py.

Usage:
    python download_images.py --splits train val test --workers 16
    python main.py --mode train
    python main.py --mode train --train_limit 5000 --epochs 2
    python main.py --mode eval    --checkpoint checkpoints/best_model.pt
    python main.py --mode predict --checkpoint checkpoints/best_model.pt \
                   --title "..." --image_url "..."
"""

import argparse
import logging
import sys

from config import CONFIG


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_file: str = "training.log") -> logging.Logger:
    logger = logging.getLogger("fakenews")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)]:
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


logger = setup_logging()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multimodal Fake News Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", required=True, choices=["train", "eval", "predict"])
    parser.add_argument("--checkpoint",  type=str,   default=None)
    parser.add_argument("--title",       type=str,   default=None)
    parser.add_argument("--image_url",   type=str,   default=None)
    parser.add_argument("--train_path",  type=str,   default=CONFIG["train_path"])
    parser.add_argument("--val_path",    type=str,   default=CONFIG["val_path"])
    parser.add_argument("--test_path",   type=str,   default=CONFIG["test_path"])
    parser.add_argument("--batch_size",  type=int,   default=CONFIG["batch_size"])
    parser.add_argument("--epochs",      type=int,   default=CONFIG["num_epochs"])
    parser.add_argument("--lr",          type=float, default=CONFIG["lr"])
    parser.add_argument("--train_limit", type=int,   default=CONFIG["train_limit"])
    parser.add_argument("--num_workers", type=int,   default=CONFIG["num_workers"])
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = {**CONFIG}
    cfg.update({
        "train_path":  args.train_path,
        "val_path":    args.val_path,
        "test_path":   args.test_path,
        "batch_size":  args.batch_size,
        "num_epochs":  args.epochs,
        "lr":          args.lr,
        "train_limit": args.train_limit,
        "num_workers": args.num_workers,
    })

    if args.mode == "train":
        from train import train
        best_ckpt = train(cfg)
        logger.info(f"Best checkpoint : {best_ckpt}")

    elif args.mode == "eval":
        if not args.checkpoint:
            logger.error("--checkpoint is required for eval mode")
            sys.exit(1)
        from evaluate import evaluate
        evaluate(args.checkpoint, cfg)

    elif args.mode == "predict":
        if not args.checkpoint or not args.title or not args.image_url:
            logger.error("--checkpoint, --title, and --image_url are all required for predict mode")
            sys.exit(1)
        from predict import predict_single
        predict_single(args.checkpoint, cfg, args.title, args.image_url)


if __name__ == "__main__":
    main()
