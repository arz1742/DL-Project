"""
data_loader.py

Loads and organizes the Flickr8k dataset (images + captions).
No GPU required.

Expected raw data layout (Kaggle "Flickr8k" release):
    data/raw/
        Images/                  <- all .jpg images
        captions.txt              <- image_name,caption  (one per line, header row)
"""

import os
import json
import random
from collections import defaultdict


def load_captions(captions_path: str) -> dict[str, list[str]]:
    """
    Parses the Flickr8k captions.txt file into
    {image_filename: [caption1, caption2, ...]}
    """
    image_to_captions = defaultdict(list)

    with open(captions_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header = lines[0].strip().lower()
    start_idx = 1 if "image" in header and "caption" in header else 0

    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
        # Format: image_name.jpg,caption text here
        parts = line.split(",", 1)
        if len(parts) != 2:
            continue
        image_name, caption = parts
        image_to_captions[image_name.strip()].append(caption.strip())

    print(f"Loaded captions for {len(image_to_captions)} images "
          f"({sum(len(v) for v in image_to_captions.values())} total captions)")
    return dict(image_to_captions)


def split_dataset(
    image_to_captions: dict[str, list[str]],
    train_ratio: float = 0.85,
    val_ratio: float = 0.075,
    seed: int = 42,
    val_size: int | None = None,
    test_size: int | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """
    Splits image filenames (NOT captions) into train/val/test sets, so all 5
    captions of an image always land in the same split — no leakage.

    If val_size / test_size are given they are used as fixed image counts
    (the project default is 1000 / 1000, matching Kavitha & Karpagam 2025).
    Otherwise the ratios are used.
    """
    image_names = sorted(image_to_captions.keys())
    random.Random(seed).shuffle(image_names)

    n = len(image_names)
    if val_size is not None and test_size is not None:
        if val_size + test_size >= n:
            raise ValueError(f"val_size + test_size ({val_size + test_size}) must be < {n} images")
        n_val, n_test = val_size, test_size
        n_train = n - n_val - n_test
    else:
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

    train_set = image_names[:n_train]
    val_set = image_names[n_train:n_train + n_val]
    test_set = image_names[n_train + n_val:]

    check_no_leakage(train_set, val_set, test_set)
    print(f"Split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    return train_set, val_set, test_set


def check_no_leakage(train_set, val_set, test_set) -> None:
    """Raises if any image appears in more than one split."""
    train, val, test = set(train_set), set(val_set), set(test_set)
    overlaps = {
        "train∩val": len(train & val),
        "train∩test": len(train & test),
        "val∩test": len(val & test),
    }
    bad = {k: v for k, v in overlaps.items() if v}
    if bad:
        raise AssertionError(f"Data leakage between splits: {bad}")


def save_split(
    image_to_captions: dict[str, list[str]],
    split_names: list[str],
    out_path: str,
) -> None:
    subset = {name: image_to_captions[name] for name in split_names if name in image_to_captions}
    with open(out_path, "w") as f:
        json.dump(subset, f, indent=2)
    print(f"Saved {len(subset)} entries to {out_path}")


def build_and_save_splits(
    raw_dir: str = "data/raw",
    out_dir: str = "data/processed",
    seed: int = 42,
    val_size: int = 1000,
    test_size: int = 1000,
) -> None:
    """
    End-to-end: reads Flickr8k captions.txt, splits into train/val/test
    (default 6091 / 1000 / 1000 images), and writes three JSON files to
    data/processed/.
    """
    captions_path = os.path.join(raw_dir, "captions.txt")
    if not os.path.exists(captions_path):
        raise FileNotFoundError(
            f"Could not find {captions_path}. "
            "Download Flickr8k from Kaggle and place captions.txt + Images/ "
            "inside data/raw/ before running this script."
        )

    os.makedirs(out_dir, exist_ok=True)
    image_to_captions = load_captions(captions_path)
    train_set, val_set, test_set = split_dataset(
        image_to_captions, seed=seed, val_size=val_size, test_size=test_size
    )

    save_split(image_to_captions, train_set, os.path.join(out_dir, "train_captions.json"))
    save_split(image_to_captions, val_set, os.path.join(out_dir, "val_captions.json"))
    save_split(image_to_captions, test_set, os.path.join(out_dir, "test_captions.json"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build image-level train/val/test splits.")
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--test_size", type=int, default=1000)
    a = parser.parse_args()
    build_and_save_splits(a.raw_dir, a.out_dir, a.seed, a.val_size, a.test_size)
