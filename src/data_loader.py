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
) -> tuple[list[str], list[str], list[str]]:
    """
    Splits image filenames into train/val/test sets.
    Ratios default to roughly 85/7.5/7.5, adjust as needed for your report.
    """
    image_names = sorted(image_to_captions.keys())
    random.Random(seed).shuffle(image_names)

    n = len(image_names)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_set = image_names[:n_train]
    val_set = image_names[n_train:n_train + n_val]
    test_set = image_names[n_train + n_val:]

    print(f"Split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    return train_set, val_set, test_set


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
) -> None:
    """
    End-to-end: reads Flickr8k captions.txt, splits into train/val/test,
    and writes three JSON files to data/processed/.
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
    train_set, val_set, test_set = split_dataset(image_to_captions, seed=seed)

    save_split(image_to_captions, train_set, os.path.join(out_dir, "train_captions.json"))
    save_split(image_to_captions, val_set, os.path.join(out_dir, "val_captions.json"))
    save_split(image_to_captions, test_set, os.path.join(out_dir, "test_captions.json"))


if __name__ == "__main__":
    build_and_save_splits()
