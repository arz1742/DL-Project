"""
tests/test_data_loader.py

Tests the captions.txt parsing and train/val/test splitting logic using
small in-memory fixtures — doesn't require the actual Flickr8k dataset.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tempfile
from src.data_loader import load_captions, split_dataset


def _write_temp_captions_file(rows: list[str]) -> str:
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w") as f:
        f.write("image,caption\n")
        for row in rows:
            f.write(row + "\n")
    return path


def test_load_captions_groups_by_image():
    rows = [
        "dog1.jpg,A dog runs in the park",
        "dog1.jpg,A brown dog running outside",
        "cat1.jpg,A cat sleeps on the sofa",
    ]
    path = _write_temp_captions_file(rows)
    result = load_captions(path)
    os.remove(path)

    assert len(result) == 2
    assert len(result["dog1.jpg"]) == 2
    assert len(result["cat1.jpg"]) == 1


def test_split_dataset_produces_disjoint_sets():
    image_to_captions = {f"img{i}.jpg": [f"caption for image {i}"] for i in range(100)}
    train, val, test = split_dataset(image_to_captions, train_ratio=0.8, val_ratio=0.1, seed=1)

    assert len(train) + len(val) + len(test) == 100
    assert set(train).isdisjoint(set(val))
    assert set(train).isdisjoint(set(test))
    assert set(val).isdisjoint(set(test))


def test_split_dataset_is_reproducible_with_seed():
    image_to_captions = {f"img{i}.jpg": [f"caption {i}"] for i in range(50)}
    split_a = split_dataset(image_to_captions, seed=7)
    split_b = split_dataset(image_to_captions, seed=7)
    assert split_a == split_b


if __name__ == "__main__":
    test_load_captions_groups_by_image()
    test_split_dataset_produces_disjoint_sets()
    test_split_dataset_is_reproducible_with_seed()
    print("All data_loader tests passed.")
