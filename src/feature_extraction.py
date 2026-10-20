"""
feature_extraction.py

Runs ResNet50 (ImageNet weights) over every image ONCE and saves the result,
so training never has to touch raw images again.

Two outputs, one per encoder mode used in the ablation study:

  --mode pooled  ->  data/features/resnet50_pooled.npy   (N, 2048)        float32
      The global-average-pooled feature vector. Used by every FROZEN-encoder
      config: ResNet50 is a fixed feature extractor.

  --mode conv4   ->  data/features/resnet50_conv4.npy    (N, 14, 14, 1024) float16
      The output of ResNet50's 4th stage (just before the last block, conv5).
      Used by every FINE-TUNED config: training runs ResNet50's last block
      (conv5_x, ~15M weights) on top of these maps and updates its weights,
      while stages 1-4 stay frozen. Pre-computing stage 4 makes real
      fine-tuning cheap enough for a Colab L4.

Both files come with an index file listing the image names in row order:
      data/features/image_index.json

Images are processed in batches (default 64), so the full Flickr8k set takes
roughly 2-4 minutes on an L4 GPU (much longer on CPU).

Usage:
    python src/feature_extraction.py --mode pooled
    python src/feature_extraction.py --mode conv4
    python src/feature_extraction.py --mode pooled --limit 200   # quick test
"""

import os
import sys
import json
import argparse

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

CONV4_LAYER = "conv4_block6_out"  # last layer of ResNet50 stage 4 -> (14, 14, 1024)


def build_resnet50_extractor(mode: str, weights: str | None = "imagenet"):
    """
    Returns a Keras model mapping a preprocessed 224x224x3 image to either the
    pooled 2048-d vector ("pooled") or the stage-4 feature map ("conv4").
    """
    import keras
    from keras.applications.resnet50 import ResNet50

    base = ResNet50(weights=weights, include_top=False, pooling="avg",
                    input_shape=(224, 224, 3))
    base.trainable = False
    if mode == "pooled":
        return base
    if mode == "conv4":
        return keras.Model(base.input, base.get_layer(CONV4_LAYER).output,
                           name="resnet50_to_conv4")
    raise ValueError(f"Unknown mode {mode!r} (expected 'pooled' or 'conv4')")


def load_and_preprocess(path: str, target_size=(224, 224)) -> np.ndarray:
    """Loads one image file and applies ResNet50's ImageNet preprocessing."""
    from keras.applications.resnet50 import preprocess_input
    from keras.utils import load_img, img_to_array

    img = load_img(path, target_size=target_size)
    return preprocess_input(img_to_array(img))


def all_split_image_names(processed_dir: str) -> list[str]:
    """Image names from train + val + test split files, in a fixed order."""
    names = []
    for split in ["train_captions.json", "val_captions.json", "test_captions.json"]:
        split_path = os.path.join(processed_dir, split)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"{split_path} not found. Run src/data_loader.py first.")
        with open(split_path) as f:
            names.extend(json.load(f).keys())
    return sorted(set(names))


def extract_features(image_dir: str, image_names: list[str], mode: str,
                     batch_size: int = 64, weights: str | None = "imagenet") -> np.ndarray:
    """Runs the extractor over all images in batches; returns one stacked array."""
    model = build_resnet50_extractor(mode, weights=weights)
    dtype = np.float16 if mode == "conv4" else np.float32
    out_shape = (len(image_names),) + tuple(model.output.shape[1:])
    out = np.zeros(out_shape, dtype=dtype)

    for start in tqdm(range(0, len(image_names), batch_size), desc=f"ResNet50 [{mode}]"):
        batch_names = image_names[start:start + batch_size]
        batch = np.stack([load_and_preprocess(os.path.join(image_dir, n)) for n in batch_names])
        out[start:start + len(batch_names)] = model.predict_on_batch(batch).astype(dtype)
    return out


def feature_paths(out_dir: str, mode: str) -> tuple[str, str]:
    return (os.path.join(out_dir, f"resnet50_{mode}.npy"),
            os.path.join(out_dir, "image_index.json"))


def load_features(out_dir: str, mode: str, mmap: bool = True) -> tuple[np.ndarray, dict[str, int]]:
    """Loads a saved feature array plus {image_name: row}. conv4 is memory-mapped."""
    arr_path, index_path = feature_paths(out_dir, mode)
    arr = np.load(arr_path, mmap_mode="r" if mmap else None)
    with open(index_path) as f:
        names = json.load(f)
    if len(names) != arr.shape[0]:
        raise ValueError(f"{index_path} lists {len(names)} images but {arr_path} has {arr.shape[0]} rows")
    return arr, {n: i for i, n in enumerate(names)}


def main():
    parser = argparse.ArgumentParser(description="Extract ResNet50 image features once.")
    parser.add_argument("--mode", choices=["pooled", "conv4"], default="pooled")
    parser.add_argument("--image_dir", default="data/raw/Images")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--out_dir", default="data/features")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N images (quick test).")
    parser.add_argument("--cnn_weights", default="imagenet",
                        help="'imagenet' (always, for real runs) or 'none' (offline tests only).")
    args = parser.parse_args()
    weights = None if args.cnn_weights == "none" else args.cnn_weights

    os.makedirs(args.out_dir, exist_ok=True)
    names = all_split_image_names(args.processed_dir)
    missing = [n for n in names if not os.path.exists(os.path.join(args.image_dir, n))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing from {args.image_dir}, e.g. {missing[:3]}")
    if args.limit:
        names = names[:args.limit]
        print(f"[test mode] only the first {args.limit} images")

    arr_path, index_path = feature_paths(args.out_dir, args.mode)
    if os.path.exists(index_path):
        with open(index_path) as f:
            if json.load(f) != names:
                raise ValueError(f"{index_path} already exists with a different image list. "
                                 "Delete data/features/ and re-extract both modes.")

    print(f"Extracting {args.mode} features for {len(names)} images")
    feats = extract_features(args.image_dir, names, args.mode, args.batch_size, weights)
    np.save(arr_path, feats)
    with open(index_path, "w") as f:
        json.dump(names, f)
    print(f"Saved {feats.shape} {feats.dtype} -> {arr_path}  ({feats.nbytes / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
