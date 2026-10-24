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

  --mode grid    ->  data/features/resnet50_grid.npy     (N, 7, 7, 2048)   float16
      ResNet50's final 7x7 feature map: 49 image regions of 2048 values each.
      Used by the ATTENTION models (Part 2), which look at a different region
      for each word they generate. ~1.6 GB.

  --mode vgg16   ->  data/features/vgg16_pooled.npy      (N, 512)          float32
      VGG16 (2014) pooled features, for the encoder comparison.

  --mode clip    ->  data/features/clip_vitb32.npy       (N, 512)          float32
      CLIP ViT-B/32 image embeddings (OpenAI 2021; trained on 400M image-text
      pairs). Needs PyTorch + transformers (preinstalled on Colab).

All files come with an index file listing the image names in row order:
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
CONV5_LAYER = "conv5_block3_out"  # last layer of ResNet50 -> (7, 7, 2048) region grid
MODES = ("pooled", "conv4", "grid", "vgg16", "clip")
FEATURE_FILES = {"pooled": "resnet50_pooled.npy", "conv4": "resnet50_conv4.npy", "grid": "resnet50_grid.npy",
                 "vgg16": "vgg16_pooled.npy", "clip": "clip_vitb32.npy"}
FEATURE_DIMS = {"pooled": 2048, "vgg16": 512, "clip": 512}   # vector modes used by frozen configs
CLIP_MODEL = "openai/clip-vit-base-patch32"


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
    if mode == "grid":
        return keras.Model(base.input, base.get_layer(CONV5_LAYER).output,
                           name="resnet50_to_grid")
    raise ValueError(f"Unknown mode {mode!r} (expected one of {MODES})")


def load_and_preprocess(path: str, target_size=(224, 224), model: str = "resnet50") -> np.ndarray:
    """Loads one image file and applies the CNN's ImageNet preprocessing."""
    from keras.utils import load_img, img_to_array
    if model == "vgg16":
        from keras.applications.vgg16 import preprocess_input
    else:
        from keras.applications.resnet50 import preprocess_input

    img = load_img(path, target_size=target_size)
    return preprocess_input(img_to_array(img))


class ClipFeaturizer:
    """CLIP ViT-B/32 image embeddings (512-d) for a list of image paths."""

    def __init__(self, model_id: str = CLIP_MODEL):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def __call__(self, paths: list[str]) -> np.ndarray:
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            feats = self.model.get_image_features(**inputs)
        return feats.float().cpu().numpy()


class ImageFeaturizer:
    """Image paths -> feature array for any mode; used for extraction and for inference."""

    def __init__(self, mode: str, weights: str | None = "imagenet"):
        self.mode = mode
        if mode == "clip":
            self.clip = ClipFeaturizer()
        elif mode == "vgg16":
            from keras.applications.vgg16 import VGG16
            self.cnn = VGG16(weights=weights, include_top=False, pooling="avg", input_shape=(224, 224, 3))
        else:
            self.cnn = build_resnet50_extractor(mode, weights=weights)

    def __call__(self, paths: list[str]) -> np.ndarray:
        if self.mode == "clip":
            return self.clip(paths)
        model = "vgg16" if self.mode == "vgg16" else "resnet50"
        batch = np.stack([load_and_preprocess(p, model=model) for p in paths])
        return np.asarray(self.cnn.predict_on_batch(batch))


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
    featurize = ImageFeaturizer(mode, weights)
    dtype = np.float16 if mode in ("conv4", "grid") else np.float32
    out = None
    for start in tqdm(range(0, len(image_names), batch_size), desc=f"features [{mode}]"):
        batch_names = image_names[start:start + batch_size]
        f = featurize([os.path.join(image_dir, n) for n in batch_names]).astype(dtype)
        if out is None:
            out = np.zeros((len(image_names),) + f.shape[1:], dtype=dtype)
        out[start:start + len(batch_names)] = f
    return out


def feature_paths(out_dir: str, mode: str) -> tuple[str, str]:
    return (os.path.join(out_dir, FEATURE_FILES[mode]),
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
    parser.add_argument("--mode", choices=list(MODES), default="pooled")
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
