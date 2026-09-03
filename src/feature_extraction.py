"""
feature_extraction.py

Extracts ResNet50 image features for every image in the dataset.
Runs comfortably on CPU (i7, no GPU) — this is a one-time forward pass,
no backpropagation, so it does NOT need your friend's GPU.

Two modes:
  - "frozen":   standard ImageNet-pretrained ResNet50, weights untouched.
                Used for the ResNet50(frozen) + LSTM/GRU ablation configs.
  - "finetune": same base model, but returned with trainable=True so it can
                be fine-tuned end-to-end during training (that part DOES need
                the GPU, and happens inside train.py, not here).

Usage:
    python src/feature_extraction.py --mode frozen
"""

import os
import json
import argparse
import pickle

import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model


def build_resnet50_encoder(trainable: bool = False) -> Model:
    """
    Loads ResNet50 pretrained on ImageNet, strips the final classification
    layer, and returns the pooled feature vector (2048-d) — matching the
    encoder design used in both mother papers.
    """
    base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    base_model.trainable = trainable
    return base_model


def extract_features(
    image_dir: str,
    image_filenames: list[str],
    trainable: bool = False,
    target_size: tuple[int, int] = (224, 224),
) -> dict[str, np.ndarray]:
    """
    Runs every image through the ResNet50 encoder and returns
    {filename: 2048-d feature vector}.
    """
    model = build_resnet50_encoder(trainable=trainable)
    features = {}

    for filename in tqdm(image_filenames, desc="Extracting features"):
        img_path = os.path.join(image_dir, filename)
        if not os.path.exists(img_path):
            print(f"  [skip] missing file: {img_path}")
            continue

        img = keras_image.load_img(img_path, target_size=target_size)
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        feature_vector = model.predict(img_array, verbose=0)
        features[filename] = feature_vector.squeeze()

    return features


def save_features(features: dict[str, np.ndarray], out_path: str) -> None:
    with open(out_path, "wb") as f:
        pickle.dump(features, f)
    print(f"Saved {len(features)} feature vectors to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract ResNet50 image features.")
    parser.add_argument("--mode", choices=["frozen", "finetune"], default="frozen",
                         help="'frozen' for a static feature-extraction pass. "
                              "'finetune' just labels the saved features for the "
                              "fine-tuning pipeline; actual fine-tuning happens in train.py.")
    parser.add_argument("--image_dir", default="data/raw/Images")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--out_dir", default="data/features")
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional: only process the first N images (fast demo/testing mode).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_filenames = []
    for split in ["train_captions.json", "val_captions.json", "test_captions.json"]:
        split_path = os.path.join(args.processed_dir, split)
        if not os.path.exists(split_path):
            raise FileNotFoundError(
                f"{split_path} not found. Run src/data_loader.py first."
            )
        with open(split_path, "r") as f:
            all_filenames.extend(json.load(f).keys())

    if args.limit is not None:
        all_filenames = all_filenames[:args.limit]
        print(f"[demo mode] Limiting to first {args.limit} images for a fast partial run")

    print(f"Extracting features for {len(all_filenames)} images (mode={args.mode})")
    features = extract_features(
        args.image_dir, all_filenames,
        trainable=(args.mode == "finetune"),
    )

    out_path = os.path.join(args.out_dir, f"resnet50_{args.mode}_features.pkl")
    save_features(features, out_path)


if __name__ == "__main__":
    main()
