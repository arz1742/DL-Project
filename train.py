"""
train.py

THIS is the script that needs a GPU. Everything else in this repo can be
run on a CPU-only machine — hand this file + the precomputed feature file
+ processed captions to whoever has GPU access, and have them run:

    python train.py --decoder lstm --features data/features/resnet50_frozen_features.pkl \
                     --epochs 30 --config_name lstm_frozen

Repeat for each of the four ablation configs:
    lstm + frozen features
    lstm + finetuned features
    gru  + frozen features
    gru  + finetuned features

Supports pause/resume automatically via CheckpointManager — safe to stop
and restart the script at any time without losing more than the current
epoch's progress.
"""

import os
import json
import pickle
import argparse

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.tokenizer import CaptionTokenizer
from src.models.caption_model import build_caption_model
from src.utils.checkpoint_manager import CheckpointManager


def load_data(features_path: str, captions_path: str):
    with open(features_path, "rb") as f:
        features = pickle.load(f)
    with open(captions_path, "r") as f:
        captions = json.load(f)
    return features, captions


def build_training_arrays(features, captions, tokenizer, max_length):
    """
    Expands (image, [5 captions]) pairs into individual
    (feature, input_sequence, target_word) training examples using the
    standard teacher-forcing setup for caption generation.
    """
    X_features, X_sequences, y_targets = [], [], []

    for image_name, caption_list in captions.items():
        if image_name not in features:
            continue
        feature_vector = features[image_name]

        for caption in caption_list:
            encoded = tokenizer.encode(caption, max_length)
            for t in range(1, len(encoded)):
                if encoded[t] == tokenizer.word2idx["<pad>"] and encoded[t - 1] == tokenizer.word2idx["<pad>"]:
                    continue  # skip once we're past the real sequence + one end token
                X_features.append(feature_vector)
                X_sequences.append(encoded[:t] + [tokenizer.word2idx["<pad>"]] * (max_length - t))
                y_targets.append(encoded[t])

    return (
        np.array(X_features, dtype=np.float32),
        np.array(X_sequences, dtype=np.int32),
        np.array(y_targets, dtype=np.int32),
    )


def main():
    parser = argparse.ArgumentParser(description="Train an image captioning model.")
    parser.add_argument("--decoder", choices=["lstm", "gru"], required=True)
    parser.add_argument("--features", required=True,
                         help="Path to precomputed feature .pkl "
                              "(resnet50_frozen_features.pkl or resnet50_finetune_features.pkl)")
    parser.add_argument("--train_captions", default="data/processed/train_captions.json")
    parser.add_argument("--val_captions", default="data/processed/val_captions.json")
    parser.add_argument("--tokenizer_path", default="data/processed/tokenizer.json")
    parser.add_argument("--config_name", required=True,
                         help="Label for this run, e.g. 'lstm_frozen' — used for checkpoint filenames.")
    parser.add_argument("--max_length", type=int, default=34)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--units", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    args = parser.parse_args()

    print(f"=== Training config: {args.config_name} ===")

    # --- Load data ---
    train_features, train_captions = load_data(args.features, args.train_captions)
    val_features, val_captions = load_data(args.features, args.val_captions)
    tokenizer = CaptionTokenizer.load(args.tokenizer_path)

    print("Building training arrays (this may take a minute)...")
    X_feat_train, X_seq_train, y_train = build_training_arrays(
        train_features, train_captions, tokenizer, args.max_length
    )
    X_feat_val, X_seq_val, y_val = build_training_arrays(
        val_features, val_captions, tokenizer, args.max_length
    )
    print(f"Train examples: {len(y_train)} | Val examples: {len(y_val)}")

    # --- Build model ---
    model = build_caption_model(
        vocab_size=tokenizer.vocab_size,
        max_length=args.max_length,
        decoder_type=args.decoder,
        embedding_dim=args.embedding_dim,
        units=args.units,
        learning_rate=args.learning_rate,
    )
    model.summary()

    # --- Resume if a checkpoint exists ---
    ckpt_mgr = CheckpointManager(args.checkpoint_dir, args.config_name)
    start_epoch, history = ckpt_mgr.try_resume(model)

    if start_epoch >= args.epochs:
        print(f"Already trained {start_epoch} epochs (target {args.epochs}). Nothing to do.")
        return

    # --- Train, one epoch at a time, checkpointing after each ---
    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        result = model.fit(
            [X_feat_train, X_seq_train], y_train,
            validation_data=([X_feat_val, X_seq_val], y_val),
            batch_size=args.batch_size,
            epochs=1,
            verbose=1,
        )
        for key in ["loss", "accuracy", "val_loss", "val_accuracy"]:
            history.setdefault(key, []).append(float(result.history[key][0]))

        ckpt_mgr.save(model, epoch, history)

    # --- Save final metrics summary for the ablation comparison ---
    results_path = os.path.join("experiments", "results", f"{args.config_name}_metrics.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Metrics saved to {results_path}")
    print("Send back the checkpoints/ folder and experiments/results/ file "
          "so evaluation and BLEU scoring can be run.")


if __name__ == "__main__":
    main()
