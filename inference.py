"""
inference.py

Generate a caption for ANY single image (not just dataset images) using a
trained model. This is the "give it any image, get a caption back" step —
cheap enough to run on your own CPU once you have the trained weights.

Usage:
    python inference.py --image path/to/your_photo.jpg \
                         --decoder lstm --config_name lstm_frozen --checkpoint_epoch 29
"""

import argparse
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

from src.tokenizer import CaptionTokenizer
from src.models.caption_model import build_caption_model
from src.utils.beam_search import beam_search_decode, greedy_decode


def extract_single_image_feature(image_path: str, target_size=(224, 224)) -> np.ndarray:
    encoder = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    img = keras_image.load_img(image_path, target_size=target_size)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return encoder.predict(img_array, verbose=0).squeeze()


def main():
    parser = argparse.ArgumentParser(description="Generate a caption for a single image.")
    parser.add_argument("--image", required=True, help="Path to any image file (.jpg/.png)")
    parser.add_argument("--decoder", choices=["lstm", "gru"], required=True)
    parser.add_argument("--config_name", required=True)
    parser.add_argument("--checkpoint_epoch", type=int, required=True)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--tokenizer_path", default="data/processed/tokenizer.json")
    parser.add_argument("--max_length", type=int, default=34)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--units", type=int, default=256)
    parser.add_argument("--decoding", choices=["beam", "greedy"], default="beam")
    parser.add_argument("--beam_width", type=int, default=5)
    args = parser.parse_args()

    tokenizer = CaptionTokenizer.load(args.tokenizer_path)

    model = build_caption_model(
        vocab_size=tokenizer.vocab_size,
        max_length=args.max_length,
        decoder_type=args.decoder,
        embedding_dim=args.embedding_dim,
        units=args.units,
    )
    weights_path = f"{args.checkpoint_dir}/{args.config_name}_epoch{args.checkpoint_epoch}.weights.h5"
    model.load_weights(weights_path)

    print(f"Extracting features for {args.image}...")
    feature_vector = extract_single_image_feature(args.image)

    if args.decoding == "beam":
        caption = beam_search_decode(model, feature_vector, tokenizer, args.max_length, args.beam_width)
    else:
        caption = greedy_decode(model, feature_vector, tokenizer, args.max_length)

    print(f"\nGenerated caption ({args.decoding} search): \"{caption}\"")


if __name__ == "__main__":
    main()
