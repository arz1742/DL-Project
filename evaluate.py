"""
evaluate.py

Computes BLEU-1 to BLEU-4 scores for a trained model on the test set.
Inference itself is cheap (no training), so this runs fine on CPU —
you can run this yourself once your friend sends back the trained weights.

Usage:
    python evaluate.py --decoder lstm --features data/features/resnet50_frozen_features.pkl \
                        --config_name lstm_frozen --checkpoint_epoch 29
"""

import json
import pickle
import argparse

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

from src.tokenizer import CaptionTokenizer, clean_caption
from src.models.caption_model import build_caption_model
from src.utils.beam_search import beam_search_decode, greedy_decode


def compute_bleu_scores(references: list[str], hypothesis: str) -> dict:
    """
    references: list of raw reference captions for one image (5 for Flickr8k)
    hypothesis: the model's generated caption
    """
    ref_tokens = [clean_caption(r).split() for r in references]
    hyp_tokens = clean_caption(hypothesis).split()
    smoothie = SmoothingFunction().method4

    return {
        "bleu1": sentence_bleu(ref_tokens, hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie),
        "bleu2": sentence_bleu(ref_tokens, hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie),
        "bleu3": sentence_bleu(ref_tokens, hyp_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie),
        "bleu4": sentence_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained captioning model with BLEU.")
    parser.add_argument("--decoder", choices=["lstm", "gru"], required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--test_captions", default="data/processed/test_captions.json")
    parser.add_argument("--tokenizer_path", default="data/processed/tokenizer.json")
    parser.add_argument("--config_name", required=True)
    parser.add_argument("--checkpoint_epoch", type=int, required=True)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--max_length", type=int, default=34)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--units", type=int, default=256)
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--decoding", choices=["beam", "greedy"], default="beam")
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional: only evaluate on the first N test images (faster sanity check).")
    args = parser.parse_args()

    tokenizer = CaptionTokenizer.load(args.tokenizer_path)
    with open(args.features, "rb") as f:
        features = pickle.load(f)
    with open(args.test_captions, "r") as f:
        test_captions = json.load(f)

    model = build_caption_model(
        vocab_size=tokenizer.vocab_size,
        max_length=args.max_length,
        decoder_type=args.decoder,
        embedding_dim=args.embedding_dim,
        units=args.units,
    )
    weights_path = f"{args.checkpoint_dir}/{args.config_name}_epoch{args.checkpoint_epoch}.weights.h5"
    model.load_weights(weights_path)
    print(f"Loaded weights: {weights_path}")

    image_names = list(test_captions.keys())
    if args.limit:
        image_names = image_names[:args.limit]

    all_scores = {"bleu1": [], "bleu2": [], "bleu3": [], "bleu4": []}
    sample_outputs = []

    for image_name in tqdm(image_names, desc="Evaluating"):
        if image_name not in features:
            continue
        feature_vector = features[image_name]
        references = test_captions[image_name]

        if args.decoding == "beam":
            hypothesis = beam_search_decode(model, feature_vector, tokenizer, args.max_length, args.beam_width)
        else:
            hypothesis = greedy_decode(model, feature_vector, tokenizer, args.max_length)

        scores = compute_bleu_scores(references, hypothesis)
        for key in all_scores:
            all_scores[key].append(scores[key])

        sample_outputs.append({
            "image": image_name,
            "predicted_caption": hypothesis,
            "reference_captions": references,
            "bleu4": scores["bleu4"],
        })

    avg_scores = {key: sum(vals) / len(vals) for key, vals in all_scores.items()}
    print("\n=== Average BLEU scores ===")
    for key, val in avg_scores.items():
        print(f"  {key.upper()}: {val:.4f}")

    out_path = f"experiments/results/{args.config_name}_bleu_{args.decoding}.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": args.config_name,
            "decoding": args.decoding,
            "average_scores": avg_scores,
            "sample_outputs": sample_outputs[:20],  # keep a readable sample, not all
        }, f, indent=2)
    print(f"Saved evaluation results to {out_path}")


if __name__ == "__main__":
    main()
