"""
inference.py

Caption ANY image with a trained run (runs fine on a CPU).

    python inference.py --image photo.jpg --config_name hybrid_frozen --seed 1
    python inference.py --image photo.jpg --config_name hybrid_frozen --seed 1 --beam_width 1   # greedy
    python inference.py --image photo.jpg --config_name hybrid_frozen --seed 1 --show_probs
"""

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.tokenizer import CaptionTokenizer  # noqa: E402
from src.utils.run_io import run_dir, load_trained_model  # noqa: E402
from src.utils.beam_search import CaptionGenerator  # noqa: E402
from src.feature_extraction import build_resnet50_extractor, load_and_preprocess  # noqa: E402


class Captioner:
    """Loads a trained run once; captions any number of images."""

    def __init__(self, rdir: str, processed_dir: str = "data/processed"):
        self.tokenizer = CaptionTokenizer.load(os.path.join(processed_dir, "tokenizer.json"))
        self.model, self.cfg = load_trained_model(rdir, self.tokenizer.vocab_size, processed_dir)
        mode = "pooled" if self.cfg["encoder_mode"] == "frozen" else "conv4"
        self.extractor = build_resnet50_extractor(mode)
        self.gen = CaptionGenerator(self.model, self.tokenizer, self.cfg["max_length"])

    def state(self, image_path: str) -> np.ndarray:
        feat = self.extractor.predict_on_batch(load_and_preprocess(image_path)[None])
        return self.gen.encode_images(feat)[0]

    def caption(self, image_path: str, beam_width: int = 5) -> str:
        s = self.state(image_path)[None]
        return self.gen.greedy(s)[0] if beam_width == 1 else self.gen.beam(s, beam_width)[0]


def main():
    p = argparse.ArgumentParser(description="Generate a caption for one image.")
    p.add_argument("--image", required=True)
    p.add_argument("--config_name", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--beam_width", type=int, default=5)
    p.add_argument("--show_probs", action="store_true", help="Top-5 candidate words at each step.")
    args = p.parse_args()

    c = Captioner(run_dir(args.runs_dir, args.config_name, args.seed), args.processed_dir)
    cap = c.caption(args.image, args.beam_width)
    method = "greedy" if args.beam_width == 1 else f"beam k={args.beam_width}"
    print(f'\nCaption ({method}): "{cap}"')
    if args.show_probs:
        for step in c.gen.word_probabilities(c.state(args.image), cap):
            alts = ", ".join(f"{w} {p:.2f}" for w, p in step["top"])
            print(f"  {step['word']:>12s}  p={step['prob']:.2f}   top-5: {alts}")


if __name__ == "__main__":
    main()
