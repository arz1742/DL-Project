"""
build_tokenizer.py

Builds the vocabulary from the training split's captions and saves it,
so train.py / evaluate.py / inference.py can all load the same tokenizer.
Run this once, right after data_loader.py and before feature_extraction.py
(order between these two doesn't actually matter, but do both before training).

Usage:
    python build_tokenizer.py
"""

import json
import argparse

from src.tokenizer import CaptionTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_captions", default="data/processed/train_captions.json")
    parser.add_argument("--out_path", default="data/processed/tokenizer.json")
    parser.add_argument("--min_word_freq", type=int, default=5)
    args = parser.parse_args()

    with open(args.train_captions, "r") as f:
        train_captions = json.load(f)

    all_captions = [cap for caps in train_captions.values() for cap in caps]

    tokenizer = CaptionTokenizer(min_word_freq=args.min_word_freq)
    tokenizer.build_vocab(all_captions)
    tokenizer.save(args.out_path)


if __name__ == "__main__":
    main()
