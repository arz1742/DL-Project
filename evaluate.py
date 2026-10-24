"""
evaluate.py

Scores one trained run on the 1000 test images and writes everything the
results tables, the showpiece analyses and the dashboard need:

    runs/<config>/seed<seed>/eval/metrics.json      every number, per decoding method
    runs/<config>/seed<seed>/eval/predictions.csv   one row per test image per decoding

Usage:
    python evaluate.py --config_name lstm_frozen --seed 1
    python evaluate.py --config_name hybrid_frozen --seed 1 --decodings greedy,beam1,beam3,beam5,beam10

Metrics (see src/evaluation.py and src/models/metrics.py):
  token level (teacher forcing): masked accuracy, top-5 accuracy, unmasked accuracy, loss
  caption level, per decoding:   COCO BLEU-1..4, METEOR, ROUGE-L, CIDEr; NLTK corpus BLEU;
                                 average sentence BLEU; Paper-1 protocol (max BLEU-1 over 40
                                 images); key-object recall; caption diversity
"""

import os
import sys
import csv
import json
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.tokenizer import CaptionTokenizer  # noqa: E402
from src.dataset import encode_split, CaptionBatches  # noqa: E402
from src.feature_extraction import load_features  # noqa: E402
from src.utils.run_io import run_dir, load_trained_model, write_json, read_json, feature_mode  # noqa: E402
from src.utils.beam_search import CaptionGenerator  # noqa: E402
from src import evaluation as ev  # noqa: E402,F401
from src.scoring import score_captions, print_summary, write_outputs  # noqa: E402



def parse_decoding(name: str) -> int:
    if name == "greedy":
        return 1
    if name.startswith("beam"):
        return int(name[4:])
    raise ValueError(f"Unknown decoding {name!r}; use greedy or beamK, e.g. beam5")


def main():
    p = argparse.ArgumentParser(description="Evaluate a trained captioning run.")
    p.add_argument("--config_name", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--features_dir", default="data/features")
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--decodings", default="greedy,beam3,beam5")
    p.add_argument("--limit", type=int, default=None, help="Only the first N images (quick check).")
    p.add_argument("--no_meteor", action="store_true", help="Skip METEOR (it is the slowest metric).")
    args = p.parse_args()

    rdir = run_dir(args.runs_dir, args.config_name, args.seed)
    state = read_json(os.path.join(rdir, "state.json"), {})
    if not state.get("done"):
        raise RuntimeError(f"{rdir} has not finished training.")
    out_dir = os.path.join(rdir, "eval" if args.split == "test" else "eval_val")
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = CaptionTokenizer.load(os.path.join(args.processed_dir, "tokenizer.json"))
    model, cfg = load_trained_model(rdir, tokenizer.vocab_size, args.processed_dir)
    from src.models.caption_model import compile_caption_model
    compile_caption_model(model, mask_padding_in_loss=cfg["mask_padding_in_loss"])

    with open(os.path.join(args.processed_dir, f"{args.split}_captions.json")) as f:
        refs = json.load(f)
    with open(os.path.join(args.processed_dir, "train_captions.json")) as f:
        train_caps = json.load(f)
    if args.limit:
        refs = dict(sorted(refs.items())[:args.limit])
    names = sorted(refs)

    feats, name_to_row = load_features(args.features_dir, feature_mode(cfg))
    img_feats = np.asarray(feats[[name_to_row[n] for n in names]], np.float32)

    print(f"=== Evaluating {args.config_name} seed {args.seed} on {len(names)} {args.split} images ===")
    metrics = {"config_name": args.config_name, "seed": args.seed, "split": args.split,
               "n_images": len(names), "best_epoch": state["best_epoch"] + 1,
               "train_seconds": state.get("train_seconds")}

    # ---- token-level metrics (teacher forcing, same definitions as training) ----
    rows, tok, _ = encode_split(refs, tokenizer, cfg["max_length"], name_to_row)
    token = model.evaluate(CaptionBatches(feats, rows, tok, 256, shuffle=False),
                           verbose=0, return_dict=True)
    metrics["token_metrics"] = {k: float(v) for k, v in token.items()}
    print("Token metrics:", {k: round(v, 4) for k, v in metrics["token_metrics"].items()})

    # ---- caption generation ----
    gen = CaptionGenerator(model, tokenizer, cfg["max_length"])
    states = gen.encode_images(img_feats)
    metrics["decodings"] = {}
    pred_rows = []
    for dec in args.decodings.split(","):
        k = parse_decoding(dec)
        t0 = time.time()
        caps = gen.greedy(states) if dec == "greedy" else gen.beam(states, k)
        hyps = dict(zip(names, caps))
        secs = time.time() - t0

        m, rows_for_dec = score_captions(refs, hyps, train_caps, dec, use_meteor=not args.no_meteor)
        m.update(beam_width=k, decode_seconds=round(secs, 1))
        metrics["decodings"][dec] = m
        pred_rows += rows_for_dec
        print_summary(dec, m, secs)

    write_outputs(out_dir, metrics, pred_rows)


if __name__ == "__main__":
    main()
