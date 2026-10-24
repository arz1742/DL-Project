"""
scripts/significance.py

Are the differences between models real, or noise? Paired bootstrap over the
1000 test images (Koehn 2004; Dror et al. 2018), for the comparisons the
report's claims rest on.

For each model, each image's score is averaged over its seeds; two models are
compared on the SAME images (paired). Resampling the images 10,000 times gives
a 95% confidence interval for the difference and a two-sided p-value. With
several comparisons at once, p-values are Holm-corrected.

Metrics: per-image CIDEr (always) and per-image CLIP-S (if
scripts/clipscore.py has been run).

    python scripts/significance.py
Writes results/table_H_significance.md. CPU only, ~1 min.
"""

import os
import sys
import glob
import json
import argparse

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.tokenizer import clean_caption  # noqa: E402

LABELS = {"lstm_frozen": "C1 LSTM", "gru_frozen": "C2 GRU", "hybrid_frozen": "C3 Hybrid",
          "lstm_finetuned": "C4 LSTM ft", "gru_finetuned": "C5 GRU ft", "hybrid_finetuned": "C6 Hybrid ft",
          "hybrid_vgg16": "Hybrid + VGG16", "hybrid_clip": "Hybrid + CLIP",
          "attention": "A1 Attention", "transformer": "A2 Transformer",
          "blip_base": "BLIP-base", "blip_large": "BLIP-large"}

# (question in the report, model A, model B)
COMPARISONS = [
    ("Decoder: GRU vs LSTM (frozen)", "gru_frozen", "lstm_frozen"),
    ("Decoder: hybrid vs LSTM (frozen)", "hybrid_frozen", "lstm_frozen"),
    ("Decoder: hybrid vs GRU (frozen)", "hybrid_frozen", "gru_frozen"),
    ("Fine-tuning: LSTM", "lstm_finetuned", "lstm_frozen"),
    ("Fine-tuning: GRU", "gru_finetuned", "gru_frozen"),
    ("Fine-tuning: hybrid", "hybrid_finetuned", "hybrid_frozen"),
    ("Encoder: ResNet50 vs VGG16", "hybrid_frozen", "hybrid_vgg16"),
    ("Encoder: CLIP vs ResNet50", "hybrid_clip", "hybrid_frozen"),
    ("Attention LSTM vs best classic (C6)", "attention", "hybrid_finetuned"),
    ("Transformer vs attention LSTM", "transformer", "attention"),
    ("CLIP encoder vs attention LSTM", "hybrid_clip", "attention"),
]


def per_image_cider(runs_dir: str, refs: dict) -> pd.DataFrame:
    """Per-image CIDEr for every run (beam 5; BLIP beam 3), averaged over seeds."""
    from pycocoevalcap.cider.cider import Cider

    names = sorted(refs)
    gts = {n: [" ".join(clean_caption(r).split()) for r in refs[n]] for n in names}
    cols = {}
    by_cfg = {}
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "seed*", "eval", "predictions.csv"))):
        cfg = path.split(os.sep)[-4]
        df = pd.read_csv(path, keep_default_na=False)
        dec = "beam5" if "beam5" in set(df.decoding) else sorted(set(df.decoding))[-1]
        caps = dict(zip(df[df.decoding == dec].image, df[df.decoding == dec].caption))
        if not all(n in caps for n in names):
            continue
        res = {n: [" ".join(clean_caption(caps[n]).split()) or "<empty>"] for n in names}
        _, scores = Cider().compute_score(gts, res)
        by_cfg.setdefault(cfg, []).append(np.asarray(scores, float))
    for cfg, arrs in by_cfg.items():
        cols[cfg] = np.mean(arrs, axis=0)
    return pd.DataFrame(cols, index=names)


def per_image_clip(out_dir: str) -> pd.DataFrame | None:
    path = os.path.join(out_dir, "clipscore_per_image.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df.groupby(["config", "image"])["clip_s"].mean().unstack(0)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"diff": d.mean(), "lo": lo, "hi": hi, "p": max(p, 1 / n_boot), "rel": d.mean() / max(b.mean(), 1e-9)}


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist()


def table(scores: pd.DataFrame, metric: str, n_boot: int) -> str:
    rows = []
    for question, a, b in COMPARISONS:
        if a in scores and b in scores:
            r = paired_bootstrap(scores[a].values, scores[b].values, n_boot)
            rows.append((question, a, b, r))
    if not rows:
        return f"(no {metric} comparisons available)"
    adj = holm([r["p"] for *_, r in rows])
    lines = [f"| Comparison | {metric} A | {metric} B | Difference (A − B) | 95% CI | Change | p (Holm) | Real difference? |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (question, a, b, r), p in zip(rows, adj):
        verdict = "Yes" if p < 0.05 else "No (within noise)"
        lines.append(f"| {question}: {LABELS.get(a, a)} vs {LABELS.get(b, b)} | {scores[a].mean():.3f} | "
                     f"{scores[b].mean():.3f} | {r['diff']:+.3f} | [{r['lo']:+.3f}, {r['hi']:+.3f}] | "
                     f"{100 * r['rel']:+.1f}% | {'<0.001' if p < 0.001 else f'{p:.3f}'} | {verdict} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--n_boot", type=int, default=10000)
    args = p.parse_args()

    with open(os.path.join(args.processed_dir, "test_captions.json")) as f:
        refs = json.load(f)
    parts = ["# Table H: Which differences are real? (paired bootstrap over test images, "
             f"{args.n_boot:,} resamples, scores averaged over seeds)\n",
             "## CIDEr\n", table(per_image_cider(args.runs_dir, refs), "CIDEr", args.n_boot)]
    clip = per_image_clip(args.out_dir)
    if clip is not None:
        parts += ["\n## CLIP-S (reference-free)\n", table(clip, "CLIP-S", args.n_boot)]
    parts.append("\n\"Real difference\" = Holm-corrected p < 0.05: the gap is very unlikely to come from which "
                 "test images happened to be chosen. Seed-to-seed variation is shown separately (± in Tables A, D, F).")
    text = "\n".join(parts) + "\n"
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "table_H_significance.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
