"""
scripts/aggregate_results.py

Collects every evaluated run in runs/ into the three report tables and one
JSON file for the dashboard.

    python scripts/aggregate_results.py

Writes to results/:
    all_runs.csv            one row per (config, seed, decoding)
    table_A_ablation.md     core configs, beam 5, mean ± std over seeds
    table_B_accuracy.md     the accuracy illusion: Paper 1 vs our runs
    table_C_decoding.md     greedy vs beam, BLEU-1 (full test set and Paper-1 protocol)
    summary.json            everything above, for the dashboard
"""

import os
import sys
import json
import glob
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

CORE = ["lstm_frozen", "gru_frozen", "hybrid_frozen",
        "lstm_finetuned", "gru_finetuned", "hybrid_finetuned"]
LABELS = {
    "lstm_frozen": "C1 LSTM, frozen", "gru_frozen": "C2 GRU, frozen",
    "hybrid_frozen": "C3 Hybrid LSTM→GRU, frozen", "lstm_finetuned": "C4 LSTM, fine-tuned",
    "gru_finetuned": "C5 GRU, fine-tuned", "hybrid_finetuned": "C6 Hybrid LSTM→GRU, fine-tuned",
    "paper_protocol": "Paper-protocol replication",
    "attention": "A1 LSTM + soft attention", "transformer": "A2 Transformer decoder",
    "hybrid_vgg16": "Hybrid + VGG16 (2014)", "hybrid_clip": "Hybrid + CLIP ViT-B/32 (2021)",
    "blip_base": "BLIP-base (pretrained, 2022)", "blip_large": "BLIP-large (pretrained, 2022)",
}
# Kavitha & Karpagam (2025), Tables 2 and 3
PAPER1 = {
    "accuracy": {"lstm": 0.8459, "gru": 0.8594, "hybrid": 0.8932},
    "bleu1_greedy": {"lstm": 0.2698, "gru": 0.3986, "hybrid": 0.5142},
    "bleu1_beam3": {"lstm": 0.4286, "gru": 0.4935, "hybrid": 0.6034},
}


def collect(runs_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "seed*", "eval", "metrics.json"))):
        with open(path) as f:
            m = json.load(f)
        tok = m.get("token_metrics") or {"masked_acc": None, "top5_acc": None,
                                         "unmasked_acc": None, "loss": None}  # pretrained models have none
        for dec, d in m["decodings"].items():
            row = {"config": m["config_name"], "seed": m["seed"], "decoding": dec,
                   "beam_width": d["beam_width"], "best_epoch": m["best_epoch"],
                   "train_seconds": m.get("train_seconds"),
                   "masked_acc": tok["masked_acc"], "top5_acc": tok["top5_acc"],
                   "unmasked_acc": tok["unmasked_acc"], "test_loss": tok["loss"],
                   "key_object_recall": d["key_object"]["recall"],
                   "paper_corpus_bleu1": d["paper_protocol"].get("corpus_bleu1"),
                   "paper_max_bleu1": d["paper_protocol"]["max_bleu1"],
                   "paper_mean_bleu1": d["paper_protocol"]["mean_bleu1"],
                   "unique_caption_pct": d["diversity"]["unique_caption_pct"],
                   "vocab_used": d["diversity"]["vocab_used"],
                   "avg_length": d["diversity"]["avg_length"]}
            row.update({k: v for k, v in d["coco"].items() if isinstance(v, (int, float))})
            row.update({f"nltk_{k}": v for k, v in d["nltk_corpus"].items()})
            rows.append(row)
    return pd.DataFrame(rows)


def mean_std(df: pd.DataFrame, col: str) -> str:
    v = df[col].dropna()
    if v.empty:
        return "—"
    if col.endswith("_pct"):
        return f"{v.mean():.1f}% ± {v.std(ddof=1):.1f}" if len(v) > 1 else f"{v.mean():.1f}%"
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}"


def table_a(df: pd.DataFrame, decoding: str) -> str:
    cols = ["bleu1", "bleu4", "meteor", "rouge_l", "cider", "masked_acc", "key_object_recall"]
    head = "| Config | Seeds | " + " | ".join(c.replace("_", " ") for c in cols) + " |"
    lines = [head, "|" + " --- |" * (len(cols) + 2)]
    for cfg in CORE:
        sub = df[(df.config == cfg) & (df.decoding == decoding)]
        if sub.empty:
            continue
        lines.append(f"| {LABELS[cfg]} | {sub.seed.nunique()} | "
                     + " | ".join(mean_std(sub, c) for c in cols) + " |")
    return "\n".join(lines)


PART2 = ["attention", "transformer", "blip_base", "blip_large"]


def table_d(df: pd.DataFrame, decoding: str) -> str:
    """Part 2: best classic model vs attention models vs a pretrained ceiling."""
    cols = ["bleu1", "bleu4", "meteor", "cider", "key_object_recall", "unique_caption_pct"]
    best = best_core_config(df, decoding)
    head = "| Model | Seeds | " + " | ".join(c.replace("_", " ") for c in cols) + " |"
    lines = [head, "|" + " --- |" * (len(cols) + 2)]
    for cfg in ([best] if best else []) + PART2:
        dec = decoding if cfg not in ("blip_base", "blip_large") else "beam3"
        sub = df[(df.config == cfg) & (df.decoding == dec)]
        if sub.empty:
            continue
        lines.append(f"| {LABELS.get(cfg, cfg)} | {sub.seed.nunique()} | "
                     + " | ".join(mean_std(sub, c) for c in cols) + " |")
    lines.append("\nBLIP was pretrained on ~129M web images with COCO captions and is shown only as a "
                 "ceiling; it is not a like-for-like comparison.")
    return "\n".join(lines)


def table_f(df: pd.DataFrame, decoding: str) -> str:
    """Encoder comparison: same hybrid decoder, frozen encoder, different image features."""
    cols = ["bleu1", "bleu4", "meteor", "cider", "key_object_recall"]
    rows = [("hybrid_vgg16", "VGG16", "2014"), ("hybrid_frozen", "ResNet50", "2015"),
            ("hybrid_clip", "CLIP ViT-B/32", "2021")]
    lines = ["| Encoder | Year | Seeds | " + " | ".join(c.replace("_", " ") for c in cols) + " |",
             "|" + " --- |" * (len(cols) + 3)]
    for cfg, name, year in rows:
        sub = df[(df.config == cfg) & (df.decoding == decoding)]
        if not sub.empty:
            lines.append(f"| {name} | {year} | {sub.seed.nunique()} | "
                         + " | ".join(mean_std(sub, c) for c in cols) + " |")
    dec_range = df[df.config.isin(["lstm_frozen", "gru_frozen", "hybrid_frozen"]) & (df.decoding == decoding)]
    if not dec_range.empty:
        m = dec_range.groupby("config")["cider"].mean()
        lines.append(f"\nFor comparison, changing the DECODER (LSTM / GRU / hybrid, ResNet50 frozen) moves "
                     f"CIDEr by {m.max() - m.min():.3f}.")
    return "\n".join(lines)


def best_core_config(df: pd.DataFrame, decoding: str) -> str | None:
    sub = df[df.config.isin(CORE) & (df.decoding == decoding)]
    if sub.empty:
        return None
    return sub.groupby("config")["cider"].mean().idxmax()


def table_b(df: pd.DataFrame, decoding: str) -> str:
    best = best_core_config(df, decoding)
    pp = df[(df.config == "paper_protocol") & (df.decoding == decoding)]
    bs = df[(df.config == best) & (df.decoding == decoding)] if best else pd.DataFrame()

    def val(sub, col):
        return mean_std(sub, col) if not sub.empty else "not run yet"

    rows = [
        ("Token accuracy, padding counted", "0.8932", "unmasked_acc"),
        ("Token accuracy, padding ignored (honest)", "not reported", "masked_acc"),
        ("Top-5 next-word accuracy", "not reported", "top5_acc"),
        ("Key-object recall", "not reported", "key_object_recall"),
        ("BLEU-1, full test set", "not reported", "bleu1"),
        ("BLEU-1 on a 40-image sample (Paper 1 protocol, reading A)", "0.6034", "paper_corpus_bleu1"),
        ("Best single-image BLEU-1 among 40 (Paper 1 protocol, reading B)", "0.6034", "paper_max_bleu1"),
    ]
    lines = [f"| Metric | Paper 1 (hybrid) | Our paper-protocol run | Our best model ({LABELS.get(best, '—')}) |",
             "| --- | --- | --- | --- |"]
    lines += [f"| {name} | {p1} | {val(pp, col)} | {val(bs, col)} |" for name, p1, col in rows]
    return "\n".join(lines)


def table_c(df: pd.DataFrame) -> str:
    lines = ["| Decoder | Paper 1 greedy* | Paper 1 beam k=3* | Ours greedy (full test) | "
             "Ours beam k=3 (full test) | Ours greedy (40-image sample*) | Ours beam k=3 (40-image sample*) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for dec in ["lstm", "gru", "hybrid"]:
        cfg = f"{dec}_frozen"
        g = df[(df.config == cfg) & (df.decoding == "greedy")]
        b = df[(df.config == cfg) & (df.decoding == "beam3")]
        lines.append(f"| {dec.upper() if dec != 'hybrid' else 'Hybrid LSTM→GRU'} | "
                     f"{PAPER1['bleu1_greedy'][dec]} | {PAPER1['bleu1_beam3'][dec]} | "
                     f"{mean_std(g, 'bleu1')} | {mean_std(b, 'bleu1')} | "
                     f"{mean_std(g, 'paper_corpus_bleu1')} | {mean_std(b, 'paper_corpus_bleu1')} |")
    lines.append("\n*Paper 1 reports \"maximum BLEU-1\" on 40 example images (its Table 3) without "
                 "defining it; the sample columns are BLEU-1 over the first 40 test images. "
                 "Full-test columns are BLEU-1 over all test images.")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--decoding", default="beam5", help="Decoding used for Tables A and B.")
    args = p.parse_args()

    df = collect(args.runs_dir)
    if df.empty:
        print(f"No evaluated runs found under {args.runs_dir}/. Run evaluate.py first.")
        return
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "all_runs.csv"), index=False)

    tables = {
        "table_A_ablation.md": f"# Table A: Ablation (test set, {args.decoding}, mean ± std over seeds)\n\n"
                               + table_a(df, args.decoding),
        "table_B_accuracy.md": f"# Table B: The accuracy illusion ({args.decoding})\n\n" + table_b(df, args.decoding),
        "table_C_decoding.md": "# Table C: Greedy vs beam search (BLEU-1, frozen encoders)\n\n" + table_c(df),
        "table_F_encoders.md": f"# Table F: Encoder comparison, hybrid decoder ({args.decoding})\n\n"
                               + table_f(df, args.decoding),
        "table_D_attention.md": f"# Table D: Part 2, attention and pretrained models ({args.decoding})\n\n"
                                + table_d(df, args.decoding),
    }
    for name, text in tables.items():
        with open(os.path.join(args.out_dir, name), "w") as f:
            f.write(text + "\n")
        print(text + "\n")

    summary = {
        "decoding_for_tables": args.decoding,
        "best_core_config": best_core_config(df, args.decoding),
        "paper1": PAPER1,
        "labels": LABELS,
        "runs": json.loads(df.to_json(orient="records")),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {len(df)} rows and {len(tables)} tables to {args.out_dir}/")


if __name__ == "__main__":
    main()
