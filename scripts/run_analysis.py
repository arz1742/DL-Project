"""
scripts/run_analysis.py

Runs the showpiece analyses on finished runs and writes everything to
results/analysis/ (figures for the report, JSON/CSV for the dashboard).

    python scripts/run_analysis.py --parts errors,tsne,heatmaps,beamtree --image_dir /content/data_local/raw/Images

Parts (all run on CPU; heatmaps and beamtree are faster on GPU):
  errors     Table E: hallucination rate (CHAIR) + error types for every model;
             failure gallery of the best classic model
  tsne       2-D map of the 1000 test images' ResNet50 features, coloured by subject
  heatmaps   per-word heatmaps for sample images: attention maps (A1, A2) and
             Grad-CAM (best frozen classic model)
  beamtree   every beam at every step for sample images (beam width 3)
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src import analysis as an  # noqa: E402
from src.evaluation import main_subject  # noqa: E402

CLASSIC = ["lstm_frozen", "gru_frozen", "hybrid_frozen", "lstm_finetuned", "gru_finetuned", "hybrid_finetuned"]
PART2 = ["attention", "transformer", "blip_base", "blip_large"]
LABELS = {"lstm_frozen": "C1 LSTM, frozen", "gru_frozen": "C2 GRU, frozen", "hybrid_frozen": "C3 Hybrid, frozen",
          "lstm_finetuned": "C4 LSTM, fine-tuned", "gru_finetuned": "C5 GRU, fine-tuned",
          "hybrid_finetuned": "C6 Hybrid, fine-tuned", "paper_protocol": "Paper protocol",
          "attention": "A1 LSTM + attention", "transformer": "A2 Transformer",
          "blip_base": "BLIP-base (pretrained)", "blip_large": "BLIP-large (pretrained)"}
# Fixed sample of test images for heatmaps and beam trees (the first N by name
# with a clear main subject), so every model is shown on the same photos.
N_SAMPLES = 8


def load_preds(runs_dir: str, config: str, seed: int, decoding: str) -> pd.DataFrame | None:
    path = os.path.join(runs_dir, config, f"seed{seed}", "eval", "predictions.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, keep_default_na=False)
    if decoding not in set(df.decoding):
        decoding = sorted(set(df.decoding))[-1]
    return df[df.decoding == decoding]


def best_classic(runs_dir: str, out_dir: str) -> str:
    summary = os.path.join(os.path.dirname(out_dir), "summary.json")
    if os.path.exists(summary):
        with open(summary) as f:
            best = json.load(f).get("best_core_config")
        if best:
            return best
    return "hybrid_finetuned"


def sample_images(refs: dict, n: int) -> list[str]:
    clear = [k for k in sorted(refs) if main_subject(refs[k])]
    return clear[:n]


# --------------------------------------------------------------------------- errors
def part_errors(args, refs, out_dir):
    subjects = {k: main_subject(v) for k, v in refs.items()}
    rows, per_image = [], {}
    for cfg in CLASSIC + ["paper_protocol"] + PART2:
        seeds = [0] if cfg.startswith("blip") else [1, 2, 3]
        for seed in seeds:
            preds = load_preds(args.runs_dir, cfg, seed, "beam5" if not cfg.startswith("blip") else "beam3")
            if preds is None:
                continue
            caps = dict(zip(preds.image, preds.caption))
            h = an.hallucination_scores(caps, refs)
            t = an.error_taxonomy(caps, refs, subjects)
            rows.append({"config": cfg, "seed": seed, "chair_i": h["chair_i"], "chair_s": h["chair_s"],
                         "objects_per_caption": h["objects_per_caption"], **t["rates"],
                         "top_hallucinated": ", ".join(f"{o} ({k})" for o, k in h["top_hallucinated"][:5])})
            if seed in (0, 1):
                per_image[cfg] = {"captions": caps, "labels": t["per_image"],
                                  "bleu4": dict(zip(preds.image, preds.bleu4))}
    if not rows:
        print("[errors] no evaluated runs found")
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "errors_all_runs.csv"), index=False)

    cols = ["chair_s", "chair_i"] + an.ERROR_TYPES + ["no_detected_error"]
    lines = ["| Model | Seeds | " + " | ".join(c.replace("_", " ") for c in cols) + " | Most hallucinated (seed 1) |",
             "|" + " --- |" * (len(cols) + 3)]
    for cfg, sub in df.groupby("config", sort=False):
        cells = []
        for c in cols:
            v = sub[c]
            cells.append(f"{100 * v.mean():.1f}%" + (f" ± {100 * v.std(ddof=1):.1f}" if len(v) > 1 else ""))
        lines.append(f"| {LABELS.get(cfg, cfg)} | {len(sub)} | " + " | ".join(cells)
                     + f" | {sub.iloc[0].top_hallucinated} |")
    table = ("# Table E: Hallucination and error types (test set, beam 5; BLIP beam 3)\n\n"
             + "\n".join(lines)
             + "\n\nchair s = captions naming at least one object no reference mentions; chair i = share of "
               "object mentions that are made up. Error types are detected automatically and can overlap; "
               "check the failure gallery by hand before quoting them.\n")
    with open(os.path.join(out_dir, "table_E_errors.md"), "w") as f:
        f.write(table)
    print(table)

    # failure gallery: 12 lowest-BLEU-4 captions of the best classic model, with labels
    best = best_classic(args.runs_dir, out_dir)
    if best in per_image:
        d = per_image[best]
        worst = sorted(d["bleu4"], key=d["bleu4"].get)[:12]
        gallery = [{"image": k, "caption": d["captions"][k], "bleu4": d["bleu4"][k],
                    "labels": d["labels"][k], "references": refs[k]} for k in worst]
        with open(os.path.join(out_dir, "failure_gallery.json"), "w") as f:
            json.dump({"config": best, "items": gallery}, f, indent=2)
        if args.image_dir:
            draw_gallery(gallery, args.image_dir, os.path.join(out_dir, "failure_gallery.png"), best)
    with open(os.path.join(out_dir, "errors_per_image.json"), "w") as f:
        json.dump({k: v["labels"] for k, v in per_image.items()}, f)


def draw_gallery(items, image_dir, path, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap
    from PIL import Image

    fig, axes = plt.subplots(3, 4, figsize=(16, 13))
    for ax, it in zip(axes.flat, items):
        p = os.path.join(image_dir, it["image"])
        if os.path.exists(p):
            ax.imshow(Image.open(p).convert("RGB"))
        labels = ", ".join(it["labels"]) or "no rule fired"
        ax.set_title(textwrap.fill(f"\"{it['caption']}\"", 34) + f"\n[{labels}]", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Failure gallery: 12 lowest BLEU-4 captions, {LABELS.get(config, config)}", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- t-SNE
def part_tsne(args, refs, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.feature_extraction import load_features

    feats, idx = load_features(args.features_dir, "pooled", mmap=False)
    names = [n for n in sorted(refs) if n in idx]
    xy = an.tsne_coordinates(feats[[idx[n] for n in names]])
    subj = [main_subject(refs[n]) or "other" for n in names]
    df = pd.DataFrame({"image": names, "x": xy[:, 0], "y": xy[:, 1], "subject": subj})
    df.to_csv(os.path.join(out_dir, "tsne_test_images.csv"), index=False)

    top = [s for s, _ in df[df.subject != "other"].subject.value_counts().head(8).items()]
    fig, ax = plt.subplots(figsize=(9, 8))
    rest = df[~df.subject.isin(top)]
    ax.scatter(rest.x, rest.y, s=8, c="#c9c9c9", label="other / unclear")
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(top):
        sub = df[df.subject == s]
        ax.scatter(sub.x, sub.y, s=12, color=cmap(i), label=f"{s} ({len(sub)})")
    ax.legend(markerscale=2, fontsize=10, loc="best")
    ax.set_title("t-SNE of ResNet50 features, 1000 test images (colour = main subject in references)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tsne_test_images.png"), dpi=130)
    plt.close(fig)
    print(f"[tsne] {len(df)} images -> tsne_test_images.png / .csv")


# --------------------------------------------------------------------------- heatmaps
def part_heatmaps(args, refs, out_dir):
    from PIL import Image
    from src.tokenizer import CaptionTokenizer
    from src.utils.run_io import run_dir, load_trained_model
    from src.utils.beam_search import CaptionGenerator
    from src.feature_extraction import load_features

    tok = CaptionTokenizer.load(os.path.join(args.processed_dir, "tokenizer.json"))
    grid, gidx = load_features(args.features_dir, "grid")
    samples = sample_images(refs, args.n_samples)
    hdir = os.path.join(out_dir, "heatmaps")
    os.makedirs(hdir, exist_ok=True)
    gradcam_cfg = "hybrid_frozen"
    record = {"images": samples, "models": {}}

    for cfg in ["attention", "transformer", gradcam_cfg]:
        rdir = run_dir(args.runs_dir, cfg, 1)
        if not os.path.exists(os.path.join(rdir, "best.weights.h5")):
            print(f"[heatmaps] skipping {cfg}: not trained")
            continue
        model, c = load_trained_model(rdir, tok.vocab_size, args.processed_dir)
        gen = CaptionGenerator(model, tok, c["max_length"])
        method = "Grad-CAM" if cfg == gradcam_cfg else "attention"
        record["models"][cfg] = {"method": method, "items": {}}
        for img in samples:
            g = np.asarray(grid[gidx[img]], np.float32)
            if cfg == gradcam_cfg:
                state = gen.encode_images(g.mean(axis=(0, 1))[None])
                cap = gen.beam(state, 3)[0]
                maps = an.gradcam_maps(model, g, cap, tok, c["max_length"])
            else:
                state = gen.encode_images(g[None])
                cap = gen.beam(state, 3)[0]
                maps = gen.attention_maps(state[0], cap)
            record["models"][cfg]["items"][img] = {
                "caption": cap, "words": [m["word"] for m in maps],
                "maps": [np.round(m["map"], 4).tolist() for m in maps]}
            if args.image_dir and os.path.exists(os.path.join(args.image_dir, img)):
                an.word_strip(Image.open(os.path.join(args.image_dir, img)), maps,
                              f"{LABELS[cfg]} ({method}): \"{cap}\"",
                              os.path.join(hdir, f"{cfg}__{an.safe_name(img)}.png"))
        print(f"[heatmaps] {cfg}: {len(samples)} images")
    with open(os.path.join(hdir, "heatmaps.json"), "w") as f:
        json.dump(record, f)


# --------------------------------------------------------------------------- beam trees
def part_beamtree(args, refs, out_dir):
    from src.tokenizer import CaptionTokenizer
    from src.utils.run_io import run_dir, load_trained_model
    from src.utils.beam_search import CaptionGenerator
    from src.feature_extraction import load_features

    best = best_classic(args.runs_dir, out_dir)
    mode = "pooled" if best.endswith("frozen") else "conv4"
    tok = CaptionTokenizer.load(os.path.join(args.processed_dir, "tokenizer.json"))
    feats, idx = load_features(args.features_dir, mode)
    model, c = load_trained_model(run_dir(args.runs_dir, best, 1), tok.vocab_size, args.processed_dir)
    gen = CaptionGenerator(model, tok, c["max_length"])
    trees = {}
    tdir = os.path.join(out_dir, "beamtrees")
    os.makedirs(tdir, exist_ok=True)
    for img in sample_images(refs, args.n_samples):
        state = gen.encode_images(np.asarray(feats[idx[img]], np.float32)[None])[0]
        tr = gen.beam_trace(state, 3)
        tr["greedy"] = gen.greedy(state[None])[0]
        trees[img] = tr
        draw_beamtree(tr, os.path.join(tdir, f"{an.safe_name(img)}.png"))
    with open(os.path.join(tdir, "beamtrees.json"), "w") as f:
        json.dump({"config": best, "beam_width": 3, "trees": trees}, f)
    print(f"[beamtree] {best}: {len(trees)} images")


def draw_beamtree(trace: dict, path: str):
    """Columns = steps, rows = the 3 beams kept; lines join each word to the beam it extends."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = trace["steps"]
    k = len(steps[0]["word"])
    fig, ax = plt.subplots(figsize=(1.5 * len(steps) + 3.5, 1.2 * k + 1.5))
    best_row = int(np.argmax(trace["final_scores"]))
    # trace back the winning path so it can be highlighted
    on_path, row = set(), best_row
    for t in range(len(steps) - 1, -1, -1):
        on_path.add((t, row))
        row = steps[t]["parent"][row]
    ax.text(-1, 0, "<start>", ha="center", va="center", fontsize=10, zorder=2,
            bbox=dict(boxstyle="round", fc="#fde0dd", ec="#555"))
    for t, st in enumerate(steps):
        for r in range(k):
            parent_row = st["parent"][r] if t > 0 else 0
            hot = (t, r) in on_path
            ax.plot([t - 1, t], [-parent_row, -r], color="#d62728" if hot else "#999",
                    lw=2.5 if hot else 1, zorder=1)
            w = st["word"][r] if not (t > 0 and steps[t - 1]["finished"][st["parent"][r]]) else ""
            ax.text(t, -r, w, ha="center", va="center", fontsize=10, zorder=2,
                    bbox=dict(boxstyle="round", fc="#fde0dd" if hot else "white", ec="#555"))
    ax.set_title(f"Beam search (k = {k}).  Beam: \"{trace['caption']}\"   Greedy: \"{trace['greedy']}\"",
                 fontsize=11)
    ax.set_xlim(-1.7, len(steps) - 0.3); ax.set_ylim(-k + 0.4, 0.6)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parts", default="errors,tsne,heatmaps,beamtree")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--features_dir", default="data/features")
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--out_dir", default="results/analysis")
    p.add_argument("--image_dir", default=None, help="Raw images, for the figures.")
    p.add_argument("--n_samples", type=int, default=N_SAMPLES)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.processed_dir, "test_captions.json")) as f:
        refs = json.load(f)
    parts = {"errors": part_errors, "tsne": part_tsne, "heatmaps": part_heatmaps, "beamtree": part_beamtree}
    for name in args.parts.split(","):
        parts[name](args, refs, args.out_dir)
    print(f"Done. Outputs in {args.out_dir}/")


if __name__ == "__main__":
    main()
