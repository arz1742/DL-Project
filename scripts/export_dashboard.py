"""
scripts/export_dashboard.py

Packs every result the dashboard shows into dashboard/data/ (JSON + small
JPEG thumbnails) and zips it, so the static website can be hosted anywhere
(Vercel) without TensorFlow, Drive or the dataset.

    python scripts/export_dashboard.py --image_dir /content/data_local/raw/Images \
        --zip /content/drive/MyDrive/DL-Project/dashboard_data.zip

Needs: finished runs (runs/), results/ (aggregate_results.py, clipscore.py,
run_analysis.py) and the grid features (for attention maps). ~3 min.

Writes:
  summary.json     every metric per model (mean ± sd over seeds), significance, errors, paper numbers
  curves.json      training curves (validation loss / accuracy per epoch, every seed)
  gallery.json     all 1000 test images: references + each model's caption and scores
  attention.json   per-word 7x7 maps for 40 images: A1, A2 (attention) and C3 (Grad-CAM)
  beamtrees.json   beam-search trees for 8 images
  tsne.json        2-D map of the test images
  img/s/*.jpg      224 px thumbnails of the 1000 test images
  img/l/*.jpg      480 px versions of the 48 images used in heatmaps and beam trees
"""

import os
import sys
import json
import glob
import shutil
import argparse
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.evaluation import main_subject  # noqa: E402

MODELS = {  # config -> (label, family)
    "lstm_frozen": ("C1 LSTM", "Part 1: classic decoders"),
    "gru_frozen": ("C2 GRU", "Part 1: classic decoders"),
    "hybrid_frozen": ("C3 Hybrid LSTM→GRU", "Part 1: classic decoders"),
    "lstm_finetuned": ("C4 LSTM, fine-tuned", "Part 1: classic decoders"),
    "gru_finetuned": ("C5 GRU, fine-tuned", "Part 1: classic decoders"),
    "hybrid_finetuned": ("C6 Hybrid, fine-tuned", "Part 1: classic decoders"),
    "paper_protocol": ("Paper-protocol replication", "Replication"),
    "hybrid_vgg16": ("Hybrid + VGG16", "Encoder comparison"),
    "hybrid_clip": ("Hybrid + CLIP", "Encoder comparison"),
    "attention": ("A1 LSTM + attention", "Part 2: attention"),
    "transformer": ("A2 Transformer", "Part 2: attention"),
    "blip_base": ("BLIP-base (pretrained)", "Pretrained reference"),
    "blip_large": ("BLIP-large (pretrained)", "Pretrained reference"),
}
GALLERY_MODELS = ["lstm_frozen", "hybrid_finetuned", "hybrid_clip", "attention", "transformer", "blip_large"]
METRICS = ["bleu1", "bleu2", "bleu3", "bleu4", "meteor", "rouge_l", "cider", "masked_acc", "top5_acc",
           "unmasked_acc", "key_object_recall", "unique_caption_pct", "avg_length", "paper_corpus_bleu1",
           "paper_max_bleu1"]
PAPER1 = {"accuracy": {"lstm": 0.8459, "gru": 0.8594, "hybrid": 0.8932},
          "bleu1_greedy": {"lstm": 0.2698, "gru": 0.3986, "hybrid": 0.5142},
          "bleu1_beam3": {"lstm": 0.4286, "gru": 0.4935, "hybrid": 0.6034},
          "meteor": 0.3124}


def _clean(x):
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else round(float(x), 5)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def agg(df: pd.DataFrame, cols: list[str]) -> dict:
    out = {}
    for c in cols:
        if c in df and df[c].notna().any():
            v = df[c].dropna()
            out[c] = {"mean": _clean(v.mean()), "sd": _clean(v.std(ddof=1)) if len(v) > 1 else None, "n": int(len(v))}
    return out


def build_summary(args) -> dict:
    runs = pd.read_csv(os.path.join(args.results_dir, "all_runs.csv"))
    summary = {"models": {}, "paper1": PAPER1, "decodings": {}}
    for cfg, (label, family) in MODELS.items():
        sub = runs[runs.config == cfg]
        if sub.empty:
            continue
        main_dec = "beam3" if cfg.startswith("blip") else "beam5"
        state = {}
        sp = os.path.join(args.runs_dir, cfg, "seed1" if not cfg.startswith("blip") else "seed0", "state.json")
        if os.path.exists(sp):
            with open(sp) as f:
                state = json.load(f)
        summary["models"][cfg] = {
            "label": label, "family": family, "seeds": int(sub.seed.nunique()), "decoding": main_dec,
            "metrics": agg(sub[sub.decoding == main_dec], METRICS),
            "params": (state.get("params") or {}).get("trainable"),
            "train_minutes": _clean(sub.train_seconds.dropna().mean() / 60) if "train_seconds" in sub and sub.train_seconds.notna().any() else None,
            "best_epoch": _clean(sub.best_epoch.dropna().mean()) if sub.best_epoch.notna().any() else None,
        }
        summary["decodings"][cfg] = {d: agg(sub[sub.decoding == d], ["bleu1", "bleu4", "cider", "paper_corpus_bleu1"])
                                     for d in sorted(set(sub.decoding))}

    cs = os.path.join(args.results_dir, "clipscore_per_image.csv")
    if os.path.exists(cs):
        c = pd.read_csv(cs)
        per_seed = c.groupby(["config", "seed"])[["clip_s", "refclip_s"]].mean().reset_index()
        for cfg, sub in per_seed.groupby("config"):
            entry = summary["models"].setdefault(cfg, {"label": "Human caption", "family": "Human reference",
                                                       "seeds": 1, "metrics": {}})
            entry["metrics"].update(agg(sub, ["clip_s", "refclip_s"]))
        if "human" in summary["models"]:
            summary["models"]["human"]["label"] = "Human caption"

    er = os.path.join(args.results_dir, "analysis", "errors_all_runs.csv")
    if os.path.exists(er):
        e = pd.read_csv(er)
        cols = ["chair_s", "chair_i", "wrong_subject", "hallucination", "wrong_colour", "wrong_count",
                "generic", "repetition", "no_detected_error"]
        for cfg, sub in e.groupby("config"):
            if cfg in summary["models"]:
                summary["models"][cfg]["errors"] = agg(sub, cols)
                summary["models"][cfg]["top_hallucinated"] = str(sub.iloc[0].top_hallucinated)

    summary["significance"] = build_significance(args)
    return summary


def build_significance(args) -> dict:
    import importlib.util
    spec = importlib.util.spec_from_file_location("sig", os.path.join(ROOT, "scripts", "significance.py"))
    sig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sig)
    with open(os.path.join(args.processed_dir, "test_captions.json")) as f:
        refs = json.load(f)
    out = {}
    tables = {"cider": sig.per_image_cider(args.runs_dir, refs), "clip_s": sig.per_image_clip(args.results_dir)}
    for metric, scores in tables.items():
        if scores is None:
            continue
        rows = []
        for question, a, b in sig.COMPARISONS:
            if a in scores and b in scores:
                r = sig.paired_bootstrap(scores[a].values, scores[b].values, args.n_boot)
                rows.append({"question": question, "a": a, "b": b, "a_mean": scores[a].mean(),
                             "b_mean": scores[b].mean(), **r})
        for r, p in zip(rows, sig.holm([r["p"] for r in rows])):
            r["p_holm"] = p
        out[metric] = [{k: _clean(v) for k, v in r.items()} for r in rows]
    return out


def build_curves(args) -> dict:
    out = {}
    for path in sorted(glob.glob(os.path.join(args.runs_dir, "*", "seed*", "history.json"))):
        cfg, seed = path.split(os.sep)[-3], path.split(os.sep)[-2]
        if cfg not in MODELS:
            continue
        with open(path) as f:
            h = json.load(f)
        keep = {k: [round(float(x), 4) for x in v] for k, v in h.items()
                if k in ("loss", "val_loss", "masked_acc", "val_masked_acc", "val_top5_acc", "val_unmasked_acc")}
        out.setdefault(cfg, {})[seed] = keep
    return out


def build_gallery(args, refs: dict) -> dict:
    names = sorted(refs)
    caps, scores = {}, {}
    for cfg in GALLERY_MODELS:
        seed = 0 if cfg.startswith("blip") else 1
        p = os.path.join(args.runs_dir, cfg, f"seed{seed}", "eval", "predictions.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, keep_default_na=False)
        dec = "beam3" if cfg.startswith("blip") else "beam5"
        df = df[df.decoding == dec]
        caps[cfg] = dict(zip(df.image, df.caption))
        scores[cfg] = dict(zip(df.image, df.bleu4))
    clip = {}
    cs = os.path.join(args.results_dir, "clipscore_per_image.csv")
    if os.path.exists(cs):
        c = pd.read_csv(cs)
        c = c[c.seed.isin([0, 1])]
        for cfg, sub in c.groupby("config"):
            clip[cfg] = dict(zip(sub.image, sub.clip_s))
    labels = {}
    ep = os.path.join(args.results_dir, "analysis", "errors_per_image.json")
    if os.path.exists(ep):
        with open(ep) as f:
            labels = json.load(f)
    items = []
    for n in names:
        item = {"id": n.rsplit(".", 1)[0], "subject": main_subject(refs[n]) or "", "refs": refs[n], "m": {}}
        for cfg in caps:
            if n in caps[cfg]:
                item["m"][cfg] = {"c": caps[cfg][n], "b4": _clean(scores[cfg].get(n)),
                                  "cs": _clean(clip.get(cfg, {}).get(n)), "e": labels.get(cfg, {}).get(n, [])}
        if "human" in clip:
            item["human_cs"] = _clean(clip["human"].get(n))
        items.append(item)
    return {"models": [c for c in GALLERY_MODELS if c in caps], "items": items}


def build_attention(args, refs: dict, n_images: int) -> tuple[dict, list[str]]:
    from src.tokenizer import CaptionTokenizer
    from src.utils.run_io import run_dir, load_trained_model
    from src.utils.beam_search import CaptionGenerator
    from src.feature_extraction import load_features
    from src import analysis as an

    tok = CaptionTokenizer.load(os.path.join(args.processed_dir, "tokenizer.json"))
    grid, gidx = load_features(args.features_dir, "grid")
    clear = [n for n in sorted(refs) if main_subject(refs[n])]
    step = max(1, len(clear) // n_images)
    images = clear[::step][:n_images]
    out = {"images": [n.rsplit(".", 1)[0] for n in images], "models": {}}
    for cfg, method in [("attention", "attention"), ("transformer", "attention"), ("hybrid_frozen", "Grad-CAM")]:
        rdir = run_dir(args.runs_dir, cfg, 1)
        if not os.path.exists(os.path.join(rdir, "best.weights.h5")):
            continue
        model, c = load_trained_model(rdir, tok.vocab_size, args.processed_dir)
        gen = CaptionGenerator(model, tok, c["max_length"])
        items = {}
        for n in images:
            g = np.asarray(grid[gidx[n]], np.float32)
            if method == "Grad-CAM":
                cap = gen.beam(gen.encode_images(g.mean(axis=(0, 1))[None]), 3)[0]
                maps = an.gradcam_maps(model, g, cap, tok, c["max_length"])
            else:
                st = gen.encode_images(g[None])
                cap = gen.beam(st, 3)[0]
                maps = gen.attention_maps(st[0], cap)
            items[n.rsplit(".", 1)[0]] = {"caption": cap, "words": [m["word"] for m in maps],
                                          "maps": [[round(float(x), 4) for x in m["map"].reshape(-1)] for m in maps]}
        out["models"][cfg] = {"label": MODELS[cfg][0], "method": method, "items": items}
        print(f"[attention] {cfg}: {len(items)} images")
    return out, images


def save_thumb(src: str, dst: str, longest: int, quality: int):
    from PIL import Image
    im = Image.open(src).convert("RGB")
    im.thumbnail((longest, longest))
    im.save(dst, "JPEG", quality=quality, optimize=True, progressive=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--features_dir", default="data/features")
    p.add_argument("--image_dir", default="data/raw/Images")
    p.add_argument("--out_dir", default="dashboard/data")
    p.add_argument("--zip", default=None,
                   help="Write a zip with everything the local dashboard needs: dashboard/data, results/, "
                        "the live models' weights and the tokenizer. Unzip it into the project folder.")
    p.add_argument("--n_attention", type=int, default=40)
    p.add_argument("--n_boot", type=int, default=5000)
    args = p.parse_args()

    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(os.path.join(args.out_dir, "img", "s"))
    os.makedirs(os.path.join(args.out_dir, "img", "l"))
    with open(os.path.join(args.processed_dir, "test_captions.json")) as f:
        refs = json.load(f)

    def dump(name, obj):
        with open(os.path.join(args.out_dir, name), "w") as f:
            json.dump(obj, f, separators=(",", ":"), default=_clean)
        print(f"  {name}: {os.path.getsize(os.path.join(args.out_dir, name)) / 1e3:.0f} kB")

    print("Summary + significance ...")
    dump("summary.json", build_summary(args))
    dump("curves.json", build_curves(args))
    print("Gallery ...")
    dump("gallery.json", build_gallery(args, refs))

    tsne_csv = os.path.join(args.results_dir, "analysis", "tsne_test_images.csv")
    if os.path.exists(tsne_csv):
        t = pd.read_csv(tsne_csv, keep_default_na=False)
        dump("tsne.json", {"points": [[r.image.rsplit(".", 1)[0], round(r.x, 2), round(r.y, 2), r.subject]
                                      for r in t.itertuples()]})

    large = set()
    bt = os.path.join(args.results_dir, "analysis", "beamtrees", "beamtrees.json")
    if os.path.exists(bt):
        with open(bt) as f:
            trees = json.load(f)
        trees["trees"] = {k.rsplit(".", 1)[0]: v for k, v in trees["trees"].items()}
        large |= {k + ".jpg" for k in trees["trees"]}
        dump("beamtrees.json", trees)

    print("Attention maps ...")
    att, att_images = build_attention(args, refs, args.n_attention)
    large |= set(att_images)
    dump("attention.json", att)

    print("Images ...")
    for n in sorted(refs):
        save_thumb(os.path.join(args.image_dir, n), os.path.join(args.out_dir, "img", "s", n), 224, 70)
    for n in sorted(large):
        save_thumb(os.path.join(args.image_dir, n), os.path.join(args.out_dir, "img", "l", n), 480, 80)

    total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(args.out_dir) for f in fs)
    print(f"dashboard/data: {total / 1e6:.1f} MB")
    if args.zip:
        write_bundle(args)


# hybrid_finetuned is left out: its weights hold the ResNet50 conv5 block (~200 MB) for a 1-2% gain over C3.
LIVE = ["hybrid_frozen", "attention", "transformer", "hybrid_clip"]


def write_bundle(args):
    """Zip paths are relative to the project folder, so unzipping there puts everything in place."""
    os.makedirs(os.path.dirname(args.zip) or ".", exist_ok=True)
    entries = []
    for base, arc in [(args.out_dir, "dashboard/data"), (args.results_dir, "results")]:
        for dp, _, fs in os.walk(base):
            for f in fs:
                full = os.path.join(dp, f)
                entries.append((full, os.path.join(arc, os.path.relpath(full, base))))
    for cfg in LIVE:
        for f in ["config.yaml", "best.weights.h5", "state.json"]:
            full = os.path.join(args.runs_dir, cfg, "seed1", f)
            if os.path.exists(full):
                entries.append((full, f"runs/{cfg}/seed1/{f}"))
    entries.append((os.path.join(args.processed_dir, "tokenizer.json"), "data/processed/tokenizer.json"))
    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
        for full, arc in entries:
            z.write(full, arc)
    print(f"Bundle: {args.zip} ({os.path.getsize(args.zip) / 1e6:.0f} MB, {len(entries)} files)")


if __name__ == "__main__":
    main()
