"""
scripts/clipscore.py

Reference-free caption scoring with CLIPScore (Hessel et al., EMNLP 2021):
how well a caption matches the IMAGE itself, according to a separate
image-text model, with no reference sentences involved.

    CLIP-S(c, v)      = 2.5 * max(cos(text("A photo depicts " + c), image(v)), 0)
    RefCLIP-S(c, R, v) = harmonic mean of CLIP-S and max over references of cos(c, r)

Used to test the claim that BLEU punishes correct captions written in a
different style (e.g. BLIP-large), and as the stand-in for a human study.

To avoid scoring our CLIP-encoder model with the very network it was trained
on (ViT-B/32), the default scorer is the larger CLIP ViT-L/14.

    python scripts/clipscore.py                     # every evaluated run + the human captions
    python scripts/clipscore.py --scorer openai/clip-vit-base-patch32   # paper's original scorer

Writes results/clipscore_per_image.csv and results/table_G_clipscore.md.
Needs PyTorch + transformers (Colab has both); ~5 min on an L4.
"""

import os
import sys
import glob
import json
import argparse
import faulthandler

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

LABELS = {"lstm_frozen": "C1 LSTM, frozen", "gru_frozen": "C2 GRU, frozen", "hybrid_frozen": "C3 Hybrid, frozen",
          "lstm_finetuned": "C4 LSTM, fine-tuned", "gru_finetuned": "C5 GRU, fine-tuned",
          "hybrid_finetuned": "C6 Hybrid, fine-tuned", "paper_protocol": "Paper protocol",
          "hybrid_vgg16": "Hybrid + VGG16", "hybrid_clip": "Hybrid + CLIP",
          "attention": "A1 LSTM + attention", "transformer": "A2 Transformer",
          "blip_base": "BLIP-base (pretrained)", "blip_large": "BLIP-large (pretrained)",
          "human": "Human caption (1 of the 5 references)"}
PREFIX = "A photo depicts "


class ClipScorer:
    def __init__(self, model_id: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def images(self, paths: list[str], batch: int = 64) -> np.ndarray:
        from PIL import Image
        out = []
        for i in range(0, len(paths), batch):
            ims = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
            px = self.processor(images=ims, return_tensors="pt")["pixel_values"].to(self.device)
            with self.torch.no_grad():
                v = self.model.vision_model(pixel_values=px)
                f = self.model.visual_projection(v.pooler_output if hasattr(v, "pooler_output") else v[1])
            out.append(f.float().cpu().numpy())
        return _norm(np.concatenate(out))

    def texts(self, texts: list[str], batch: int = 256) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            tok = self.processor(text=texts[i:i + batch], return_tensors="pt", padding=True,
                                 truncation=True, max_length=77)
            tok = {k: v.to(self.device) for k, v in tok.items() if k in ("input_ids", "attention_mask")}
            with self.torch.no_grad():
                t = self.model.text_model(**tok)
                f = self.model.text_projection(t.pooler_output if hasattr(t, "pooler_output") else t[1])
            out.append(f.float().cpu().numpy())
        return _norm(np.concatenate(out))


def _norm(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)


def clip_s(cap_emb: np.ndarray, img_emb: np.ndarray) -> np.ndarray:
    return 2.5 * np.clip((cap_emb * img_emb).sum(1), 0, None)


def refclip_s(cs: np.ndarray, cap_emb: np.ndarray, ref_embs: np.ndarray) -> np.ndarray:
    """ref_embs: (N, 5, d). Harmonic mean of CLIP-S and the best caption-reference cosine."""
    ref_sim = np.clip(np.einsum("nd,nkd->nk", cap_emb, ref_embs).max(1), 0, None)
    return np.where(cs + ref_sim > 0, 2 * cs * ref_sim / np.maximum(cs + ref_sim, 1e-8), 0.0)


def collect_runs(runs_dir: str) -> list[tuple[str, int, pd.DataFrame]]:
    out = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "seed*", "eval", "predictions.csv"))):
        cfg = path.split(os.sep)[-4]
        seed = int(path.split(os.sep)[-3][4:])
        df = pd.read_csv(path, keep_default_na=False)
        dec = "beam5" if "beam5" in set(df.decoding) else sorted(set(df.decoding))[-1]
        out.append((cfg, seed, df[df.decoding == dec].sort_values("image")))
    return out


def summarise(per: pd.DataFrame, out_dir: str) -> str:
    g = per.groupby(["config", "seed"])[["clip_s", "refclip_s"]].mean().reset_index()
    rows = []
    for cfg, sub in g.groupby("config"):
        rows.append({"config": cfg, "seeds": len(sub),
                     "clip_s": sub.clip_s.mean(), "clip_s_sd": sub.clip_s.std(ddof=1) if len(sub) > 1 else None,
                     "refclip_s": sub.refclip_s.mean(),
                     "refclip_s_sd": sub.refclip_s.std(ddof=1) if len(sub) > 1 else None})
    t = pd.DataFrame(rows).sort_values("clip_s", ascending=False)

    bleu = {}
    runs_csv = os.path.join(out_dir, "all_runs.csv")
    if os.path.exists(runs_csv):
        ar = pd.read_csv(runs_csv)
        ar = ar[ar.decoding.isin(["beam5", "beam3"])]
        ar = ar[~((ar.config.str.startswith("blip")) & (ar.decoding != "beam3")) &
                ~((~ar.config.str.startswith("blip")) & (ar.decoding != "beam5"))]
        bleu = ar.groupby("config")["bleu4"].mean().to_dict()
    t["bleu4"] = t.config.map(bleu)
    t["rank_clip_s"] = t.clip_s.rank(ascending=False, method="min").astype(int)
    model_rows = t[t.config != "human"].copy()
    model_rows["rank_bleu4"] = model_rows.bleu4.rank(ascending=False, method="min")
    t = t.merge(model_rows[["config", "rank_bleu4"]], on="config", how="left")

    def f(m, s):
        return f"{m:.3f}" + (f" ± {s:.3f}" if pd.notna(s) else "")

    lines = ["| Model | Seeds | CLIP-S (vs image) | RefCLIP-S (image + references) | BLEU-4 | Rank by CLIP-S | Rank by BLEU-4 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in t.itertuples():
        lines.append(f"| {LABELS.get(r.config, r.config)} | {r.seeds} | {f(r.clip_s, r.clip_s_sd)} | "
                     f"{f(r.refclip_s, r.refclip_s_sd)} | {'—' if pd.isna(r.bleu4) else f'{r.bleu4:.3f}'} | "
                     f"{r.rank_clip_s} | {'—' if pd.isna(r.rank_bleu4) else int(r.rank_bleu4)} |")
    if len(model_rows) > 2 and model_rows.bleu4.notna().sum() > 2:
        from scipy.stats import spearmanr
        mr = model_rows.dropna(subset=["bleu4"])
        rho = spearmanr(mr.clip_s, mr.bleu4).correlation
        lines.append(f"\nSpearman rank correlation between CLIP-S and BLEU-4 across models: {rho:.2f}.")
    return "\n".join(lines)


def main():
    faulthandler.enable()
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--image_dir", default="data/raw/Images")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--scorer", default="openai/clip-vit-large-patch14")
    args = p.parse_args()

    with open(os.path.join(args.processed_dir, "test_captions.json")) as f:
        refs = json.load(f)
    names = sorted(refs)
    runs = collect_runs(args.runs_dir)
    if not runs:
        print("No evaluated runs found.")
        return

    scorer = ClipScorer(args.scorer)
    print(f"Scorer: {args.scorer} | {len(names)} test images | {len(runs)} runs")
    img = scorer.images([os.path.join(args.image_dir, n) for n in names])
    n_ref = min(len(refs[n]) for n in names)
    ref_text = [PREFIX + refs[n][k] for n in names for k in range(n_ref)]
    ref_emb = scorer.texts(ref_text).reshape(len(names), n_ref, -1)

    rows = []
    # Human control: reference 1 scored against the image, compared with the other references.
    human_cs = clip_s(ref_emb[:, 0], img)
    human_ref = refclip_s(human_cs, ref_emb[:, 0], ref_emb[:, 1:])
    rows += [{"config": "human", "seed": 0, "image": n, "caption": refs[n][0], "clip_s": a, "refclip_s": b}
             for n, a, b in zip(names, human_cs, human_ref)]

    for cfg, seed, df in runs:
        caps = dict(zip(df.image, df.caption))
        missing = [n for n in names if n not in caps]
        if missing:
            print(f"[skip] {cfg} seed {seed}: {len(missing)} test images missing")
            continue
        emb = scorer.texts([PREFIX + caps[n] for n in names])
        cs = clip_s(emb, img)
        rc = refclip_s(cs, emb, ref_emb)
        rows += [{"config": cfg, "seed": seed, "image": n, "caption": caps[n], "clip_s": a, "refclip_s": b}
                 for n, a, b in zip(names, cs, rc)]
        print(f"  {cfg:18s} seed {seed}: CLIP-S {cs.mean():.3f}  RefCLIP-S {rc.mean():.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(args.out_dir, "clipscore_per_image.csv"), index=False)
    table = (f"# Table G: CLIPScore, reference-free caption quality (scorer: {args.scorer})\n\n"
             + summarise(per, args.out_dir)
             + "\n\nCLIP-S judges the caption against the photo only; BLEU-4 against the 5 reference sentences. "
               "The human row is one of the 5 references, scored the same way.\n")
    with open(os.path.join(args.out_dir, "table_G_clipscore.md"), "w") as f:
        f.write(table)
    print("\n" + table)


if __name__ == "__main__":
    main()
