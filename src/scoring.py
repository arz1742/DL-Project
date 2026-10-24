"""
src/scoring.py

Caption-level scoring shared by evaluate.py (our models) and
scripts/eval_pretrained.py (BLIP). Deliberately imports no TensorFlow, so the
PyTorch-based BLIP script never loads both frameworks onto the GPU at once.
"""

import os
import csv

from src import evaluation as ev
from src.utils.run_io import write_json


def score_captions(refs: dict, hyps: dict, train_caps: dict, dec: str, use_meteor: bool = True):
    """Every caption-level metric for one set of generated captions, plus per-image rows."""
    per_img = ev.sentence_bleu_per_image(refs, hyps)
    m = {
        "coco": ev.coco_scores(refs, hyps, use_meteor=use_meteor),
        "nltk_corpus": ev.nltk_corpus_bleu(refs, hyps),
        "sentence_avg_smoothed": ev.sentence_bleu_avg(per_img),
        "paper_protocol": ev.paper_protocol(refs, hyps),
        "key_object": ev.key_object_recall(refs, hyps),
        "diversity": ev.diversity(hyps, refs, train_caps),
    }
    rows = []
    for n in sorted(hyps):
        subj = ev.main_subject(refs[n])
        rows.append({
            "image": n, "decoding": dec, "caption": hyps[n],
            "bleu1": round(per_img[n]["bleu1"], 4), "bleu4": round(per_img[n]["bleu4"], 4),
            "subject": subj or "", "subject_hit": "" if subj is None else int(ev.names_subject(hyps[n], subj)),
            "references": " | ".join(refs[n]),
        })
    return m, rows


def print_summary(dec: str, m: dict, secs: float) -> None:
    c = m["coco"]
    print(f"[{dec:7s}] BLEU-1 {c['bleu1']:.4f}  BLEU-4 {c['bleu4']:.4f}  CIDEr {c['cider']:.4f}  "
          f"METEOR {c.get('meteor', float('nan')):.4f}  key-object {m['key_object']['recall']:.3f}  "
          f"({secs:.0f}s)")


def write_outputs(out_dir: str, metrics: dict, pred_rows: list[dict]) -> None:
    write_json(os.path.join(out_dir, "metrics.json"), metrics)
    with open(os.path.join(out_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pred_rows[0]))
        w.writeheader()
        w.writerows(pred_rows)
    print(f"Saved {out_dir}/metrics.json and predictions.csv")
