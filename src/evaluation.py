"""
src/evaluation.py

Caption-quality metrics. All functions take
    refs: {image: [5 reference captions]}   hyps: {image: generated caption}

  coco_scores          BLEU-1..4, METEOR, ROUGE-L, CIDEr with the official COCO
                       caption evaluation code (pycocoevalcap) — the field standard
  nltk_corpus_bleu     corpus BLEU-1..4 with NLTK (cross-check)
  sentence_bleu_avg    mean of per-image smoothed BLEU (what the old evaluate.py
                       reported; kept only for comparison with earlier results)
  paper_protocol       Kavitha & Karpagam (2025) Table 3 style: per-image BLEU-1
                       on 40 example images, reporting the MAX (and the mean)
  key_object_recall    does the caption name the image's main subject?
  diversity            vocabulary size, unique captions, novel captions, length
"""

import warnings
from collections import Counter

import numpy as np

# Unsmoothed BLEU on a single sentence is often 0 for 3/4-grams; NLTK warns
# every time. The behaviour is intended, so the warning is silenced.
warnings.filterwarnings("ignore", message=".*counts of \\d-gram overlaps.*")

from src.tokenizer import clean_caption

# Main-subject groups for key-object recall. A caption "names the subject" if
# it contains any word of the image's main group.
SUBJECT_GROUPS = {
    "dog": ["dog", "dogs", "puppy", "puppies"],
    "man": ["man", "men", "guy", "guys", "gentleman"],
    "woman": ["woman", "women", "lady", "ladies"],
    "boy": ["boy", "boys"],
    "girl": ["girl", "girls"],
    "child": ["child", "children", "kid", "kids", "baby", "toddler", "toddlers"],
    "people": ["people", "person", "crowd", "group"],
    "player": ["player", "players", "football", "soccer", "basketball", "baseball", "team"],
    "biker": ["biker", "bikers", "bicyclist", "cyclist", "bike", "bicycle", "motorcycle"],
    "skier": ["skier", "skiers", "snowboarder", "snowboarders", "snowboard"],
    "surfer": ["surfer", "surfers", "surfboard"],
    "climber": ["climber", "climbers", "climbing", "climbs"],
    "skateboarder": ["skateboarder", "skateboarders", "skateboard", "skater"],
    "cat": ["cat", "cats", "kitten"],
    "horse": ["horse", "horses"],
    "bird": ["bird", "birds"],
    "car": ["car", "cars", "truck"],
}
_WORD_TO_GROUP = {w: g for g, ws in SUBJECT_GROUPS.items() for w in ws}


def _tok(text: str) -> list[str]:
    return clean_caption(text).split()


# --------------------------------------------------------------------- COCO
def coco_scores(refs: dict, hyps: dict, use_meteor: bool = True) -> dict:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    keys = sorted(hyps)
    gts = {k: [" ".join(_tok(r)) for r in refs[k]] for k in keys}
    res = {k: [" ".join(_tok(hyps[k])) or "<empty>"] for k in keys}

    out = {}
    bleu, _ = Bleu(4).compute_score(gts, res, verbose=0)
    out.update({f"bleu{i + 1}": float(b) for i, b in enumerate(bleu)})
    out["cider"] = float(Cider().compute_score(gts, res)[0])
    out["rouge_l"] = float(Rouge().compute_score(gts, res)[0])
    if use_meteor:
        out.update(meteor(gts, res))
    return out


_METEOR_BROKEN = None  # set to the error message after the first failure


def meteor(gts: dict, res: dict, timeout: int = 900) -> dict:
    """
    Official METEOR 1.5 (the jar bundled with pycocoevalcap; needs Java).

    Runs the jar directly on temporary files with a time limit instead of
    using pycocoevalcap's Meteor class, whose lock deadlocks forever if the
    Java process returns anything unexpected. Gives the identical score.

    If it fails, METEOR is skipped (not replaced by a different
    implementation, which would make runs incomparable); it can be
    recomputed later from predictions.csv with scripts/rescore_meteor.py.
    """
    global _METEOR_BROKEN
    import os
    import shutil
    import subprocess
    import tempfile

    if _METEOR_BROKEN:
        return {}
    if shutil.which("java") is None:
        _METEOR_BROKEN = "java not installed"
        print("[meteor] skipped: Java is not installed (apt-get install -y default-jre-headless)")
        return {}
    import pycocoevalcap.meteor.meteor as m
    jar = os.path.join(os.path.dirname(m.__file__), "meteor-1.5.jar")

    keys = sorted(res)
    n_refs = max(len(gts[k]) for k in keys)
    with tempfile.TemporaryDirectory() as d:
        hyp_path, ref_path = os.path.join(d, "hyp.txt"), os.path.join(d, "ref.txt")
        with open(hyp_path, "w") as f:
            f.write("\n".join(res[k][0] for k in keys) + "\n")
        with open(ref_path, "w") as f:
            for k in keys:  # exactly n_refs lines per image; repeat the last if fewer
                refs = list(gts[k]) + [gts[k][-1]] * (n_refs - len(gts[k]))
                f.write("\n".join(refs) + "\n")
        cmd = ["java", "-Xmx2G", "-jar", jar, hyp_path, ref_path, "-l", "en", "-norm", "-r", str(n_refs)]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            _METEOR_BROKEN = "timeout"
            print(f"[meteor] skipped: METEOR took longer than {timeout}s")
            return {}

    for line in out.stdout.splitlines():
        if line.startswith("Final score:"):
            return {"meteor": float(line.split(":", 1)[1]), "meteor_impl": "meteor-1.5"}
    _METEOR_BROKEN = f"exit code {out.returncode}"
    tail = "\n".join((out.stderr + out.stdout).strip().splitlines()[-8:])
    print(f"[meteor] skipped: the METEOR jar failed (exit code {out.returncode}). Last output:\n{tail}")
    return {}


# --------------------------------------------------------------------- NLTK
def nltk_corpus_bleu(refs: dict, hyps: dict) -> dict:
    from nltk.translate.bleu_score import corpus_bleu

    keys = sorted(hyps)
    R = [[_tok(r) for r in refs[k]] for k in keys]
    H = [_tok(hyps[k]) for k in keys]
    weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (1 / 3, 1 / 3, 1 / 3, 0), (0.25,) * 4]
    return {f"bleu{i + 1}": float(corpus_bleu(R, H, weights=w)) for i, w in enumerate(weights)}


def sentence_bleu_per_image(refs: dict, hyps: dict, smoothing: bool = True) -> dict:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    sm = SmoothingFunction().method4 if smoothing else SmoothingFunction().method0
    out = {}
    for k in hyps:
        R, H = [_tok(r) for r in refs[k]], _tok(hyps[k])
        out[k] = {
            "bleu1": float(sentence_bleu(R, H, weights=(1, 0, 0, 0), smoothing_function=sm)) if H else 0.0,
            "bleu4": float(sentence_bleu(R, H, weights=(0.25,) * 4, smoothing_function=sm)) if H else 0.0,
        }
    return out


def sentence_bleu_avg(per_image: dict) -> dict:
    return {m: float(np.mean([v[m] for v in per_image.values()])) for m in ["bleu1", "bleu4"]}


def paper_protocol(refs: dict, hyps: dict, n_images: int = 40) -> dict:
    """
    Paper 1 reports "maximum BLEU-1 ... on 40 example images" without saying
    exactly how it was computed, so we report both readings on a fixed sample of
    the first 40 test images (sorted by name):
      corpus_bleu1  BLEU-1 computed over the 40-image sample as a whole
      max_bleu1     the best single image's BLEU-1 (an upper bound, often 1.0)
      mean_bleu1    the average single-image BLEU-1
    """
    sample = sorted(hyps)[:n_images]
    sub = {k: hyps[k] for k in sample}
    per = sentence_bleu_per_image(refs, sub, smoothing=False)
    b1 = [per[k]["bleu1"] for k in sample]
    return {"n_images": len(sample),
            "corpus_bleu1": nltk_corpus_bleu(refs, sub)["bleu1"],
            "max_bleu1": float(max(b1)), "mean_bleu1": float(np.mean(b1))}


# --------------------------------------------------------------------- subjects
def main_subject(references: list[str], min_refs: int = 3) -> str | None:
    """The subject group named by the most references (at least min_refs of 5)."""
    counts = Counter()
    for r in references:
        groups = {_WORD_TO_GROUP[w] for w in _tok(r) if w in _WORD_TO_GROUP}
        counts.update(groups)
    if not counts:
        return None
    group, n = counts.most_common(1)[0]
    return group if n >= min_refs else None


def names_subject(caption: str, group: str) -> bool:
    words = set(_tok(caption))
    return any(w in words for w in SUBJECT_GROUPS[group])


def key_object_recall(refs: dict, hyps: dict) -> dict:
    per_group = Counter(), Counter()
    for k, cap in hyps.items():
        g = main_subject(refs[k])
        if g is None:
            continue
        per_group[1][g] += 1
        per_group[0][g] += names_subject(cap, g)
    hits, totals = per_group
    n = sum(totals.values())
    return {
        "recall": float(sum(hits.values()) / n) if n else None,
        "images_with_clear_subject": n,
        "per_subject": {g: {"recall": hits[g] / totals[g], "images": totals[g]}
                        for g in sorted(totals, key=totals.get, reverse=True)},
    }


# --------------------------------------------------------------------- diversity
def diversity(hyps: dict, refs: dict, train_captions: dict | None = None) -> dict:
    caps = [" ".join(_tok(c)) for c in hyps.values()]
    ref_caps = [" ".join(_tok(r)) for rs in refs.values() for r in rs]
    out = {
        "vocab_used": len({w for c in caps for w in c.split()}),
        "reference_vocab": len({w for c in ref_caps for w in c.split()}),
        "unique_caption_pct": 100 * len(set(caps)) / max(len(caps), 1),
        "avg_length": float(np.mean([len(c.split()) for c in caps])),
        "reference_avg_length": float(np.mean([len(c.split()) for c in ref_caps])),
        "most_common_captions": Counter(caps).most_common(5),
    }
    if train_captions:
        train_set = {" ".join(_tok(c)) for cs in train_captions.values() for c in cs}
        out["novel_caption_pct"] = 100 * sum(c not in train_set for c in caps) / max(len(caps), 1)
    return out
