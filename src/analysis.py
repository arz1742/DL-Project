"""
src/analysis.py

The showpiece analyses. Each function works on files the pipeline already
writes (predictions.csv, feature arrays, trained runs), so most of them need no
GPU.

  hallucination_scores   CHAIR-style object hallucination: how often a caption
                         names an object that none of the 5 references mention
  error_taxonomy         labels every caption with its mistake types (wrong
                         subject, made-up object, wrong colour, wrong count,
                         generic, repetition)
  tsne_coordinates       2-D map of the ResNet50 image features
  gradcam_maps           where a CLASSIC (frozen-encoder) model looked for each word
  overlay / word_strip   draw a 7x7 map on top of the photo
"""

import re
from collections import Counter

import numpy as np

from src.tokenizer import clean_caption
from src.evaluation import SUBJECT_GROUPS

# --------------------------------------------------------------------------- objects
# Object groups for hallucination checks: a caption "mentions" a group if it
# contains any of its words; it hallucinates the group if no reference does.
OBJECT_GROUPS = {
    **SUBJECT_GROUPS,
    "ball": ["ball", "balls", "frisbee"],
    "water": ["water", "pool", "lake", "river", "ocean", "sea", "wave", "waves", "stream", "puddle"],
    "beach": ["beach", "sand", "sandy", "shore"],
    "snow": ["snow", "snowy", "ice"],
    "grass": ["grass", "grassy", "field", "lawn", "meadow"],
    "street": ["street", "road", "sidewalk", "city", "crosswalk"],
    "mountain": ["mountain", "mountains", "hill", "hillside", "cliff"],
    "rock": ["rock", "rocks", "rocky", "boulder"],
    "tree": ["tree", "trees", "woods", "forest"],
    "building": ["building", "buildings", "house", "wall"],
    "stairs": ["stairs", "steps", "staircase"],
    "bench": ["bench", "chair", "couch", "sofa"],
    "hat": ["hat", "helmet", "cap"],
    "stick": ["stick", "sticks", "toy"],
    "fence": ["fence", "gate"],
    "boat": ["boat", "boats", "canoe", "kayak"],
    "dirt": ["dirt", "mud", "muddy", "trail", "path"],
}
# For hallucination, all person words count as one object: calling a woman "a
# person" or "a man" is a wrong-subject error, not a made-up object.
PERSON_GROUPS = {"man", "woman", "boy", "girl", "child", "people", "player", "biker",
                 "skier", "surfer", "climber", "skateboarder"}
_OBJ_WORD = {w: ("person" if g in PERSON_GROUPS else g) for g, ws in OBJECT_GROUPS.items() for w in ws}
# words that only mean an activity/object, not a person
for _w in ["bike", "bicycle", "motorcycle", "snowboard", "surfboard", "skateboard", "football",
           "soccer", "basketball", "baseball", "climbing", "climbs"]:
    _OBJ_WORD[_w] = {"bike": "bike", "bicycle": "bike", "motorcycle": "bike"}.get(_w, "sport_gear")

COLOURS = {"black", "white", "brown", "red", "blue", "green", "yellow", "orange", "pink",
           "purple", "gray", "grey", "tan", "golden"}
NUMBERS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "several": 3, "many": 5,
           "group": 3, "crowd": 5, "couple": 2}

ERROR_TYPES = ["wrong_subject", "hallucination", "wrong_colour", "wrong_count", "generic", "repetition"]


def _words(text: str) -> list[str]:
    return clean_caption(text).split()


def objects_in(text: str) -> set[str]:
    return {_OBJ_WORD[w] for w in _words(text) if w in _OBJ_WORD}


# --------------------------------------------------------------------------- hallucination
def hallucination_scores(captions: dict, refs: dict) -> dict:
    """
    CHAIR-style metrics (Rohrbach et al. 2018), adapted to Flickr8k:
      chair_i  hallucinated object mentions / all object mentions
      chair_s  captions with at least one hallucinated object / all captions
    "Hallucinated" = the caption names an object group that none of the 5
    references mention. Also returns the most often hallucinated objects.
    """
    mentioned = hallucinated = bad_caps = 0
    counts = Counter()
    for img, cap in captions.items():
        ref_objs = set().union(*(objects_in(r) for r in refs[img]))
        cap_objs = objects_in(cap)
        made_up = cap_objs - ref_objs
        mentioned += len(cap_objs)
        hallucinated += len(made_up)
        bad_caps += bool(made_up)
        counts.update(made_up)
    n = max(len(captions), 1)
    return {"chair_i": hallucinated / max(mentioned, 1), "chair_s": bad_caps / n,
            "objects_per_caption": mentioned / n, "top_hallucinated": counts.most_common(8)}


# --------------------------------------------------------------------------- error taxonomy
def error_labels(caption: str, references: list[str], main_subject: str | None,
                 common_captions: set[str]) -> list[str]:
    """Mistake types for one caption (empty list = no detected mistake)."""
    words = _words(caption)
    ref_words = [set(_words(r)) for r in references]
    all_ref = set().union(*ref_words)
    labels = []

    if main_subject and not any(w in words for w in SUBJECT_GROUPS[main_subject]):
        labels.append("wrong_subject")
    ref_objs = set().union(*(objects_in(r) for r in references))
    if objects_in(caption) - ref_objs:
        labels.append("hallucination")
    cap_colours = COLOURS & set(words)
    if cap_colours and not (cap_colours & all_ref) and (COLOURS & all_ref):
        labels.append("wrong_colour")
    cap_nums = {NUMBERS[w] for w in words if w in NUMBERS}
    ref_nums = {NUMBERS[w] for w in all_ref if w in NUMBERS}
    if cap_nums and not (cap_nums & ref_nums):
        labels.append("wrong_count")
    if " ".join(words) in common_captions:
        labels.append("generic")
    if any(a == b for a, b in zip(words, words[1:])) or _repeated_bigram(words):
        labels.append("repetition")
    return labels


def _repeated_bigram(words: list[str]) -> bool:
    bigrams = list(zip(words, words[1:]))
    return len(bigrams) != len(set(bigrams))


def error_taxonomy(captions: dict, refs: dict, main_subjects: dict, n_common: int = 10) -> dict:
    """Labels for every caption, plus how often each error type occurs."""
    freq = Counter(" ".join(_words(c)) for c in captions.values())
    common = {c for c, k in freq.most_common(n_common) if k > 1}
    per_image = {img: error_labels(cap, refs[img], main_subjects.get(img), common)
                 for img, cap in captions.items()}
    n = max(len(per_image), 1)
    rates = {t: sum(t in v for v in per_image.values()) / n for t in ERROR_TYPES}
    rates["no_detected_error"] = sum(not v for v in per_image.values()) / n
    return {"rates": rates, "per_image": per_image}


# --------------------------------------------------------------------------- t-SNE
def tsne_coordinates(features: np.ndarray, seed: int = 0, perplexity: float = 30.0) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    x = np.asarray(features, np.float32)
    x = PCA(n_components=min(50, x.shape[0] - 1, x.shape[1]), random_state=seed).fit_transform(x)
    perplexity = min(perplexity, (x.shape[0] - 1) / 3)
    return TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(x)


# --------------------------------------------------------------------------- Grad-CAM
def gradcam_maps(model, grid: np.ndarray, caption: str, tokenizer, max_length: int) -> list[dict]:
    """
    Grad-CAM (Selvaraju et al. 2017) for a FROZEN-encoder classic model.

    ResNet50's pooled vector is the average of its 7x7x2048 grid, so the
    gradient of a word's probability with respect to the pooled input gives
    each channel's importance, and the map is ReLU(sum_k w_k * grid[:, :, k]).
    Computed for each word of `caption` given the words before it.
    """
    import tensorflow as tf

    seq_len = max_length - 1
    words = caption.split() + ["<end>"]
    ids = [tokenizer.word2idx.get(w, tokenizer.word2idx["<unk>"]) for w in words][:seq_len - 1]
    seq = np.full((1, seq_len), tokenizer.word2idx["<pad>"], np.int32)
    seq[0, 0] = tokenizer.word2idx["<start>"]
    for j, w in enumerate(ids):
        seq[0, j + 1] = w

    grid = np.asarray(grid, np.float32)
    pooled = tf.constant(grid.mean(axis=(0, 1))[None])
    out = []
    for j, w in enumerate(ids):
        with tf.GradientTape() as tape:
            tape.watch(pooled)
            probs = model([pooled, tf.constant(seq)], training=False)
            score = tf.math.log(probs[0, j, w] + 1e-9)
        weights = tape.gradient(score, pooled).numpy()[0]           # (2048,)
        cam = np.maximum((grid * weights).sum(-1), 0)               # (7, 7)
        cam = cam / cam.sum() if cam.sum() > 0 else np.full((7, 7), 1 / 49)
        out.append({"word": words[j], "map": cam})
    return out


# --------------------------------------------------------------------------- drawing
def _fit(image, longest: int = 256):
    """RGB copy of `image` with its longest side = `longest`, aspect ratio kept."""
    image = image.convert("RGB")
    w, h = image.size
    scale = longest / max(w, h)
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))))


def overlay(ax, image, att_map, title: str = "", alpha: float = 0.55):
    """Shows `image` (aspect ratio kept) with a 7x7 map upsampled and laid over it."""
    from PIL import Image

    img = _fit(image)
    m = np.asarray(Image.fromarray((att_map / att_map.max() * 255).astype(np.uint8))
                   .resize(img.size, Image.BICUBIC), np.float32) / 255
    ax.imshow(np.asarray(img))
    ax.imshow(m, cmap="inferno", alpha=alpha, vmin=0, vmax=1)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def word_strip(image, maps: list[dict], caption_label: str, path: str, max_words: int = 10):
    """One row: the photo, then one heatmap per word. Saved to `path`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    maps = maps[:max_words]
    w, h = _fit(image).size
    panel_w = 2.4 * w / max(w, h)
    fig, axes = plt.subplots(1, len(maps) + 1, figsize=(panel_w * (len(maps) + 1), 2.4 * h / max(w, h) + 0.6))
    axes[0].imshow(np.asarray(_fit(image)))
    axes[0].set_title("photo", fontsize=11)
    axes[0].axis("off")
    for ax, mp in zip(axes[1:], maps):
        overlay(ax, image, mp["map"], mp["word"])
    fig.suptitle(caption_label, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
