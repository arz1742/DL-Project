"""
tests/test_metrics_and_eval.py

Checks that each metric counts exactly what it claims to — especially the
masked vs unmasked accuracy difference behind the "accuracy illusion".
Runs on CPU in a few seconds; no dataset needed.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.models.metrics import TokenAccuracy, masked_ce, padded_ce
from src import evaluation as ev


def _one_hot_probs(pred_ids, vocab=6):
    p = np.full(pred_ids.shape + (vocab,), 0.01, np.float32)
    np.put_along_axis(p, pred_ids[..., None], 0.95, axis=-1)
    return p


def test_masked_accuracy_ignores_padding_but_unmasked_counts_it():
    # 3 real words (1 correct) then 5 pads that the model predicts as pad
    y_true = np.array([[3, 4, 5, 0, 0, 0, 0, 0]])
    y_pred = _one_hot_probs(np.array([[3, 1, 1, 0, 0, 0, 0, 0]]))
    masked = TokenAccuracy(ignore_padding=True)
    unmasked = TokenAccuracy(ignore_padding=False)
    masked.update_state(y_true, y_pred)
    unmasked.update_state(y_true, y_pred)
    assert abs(float(masked.result()) - 1 / 3) < 1e-6        # 1 of 3 real words
    assert abs(float(unmasked.result()) - 6 / 8) < 1e-6      # padding inflates it to 0.75


def test_top5_accuracy():
    y_true = np.array([[2, 0]])
    probs = np.array([[[0.0, 0.3, 0.05, 0.25, 0.2, 0.2], [1, 0, 0, 0, 0, 0]]], np.float32)
    top1, top5 = TokenAccuracy(k=1), TokenAccuracy(k=5)
    top1.update_state(y_true, probs)
    top5.update_state(y_true, probs)
    assert float(top1.result()) == 0.0 and float(top5.result()) == 1.0


def test_masked_loss_ignores_padding_positions():
    y_true = np.array([[1, 2, 0, 0]])
    good = _one_hot_probs(np.array([[1, 2, 5, 5]]))   # wrong only on padding
    perfect = _one_hot_probs(np.array([[1, 2, 0, 0]]))
    assert abs(float(masked_ce(y_true, good)) - float(masked_ce(y_true, perfect))) < 1e-6
    assert float(padded_ce(y_true, good)) > float(padded_ce(y_true, perfect))


def test_main_subject_and_recall():
    refs = {"a.jpg": ["a dog runs", "the dog jumps", "a puppy plays", "grass field", "a man and dog"],
            "b.jpg": ["sky", "blue sky", "clouds", "the sun", "a sunny day"]}
    assert ev.main_subject(refs["a.jpg"]) == "dog"
    assert ev.main_subject(refs["b.jpg"]) is None
    r = ev.key_object_recall(refs, {"a.jpg": "a brown dog on grass", "b.jpg": "a man"})
    assert r["recall"] == 1.0 and r["images_with_clear_subject"] == 1


def test_coco_bleu_is_one_for_exact_match_and_paper_protocol_keys():
    refs = {f"{i}.jpg": ["a dog runs on the grass"] * 5 for i in range(3)}
    hyps = {k: "a dog runs on the grass" for k in refs}
    s = ev.coco_scores(refs, hyps, use_meteor=False)
    assert abs(s["bleu1"] - 1.0) < 1e-6 and abs(s["bleu4"] - 1.0) < 1e-6
    pp = ev.paper_protocol(refs, hyps)
    assert set(pp) == {"n_images", "corpus_bleu1", "max_bleu1", "mean_bleu1"}


def test_diversity_counts_unique_and_novel():
    refs = {"a.jpg": ["x"], "b.jpg": ["y"]}
    d = ev.diversity({"a.jpg": "a dog runs", "b.jpg": "a dog runs"}, refs,
                     train_captions={"t.jpg": ["A dog runs ."]})
    assert d["unique_caption_pct"] == 50.0 and d["novel_caption_pct"] == 0.0
