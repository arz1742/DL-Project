"""tests/test_analysis.py — hallucination, error labels, t-SNE and Grad-CAM. CPU only."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src import analysis as an
from src.tokenizer import CaptionTokenizer
from src.models.caption_model import build_caption_model


def test_person_words_are_not_hallucinations_but_new_objects_are():
    refs = {"a": ["a person rides a bicycle on the road"] * 5}
    h = an.hallucination_scores({"a": "a man rides a bike near the water"}, refs)
    assert h["top_hallucinated"] == [("water", 1)]
    assert h["chair_s"] == 1.0 and abs(h["chair_i"] - 1 / 3) < 1e-9   # person, bike, water


def test_error_labels():
    refs = ["two black dogs run in the grass"] * 5
    labels = an.error_labels("a white cat cat sits on a bench", refs, "dog", set())
    assert {"wrong_subject", "hallucination", "wrong_colour", "repetition"} <= set(labels)
    assert an.error_labels("two black dogs run in the grass", refs, "dog", set()) == []
    assert "wrong_count" in an.error_labels("three black dogs run", refs, "dog", set())


def test_tsne_shape():
    xy = an.tsne_coordinates(np.random.rand(30, 64))
    assert xy.shape == (30, 2)


def test_gradcam_maps_are_normalised():
    tok = CaptionTokenizer(min_word_freq=1)
    tok.build_vocab(["a dog runs on the grass"])
    m = build_caption_model(tok.vocab_size, 12, "hybrid", "frozen", units=32, embedding_dim=32, cnn_weights=None)
    maps = an.gradcam_maps(m, np.random.rand(7, 7, 2048), "a dog runs", tok, 12)
    assert [x["word"] for x in maps] == ["a", "dog", "runs", "<end>"]
    for x in maps:
        assert x["map"].shape == (7, 7) and abs(x["map"].sum() - 1) < 1e-5 and (x["map"] >= 0).all()
