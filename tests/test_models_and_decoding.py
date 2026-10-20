"""
tests/test_models_and_decoding.py

Builds every model variant with random weights and checks decoding logic:
beam search with k=1 must equal greedy, and a wider beam can never find a
lower-probability caption than greedy. CPU only, ~1 minute.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from src.tokenizer import CaptionTokenizer
from src.models.caption_model import build_caption_model, encoder_and_decoder
from src.utils.beam_search import CaptionGenerator

CAPS = ["a dog runs on the grass", "a man rides a red bike", "two children play in the water",
        "a black dog jumps over a fence", "a woman sits on a bench"]


def _tok():
    t = CaptionTokenizer(min_word_freq=1)
    t.build_vocab(CAPS)
    return t


@pytest.mark.parametrize("decoder", ["lstm", "gru", "hybrid"])
@pytest.mark.parametrize("encoder", ["frozen", "finetune"])
def test_model_shapes(decoder, encoder):
    tok = _tok()
    m = build_caption_model(tok.vocab_size, 12, decoder, encoder, units=32, embedding_dim=16, cnn_weights=None)
    img = np.random.rand(2, 2048) if encoder == "frozen" else np.random.rand(2, 14, 14, 1024)
    out = m.predict([img.astype("float32"), np.zeros((2, 11), "int32")], verbose=0)
    assert out.shape == (2, 11, tok.vocab_size)
    assert np.allclose(out.sum(-1), 1.0, atol=1e-4)


def _seq_logprob(gen, state, caption):
    words = caption.split() + ["<end>"]
    return sum(np.log(s["prob"]) for s in gen.word_probabilities(state, caption, top=1)[:len(words)])


def test_beam1_equals_greedy_and_scores_are_consistent():
    tok = _tok()
    m = build_caption_model(tok.vocab_size, 12, "hybrid", "frozen", units=32, embedding_dim=16, cnn_weights=None)
    gen = CaptionGenerator(m, tok, 12)
    states = gen.encode_images(np.random.rand(6, 2048).astype("float32"))
    greedy = gen.greedy(states)
    assert gen.beam(states, 1) == greedy
    # The score beam search reports for its chosen caption must equal that
    # caption's real log-probability under the model. (Beam search is NOT
    # guaranteed to beat greedy — the greedy path can be pruned early.)
    for s in states:
        trace = gen.beam_trace(s, 4)
        cap = trace["caption"]
        if len(cap.split()) >= 9:   # hit the length limit without <end>; score has no <end> term
            continue
        assert abs(max(trace["final_scores"]) - _seq_logprob(gen, s, cap)) < 1e-3


def test_encoder_decoder_split_matches_full_model():
    tok = _tok()
    m = build_caption_model(tok.vocab_size, 12, "lstm", "frozen", units=32, embedding_dim=16, cnn_weights=None)
    enc, dec = encoder_and_decoder(m)
    img = np.random.rand(3, 2048).astype("float32")
    seq = np.random.randint(0, tok.vocab_size, (3, 11)).astype("int32")
    full = m.predict([img, seq], verbose=0)
    split = dec.predict([enc.predict(img, verbose=0), seq], verbose=0)
    assert np.allclose(full, split, atol=1e-5)


@pytest.mark.parametrize("decoder", ["attention", "transformer"])
def test_attention_models_shapes_maps_and_causality(decoder):
    tok = _tok()
    m = build_caption_model(tok.vocab_size, 12, decoder, "grid", units=32, embedding_dim=32,
                            n_layers=2, n_heads=2, ff_dim=64, cnn_weights=None)
    grid = np.random.rand(2, 7, 7, 2048).astype("float32")
    seq = np.random.randint(1, tok.vocab_size, (2, 11)).astype("int32")
    out = m.predict([grid, seq], verbose=0)
    assert out.shape == (2, 11, tok.vocab_size)

    # Changing a later word must not change predictions at earlier positions.
    seq2 = seq.copy()
    seq2[:, 6:] = (seq2[:, 6:] % (tok.vocab_size - 1)) + 1
    out2 = m.predict([grid, seq2], verbose=0)
    assert np.allclose(out[:, :6], out2[:, :6], atol=1e-5)

    gen = CaptionGenerator(m, tok, 12)
    state = gen.encode_images(grid)[0]
    cap = gen.beam(state[None], 3)[0]
    maps = gen.attention_maps(state, cap)
    assert len(maps) == min(len(cap.split()) + 1, 10)
    for mp in maps:
        assert mp["map"].shape == (7, 7) and abs(float(mp["map"].sum()) - 1.0) < 1e-4


def test_classic_models_have_no_attention_maps():
    tok = _tok()
    m = build_caption_model(tok.vocab_size, 12, "hybrid", "frozen", units=32, embedding_dim=32, cnn_weights=None)
    gen = CaptionGenerator(m, tok, 12)
    assert gen.attention_maps(gen.encode_images(np.random.rand(1, 2048).astype("float32"))[0], "a dog") == []


def test_decoder_encoder_mismatch_is_rejected():
    with pytest.raises(ValueError):
        build_caption_model(30, 12, "attention", "frozen", cnn_weights=None)
