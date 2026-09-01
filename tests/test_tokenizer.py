"""
tests/test_tokenizer.py

Basic sanity tests for the tokenizer — run with: pytest tests/
No GPU or dataset needed.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tokenizer import CaptionTokenizer, clean_caption


def test_clean_caption_lowercases_and_strips_punctuation():
    result = clean_caption("A Dog Runs, Fast!  ")
    assert result == "a dog runs fast"


def test_vocab_respects_min_word_freq():
    captions = ["a dog runs", "a dog jumps", "a cat sleeps"]
    tok = CaptionTokenizer(min_word_freq=2)
    tok.build_vocab(captions)
    # "a" and "dog" appear twice, "cat"/"sleeps"/"jumps"/"runs" appear once
    assert "a" in tok.word2idx
    assert "dog" in tok.word2idx
    assert "cat" not in tok.word2idx  # below min frequency, should map to <unk> at encode time


def test_encode_decode_roundtrip():
    captions = ["a black dog runs through the grass"]
    tok = CaptionTokenizer(min_word_freq=1)
    tok.build_vocab(captions)

    encoded = tok.encode(captions[0], max_length=15)
    decoded = tok.decode(encoded)

    assert decoded == clean_caption(captions[0])


def test_encode_pads_to_max_length():
    captions = ["a short caption"]
    tok = CaptionTokenizer(min_word_freq=1)
    tok.build_vocab(captions)

    encoded = tok.encode(captions[0], max_length=20)
    assert len(encoded) == 20


def test_encode_truncates_long_captions():
    captions = ["one two three four five six seven eight nine ten"]
    tok = CaptionTokenizer(min_word_freq=1)
    tok.build_vocab(captions)

    encoded = tok.encode(captions[0], max_length=5)
    assert len(encoded) == 5


def test_unknown_word_maps_to_unk():
    tok = CaptionTokenizer(min_word_freq=1)
    tok.build_vocab(["a dog runs"])

    encoded = tok.encode("a spaceship flies", max_length=10)
    unk_id = tok.word2idx["<unk>"]
    # "spaceship" and "flies" were never seen, so both should map to <unk>
    assert unk_id in encoded


if __name__ == "__main__":
    test_clean_caption_lowercases_and_strips_punctuation()
    test_vocab_respects_min_word_freq()
    test_encode_decode_roundtrip()
    test_encode_pads_to_max_length()
    test_encode_truncates_long_captions()
    test_unknown_word_maps_to_unk()
    print("All tokenizer tests passed.")
