"""
tokenizer.py

Handles text preprocessing and vocabulary building for the Flickr8k captions.
No GPU required — safe to run on any machine.
"""

import json
import re
from collections import Counter


START_TOKEN = "<start>"
END_TOKEN = "<end>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def clean_caption(text: str) -> str:
    """Lowercase, strip punctuation/numbers, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class CaptionTokenizer:
    """
    Simple word-level tokenizer with a frequency-based vocabulary.
    Mirrors the preprocessing style used in both mother papers
    (lowercasing, punctuation removal, unique-word vocabulary).
    """

    def __init__(self, min_word_freq: int = 5):
        self.min_word_freq = min_word_freq
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

    def build_vocab(self, captions: list[str]) -> None:
        counter = Counter()
        for cap in captions:
            counter.update(clean_caption(cap).split())

        # Reserve indices 0-3 for special tokens
        special_tokens = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]
        vocab_words = [w for w, freq in counter.items() if freq >= self.min_word_freq]

        all_tokens = special_tokens + sorted(vocab_words)
        self.word2idx = {word: idx for idx, word in enumerate(all_tokens)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

        print(f"Vocabulary built: {self.vocab_size} tokens "
              f"(min_word_freq={self.min_word_freq})")

    def encode(self, caption: str, max_length: int) -> list[int]:
        """Convert a raw caption string into a padded list of token ids."""
        words = clean_caption(caption).split()
        ids = [self.word2idx.get(START_TOKEN)]
        ids += [self.word2idx.get(w, self.word2idx[UNK_TOKEN]) for w in words]
        ids.append(self.word2idx[END_TOKEN])

        if len(ids) < max_length:
            ids += [self.word2idx[PAD_TOKEN]] * (max_length - len(ids))
        else:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: list[int], strip_special: bool = True) -> str:
        """Convert token ids back into a readable sentence."""
        words = [self.idx2word.get(i, UNK_TOKEN) for i in ids]
        if strip_special:
            words = [w for w in words if w not in
                     (START_TOKEN, END_TOKEN, PAD_TOKEN)]
        return " ".join(words)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({
                "word2idx": self.word2idx,
                "min_word_freq": self.min_word_freq,
            }, f, indent=2)
        print(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "CaptionTokenizer":
        with open(path, "r") as f:
            data = json.load(f)
        tok = cls(min_word_freq=data["min_word_freq"])
        tok.word2idx = data["word2idx"]
        tok.idx2word = {int(idx): word for word, idx in tok.word2idx.items()}
        tok.vocab_size = len(tok.word2idx)
        return tok


if __name__ == "__main__":
    # Quick smoke test — run directly to sanity-check tokenizer behaviour
    sample_captions = [
        "A black dog runs through the grass .",
        "A brown dog runs across a grassy field .",
        "A dog is running in the grass .",
    ]
    tok = CaptionTokenizer(min_word_freq=1)
    tok.build_vocab(sample_captions)
    encoded = tok.encode(sample_captions[0], max_length=15)
    print("Encoded:", encoded)
    print("Decoded:", tok.decode(encoded))
