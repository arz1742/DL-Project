"""
src/dataset.py

Feeds (image feature, caption) pairs to Keras in shuffled batches.

Every (image, caption) pair is one training example, so each epoch sees all
5 captions of every training image. A caption encoded as
    [<start>, w1, ..., wn, <end>, <pad>, ...]          (length max_length)
becomes
    input  = encoded[:-1]   ->  [<start>, w1, ..., wn, <end>, <pad>...]
    target = encoded[1:]    ->  [w1, ..., wn, <end>, <pad>, ...]
i.e. teacher forcing over the whole caption in a single forward pass.
"""

import numpy as np
import keras


def encode_split(captions: dict[str, list[str]], tokenizer, max_length: int,
                 name_to_row: dict[str, int]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns (feature_rows (N,), tokens (N, max_length), image_names (N,)) for
    every caption of every image that has features.
    """
    rows, tokens, names = [], [], []
    for image_name in sorted(captions):
        if image_name not in name_to_row:
            continue
        for cap in captions[image_name]:
            rows.append(name_to_row[image_name])
            tokens.append(tokenizer.encode(cap, max_length))
            names.append(image_name)
    if not rows:
        raise ValueError("No captions matched any extracted image features.")
    return np.asarray(rows, np.int64), np.asarray(tokens, np.int32), names


class CaptionBatches(keras.utils.PyDataset):
    def __init__(self, features: np.ndarray, rows: np.ndarray, tokens: np.ndarray,
                 batch_size: int = 32, shuffle: bool = True, seed: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.features = features
        self.rows = rows
        self.tokens = tokens
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(rows))
        if shuffle:
            self.rng.shuffle(self.order)

    def __len__(self):
        return int(np.ceil(len(self.order) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.order[idx * self.batch_size:(idx + 1) * self.batch_size]
        feature_rows = self.rows[batch]
        # Fancy-index sorted rows (faster on memory-mapped arrays), then restore order.
        uniq, inverse = np.unique(feature_rows, return_inverse=True)
        img = np.asarray(self.features[uniq], dtype=np.float32)[inverse]
        tok = self.tokens[batch]
        return (img, tok[:, :-1]), tok[:, 1:]

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)
