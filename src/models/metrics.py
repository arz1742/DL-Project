"""
src/models/metrics.py

Loss and accuracy definitions. Every metric says exactly what it counts, which
is the heart of the "accuracy illusion" comparison with Kavitha & Karpagam (2025):

  masked_acc    next-word accuracy on REAL words only (padding ignored) — honest
  top5_acc      correct next word is among the model's 5 highest guesses (padding ignored)
  unmasked_acc  plain Keras accuracy over EVERY position, padding included — the
                kind of number the paper reports. A model trained with the padded
                loss learns to predict <pad> after <end>, and since most of a
                34-token sequence is padding, this number is inflated.

Losses:
  masked_ce     cross-entropy averaged over real words only (default)
  padded_ce     cross-entropy over every position, padding included
                (used only by the paper-protocol replication run)
"""

import keras
from keras import ops

PAD_ID = 0


@keras.saving.register_keras_serializable(package="captioning")
def masked_ce(y_true, y_pred):
    y_true = ops.cast(y_true, "int32")
    ce = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    mask = ops.cast(ops.not_equal(y_true, PAD_ID), ce.dtype)
    return ops.sum(ce * mask) / ops.maximum(ops.sum(mask), 1.0)


@keras.saving.register_keras_serializable(package="captioning")
def padded_ce(y_true, y_pred):
    y_true = ops.cast(y_true, "int32")
    return ops.mean(keras.losses.sparse_categorical_crossentropy(y_true, y_pred))


@keras.saving.register_keras_serializable(package="captioning")
class TokenAccuracy(keras.metrics.Metric):
    """Top-k next-word accuracy, optionally ignoring padded positions."""

    def __init__(self, k: int = 1, ignore_padding: bool = True, name: str = "token_acc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.k = k
        self.ignore_padding = ignore_padding
        self.correct = self.add_variable(shape=(), initializer="zeros", name="correct")
        self.total = self.add_variable(shape=(), initializer="zeros", name="total")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.reshape(ops.cast(y_true, "int32"), (-1,))
        y_pred = ops.reshape(y_pred, (-1, ops.shape(y_pred)[-1]))
        hit = ops.cast(ops.in_top_k(y_true, y_pred, self.k), "float32")
        if self.ignore_padding:
            mask = ops.cast(ops.not_equal(y_true, PAD_ID), "float32")
        else:
            mask = ops.ones_like(hit)
        self.correct.assign_add(ops.sum(hit * mask))
        self.total.assign_add(ops.sum(mask))

    def result(self):
        return self.correct / ops.maximum(self.total, 1.0)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

    def get_config(self):
        return {**super().get_config(), "k": self.k, "ignore_padding": self.ignore_padding}


def standard_metrics():
    return [
        TokenAccuracy(k=1, ignore_padding=True, name="masked_acc"),
        TokenAccuracy(k=5, ignore_padding=True, name="top5_acc"),
        TokenAccuracy(k=1, ignore_padding=False, name="unmasked_acc"),
    ]
