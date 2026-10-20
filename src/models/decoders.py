"""
src/models/decoders.py

The three recurrent decoders compared in the ablation study. They are
identical except for the recurrent part, so the comparison isolates exactly
one design choice:

    lstm   : Embedding -> LSTM            -> Dense(softmax)
    gru    : Embedding -> GRU             -> Dense(softmax)
    hybrid : Embedding -> LSTM -> GRU     -> Dense(softmax)   (Kavitha & Karpagam 2025)

Inputs : [image_state (units,), tokens (T,)]
Output : next-word probabilities at EVERY position, shape (T, vocab_size).

The image state initialises the recurrent layer(s) — the "Show and Tell"
injection used by both reference papers. Training uses teacher forcing on
the whole caption in one pass (token t predicts token t+1).
"""

import keras
from keras import layers

DECODER_TYPES = ("lstm", "gru", "hybrid")


def build_decoder(decoder_type: str, vocab_size: int, seq_len: int,
                  embedding_dim: int = 256, units: int = 256, dropout: float = 0.5,
                  embedding_matrix=None, train_embeddings: bool = True) -> keras.Model:
    """
    embedding_matrix: optional (vocab_size, embedding_dim) array, e.g. GloVe
    vectors, used to initialise the word embeddings (experiment E3).
    """
    if decoder_type not in DECODER_TYPES:
        raise ValueError(f"Unknown decoder_type {decoder_type!r}, expected one of {DECODER_TYPES}")

    state_in = keras.Input(shape=(units,), name="image_state")
    tokens_in = keras.Input(shape=(seq_len,), dtype="int32", name="caption_tokens")

    emb_kwargs = {}
    if embedding_matrix is not None:
        emb_kwargs["embeddings_initializer"] = keras.initializers.Constant(embedding_matrix)
    x = layers.Embedding(vocab_size, embedding_dim, name="word_embedding",
                         trainable=train_embeddings, **emb_kwargs)(tokens_in)
    x = layers.Dropout(dropout, name="embedding_dropout")(x)

    # Captions are right-padded and the RNNs are causal, so outputs at real
    # positions never see padding; the loss ignores padded positions instead
    # of using Keras masks (this also keeps the fast cuDNN kernels).
    if decoder_type == "lstm":
        x = layers.LSTM(units, return_sequences=True, name="lstm")(x, initial_state=[state_in, state_in])
    elif decoder_type == "gru":
        x = layers.GRU(units, return_sequences=True, name="gru")(x, initial_state=state_in)
    else:  # hybrid: LSTM output feeds a GRU layer
        x = layers.LSTM(units, return_sequences=True, name="lstm")(x, initial_state=[state_in, state_in])
        x = layers.GRU(units, return_sequences=True, name="gru")(x, initial_state=state_in)

    x = layers.Dropout(dropout, name="decoder_dropout")(x)
    probs = layers.Dense(vocab_size, activation="softmax", name="word_probs")(x)
    return keras.Model([state_in, tokens_in], probs, name="text_decoder")
