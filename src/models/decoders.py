"""
src/models/decoders.py

Decoders compared in the project.

Part 1 (classic, in syllabus) — identical except for the recurrent part:
    lstm        : Embedding -> LSTM            -> Dense(softmax)
    gru         : Embedding -> GRU             -> Dense(softmax)
    hybrid      : Embedding -> LSTM -> GRU     -> Dense(softmax)   (Kavitha & Karpagam 2025)
  Input: [image_state (units,), tokens (T,)]. The image state initialises the
  recurrent layer(s) — the "Show and Tell" injection.

Part 2 (attention, beyond the syllabus) — input: [regions (49, units), tokens (T,)]:
    attention   : LSTM + soft attention over the 7x7 image regions. At every
                  word the LSTM output asks "which regions matter now?", gets a
                  weighted average of the regions (the context) and predicts the
                  next word from both. Luong-style attention (Luong et al. 2015),
                  the same idea as Show, Attend and Tell (Xu et al. 2015) but
                  computed after the LSTM so the whole caption trains in one pass.
    transformer : a small Transformer decoder (Vaswani et al. 2017): masked
                  self-attention over the words so far + cross-attention over the
                  image regions, stacked `layers` times.

Both attention decoders expose a layer named "attention_map" whose output,
shape (T, 49), says how much each word looked at each region — used for the
attention heatmaps.

Output for every decoder: next-word probabilities at every position, (T, vocab_size).
"""

import keras
from keras import layers, ops

CLASSIC_DECODERS = ("lstm", "gru", "hybrid")
ATTENTION_DECODERS = ("attention", "transformer")
DECODER_TYPES = CLASSIC_DECODERS + ATTENTION_DECODERS
N_REGIONS = 49


@keras.saving.register_keras_serializable(package="captioning")
class PositionEmbedding(layers.Layer):
    """Adds a learned embedding for each position 0..length-1."""

    def __init__(self, length: int, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.length, self.dim = length, dim
        self.pos = layers.Embedding(length, dim)

    def build(self, input_shape):
        self.pos.build((None, self.length))
        super().build(input_shape)

    def call(self, x):
        return x + self.pos(ops.arange(self.length))[None]

    def get_config(self):
        return {**super().get_config(), "length": self.length, "dim": self.dim}


@keras.saving.register_keras_serializable(package="captioning")
class HeadMean(layers.Layer):
    """(batch, heads, T, regions) attention scores -> mean over heads (batch, T, regions)."""

    def call(self, x):
        return ops.mean(x, axis=1)


def _word_embedding(tokens_in, vocab_size, embedding_dim, embedding_matrix, train_embeddings):
    kwargs = {}
    if embedding_matrix is not None:
        kwargs["embeddings_initializer"] = keras.initializers.Constant(embedding_matrix)
    return layers.Embedding(vocab_size, embedding_dim, name="word_embedding",
                            trainable=train_embeddings, **kwargs)(tokens_in)


def build_decoder(decoder_type: str, vocab_size: int, seq_len: int,
                  embedding_dim: int = 256, units: int = 256, dropout: float = 0.5,
                  embedding_matrix=None, train_embeddings: bool = True,
                  n_layers: int = 3, n_heads: int = 4, ff_dim: int = 1024) -> keras.Model:
    """
    embedding_matrix: optional (vocab_size, embedding_dim) array, e.g. GloVe
    vectors, used to initialise the word embeddings.
    n_layers / n_heads / ff_dim: Transformer only.
    """
    if decoder_type not in DECODER_TYPES:
        raise ValueError(f"Unknown decoder_type {decoder_type!r}, expected one of {DECODER_TYPES}")

    tokens_in = keras.Input(shape=(seq_len,), dtype="int32", name="caption_tokens")
    if decoder_type in CLASSIC_DECODERS:
        state_in = keras.Input(shape=(units,), name="image_state")
    else:
        state_in = keras.Input(shape=(N_REGIONS, units), name="image_regions")

    x = _word_embedding(tokens_in, vocab_size, embedding_dim, embedding_matrix, train_embeddings)

    # Captions are right-padded and every decoder is causal, so outputs at real
    # positions never see padding; the loss ignores padded positions instead
    # of using Keras masks (this also keeps the fast cuDNN kernels).
    if decoder_type == "lstm":
        x = layers.Dropout(dropout, name="embedding_dropout")(x)
        x = layers.LSTM(units, return_sequences=True, name="lstm")(x, initial_state=[state_in, state_in])
    elif decoder_type == "gru":
        x = layers.Dropout(dropout, name="embedding_dropout")(x)
        x = layers.GRU(units, return_sequences=True, name="gru")(x, initial_state=state_in)
    elif decoder_type == "hybrid":  # LSTM output feeds a GRU layer
        x = layers.Dropout(dropout, name="embedding_dropout")(x)
        x = layers.LSTM(units, return_sequences=True, name="lstm")(x, initial_state=[state_in, state_in])
        x = layers.GRU(units, return_sequences=True, name="gru")(x, initial_state=state_in)
    elif decoder_type == "attention":
        x = layers.Dropout(dropout, name="embedding_dropout")(x)
        mean_region = layers.GlobalAveragePooling1D(name="mean_region")(state_in)
        h0 = layers.Dense(units, activation="tanh", name="init_h")(mean_region)
        c0 = layers.Dense(units, activation="tanh", name="init_c")(mean_region)
        h = layers.LSTM(units, return_sequences=True, name="lstm")(x, initial_state=[h0, c0])
        query = layers.Dense(units, use_bias=False, name="attention_query")(h)          # (T, u)
        scores = layers.Dot(axes=(2, 2), name="attention_scores")([query, state_in])    # (T, 49)
        scores = layers.Rescaling(1.0 / units ** 0.5, name="attention_scale")(scores)   # scaled dot product
        alpha = layers.Softmax(axis=-1, name="attention_map")(scores)                   # (T, 49)
        context = layers.Dot(axes=(2, 1), name="attention_context")([alpha, state_in])  # (T, u)
        x = layers.Concatenate(name="lstm_and_context")([h, context])
        x = layers.Dense(units, activation="tanh", name="attention_combine")(x)
    else:  # transformer
        if embedding_dim != units:
            raise ValueError("The Transformer needs embedding_dim == units (residual connections).")
        x = PositionEmbedding(seq_len, units, name="word_positions")(x)
        regions = PositionEmbedding(N_REGIONS, units, name="region_positions")(state_in)
        x = layers.Dropout(dropout, name="embedding_dropout")(x)
        key_dim = units // n_heads
        for i in range(n_layers):
            y = layers.LayerNormalization(name=f"block{i}_norm1")(x)
            y = layers.MultiHeadAttention(n_heads, key_dim, dropout=dropout, name=f"block{i}_self_attention")(
                y, y, use_causal_mask=True)
            x = layers.Add(name=f"block{i}_add1")([x, layers.Dropout(dropout)(y)])
            y = layers.LayerNormalization(name=f"block{i}_norm2")(x)
            y, scores = layers.MultiHeadAttention(n_heads, key_dim, dropout=dropout,
                                                  name=f"block{i}_cross_attention")(
                y, regions, return_attention_scores=True)
            x = layers.Add(name=f"block{i}_add2")([x, layers.Dropout(dropout)(y)])
            y = layers.LayerNormalization(name=f"block{i}_norm3")(x)
            y = layers.Dense(ff_dim, activation="gelu", name=f"block{i}_ff1")(y)
            y = layers.Dense(units, name=f"block{i}_ff2")(y)
            x = layers.Add(name=f"block{i}_add3")([x, layers.Dropout(dropout)(y)])
        attention_map = HeadMean(name="attention_map")(scores)  # last block, averaged over heads
        x = layers.LayerNormalization(name="final_norm")(x)

    x = layers.Dropout(dropout, name="decoder_dropout")(x)
    probs = layers.Dense(vocab_size, activation="softmax", name="word_probs")(x)
    decoder = keras.Model([state_in, tokens_in], probs, name="text_decoder")
    if decoder_type == "transformer":
        # Not on the path to the output, so Keras leaves it out of the model's
        # layers; keep the tensor so attention_probe can still reach it.
        decoder._attention_map_tensor = attention_map
    return decoder


def attention_probe(decoder: keras.Model):
    """[regions, tokens] -> attention weights (T, 49), or None for classic decoders."""
    tensor = getattr(decoder, "_attention_map_tensor", None)
    if tensor is None:
        try:
            tensor = decoder.get_layer("attention_map").output
        except ValueError:
            return None
    return keras.Model(decoder.inputs, tensor, name="attention_probe")
