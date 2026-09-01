"""
src/models/encoder.py

Thin wrapper around the ResNet50 encoder, plus the "bridge" dense layer
that projects the 2048-d CNN feature into the decoder's embedding space.
"""

from tensorflow.keras import layers, Model


def build_bridge_layer(embedding_dim: int, feature_dim: int = 2048) -> Model:
    """
    Projects a precomputed ResNet50 feature vector (2048-d) down to the
    decoder's embedding dimension, and applies it as the initial hidden
    state — mirroring the Show & Tell / LRCN encoder-decoder bridge.
    """
    feature_input = layers.Input(shape=(feature_dim,), name="cnn_feature_input")
    x = layers.Dense(embedding_dim, activation="relu", name="bridge_dense")(feature_input)
    x = layers.Dropout(0.5, name="bridge_dropout")(x)
    return Model(inputs=feature_input, outputs=x, name="encoder_bridge")
