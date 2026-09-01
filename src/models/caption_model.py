"""
src/models/caption_model.py

Ties the encoder bridge and decoder (LSTM or GRU) into a single trainable
model, and provides the factory function used by train.py to build any of
the four ablation configurations from one place.
"""

from tensorflow.keras import layers, Model, optimizers

from src.models.encoder import build_bridge_layer
from src.models.decoder_lstm import build_lstm_decoder
from src.models.decoder_gru import build_gru_decoder


def build_caption_model(
    vocab_size: int,
    max_length: int,
    decoder_type: str = "lstm",
    embedding_dim: int = 256,
    units: int = 256,
    learning_rate: float = 1e-3,
) -> Model:
    """
    decoder_type: "lstm" or "gru" — selects which decoder module to use.
    Note: embedding_dim must equal `units` so the bridged feature vector
    can be used directly as the recurrent layer's initial state.
    """
    if embedding_dim != units:
        raise ValueError(
            "embedding_dim must equal units, since the bridged image "
            "feature is used directly as the decoder's initial state."
        )

    feature_input = layers.Input(shape=(2048,), name="raw_cnn_feature")
    bridge = build_bridge_layer(embedding_dim=embedding_dim)
    bridged_feature = bridge(feature_input)

    if decoder_type == "lstm":
        decoder = build_lstm_decoder(vocab_size, max_length, embedding_dim, units)
    elif decoder_type == "gru":
        decoder = build_gru_decoder(vocab_size, max_length, embedding_dim, units)
    else:
        raise ValueError(f"Unknown decoder_type: {decoder_type} (expected 'lstm' or 'gru')")

    caption_input = layers.Input(shape=(max_length,), name="caption_sequence")
    output = decoder([bridged_feature, caption_input])

    model = Model(
        inputs=[feature_input, caption_input],
        outputs=output,
        name=f"image_captioning_{decoder_type}",
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    # Smoke test — builds both variants and prints a summary.
    # Safe to run on CPU; just checks the graph is wired correctly.
    for dtype in ["lstm", "gru"]:
        print(f"\n=== Building {dtype.upper()} caption model ===")
        m = build_caption_model(vocab_size=5000, max_length=34, decoder_type=dtype)
        m.summary()
