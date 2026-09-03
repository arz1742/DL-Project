"""
src/models/decoder_gru.py

GRU decoder for caption generation — architecturally identical to the
LSTM decoder except for the recurrent unit, so that the LSTM-vs-GRU
ablation study isolates just this one variable.
"""

from tensorflow.keras import layers, Model


def build_gru_decoder(
    vocab_size: int,
    max_length: int,
    embedding_dim: int = 256,
    gru_units: int = 256,
) -> Model:
    feature_input = layers.Input(shape=(embedding_dim,), name="image_features")

    caption_input = layers.Input(shape=(max_length,), name="caption_sequence")
    x = layers.Embedding(vocab_size, embedding_dim, mask_zero=True, name="word_embedding")(caption_input)
    x = layers.Dropout(0.5)(x)

    # GRU only needs a single initial state (no separate cell state, unlike LSTM)
    gru_out = layers.GRU(gru_units, return_sequences=True, name="gru_decoder")(
        x, initial_state=feature_input
    )
    gru_out = layers.Dropout(0.5)(gru_out)

    output = layers.Dense(vocab_size, activation="softmax", name="word_predictions")(gru_out)

    model = Model(inputs=[feature_input, caption_input], outputs=output, name="gru_caption_decoder")
    return model
