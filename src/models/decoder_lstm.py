"""
src/models/decoder_lstm.py

LSTM decoder for caption generation, following the Show and Tell (Vinyals
et al., 2015) design: the CNN feature initializes the decoder state, and
an embedding + LSTM stack predicts the caption word by word.
"""

from tensorflow.keras import layers, Model


def build_lstm_decoder(
    vocab_size: int,
    max_length: int,
    embedding_dim: int = 256,
    lstm_units: int = 256,
) -> Model:
    # Image feature branch (already bridged to embedding_dim by encoder.py)
    feature_input = layers.Input(shape=(embedding_dim,), name="image_features")

    # Caption sequence branch
    caption_input = layers.Input(shape=(max_length,), name="caption_sequence")
    x = layers.Embedding(vocab_size, embedding_dim, mask_zero=True, name="word_embedding")(caption_input)
    x = layers.Dropout(0.5)(x)

    # Use image feature as the initial hidden/cell state of the LSTM
    initial_state = [feature_input, feature_input]
    lstm_out = layers.LSTM(lstm_units, return_sequences=False, name="lstm_decoder")(
        x, initial_state=initial_state
    )
    lstm_out = layers.Dropout(0.5)(lstm_out)

    output = layers.Dense(vocab_size, activation="softmax", name="word_predictions")(lstm_out)

    model = Model(inputs=[feature_input, caption_input], outputs=output, name="lstm_caption_decoder")
    return model
