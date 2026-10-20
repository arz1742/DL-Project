"""
src/models/caption_model.py

Builds the full captioning model for any ablation config from one place:

    image  ->  image_encoder  ->  state (units,)
                                     |
    tokens ->  text_decoder([state, tokens])  ->  next-word probabilities (T, V)

The two halves are kept as named sub-models so decoding can run the (possibly
expensive) image encoder ONCE per image and then call only the decoder for
each generated word.
"""

import keras

from src.models.encoder import build_image_encoder
from src.models.decoders import build_decoder
from src.models.metrics import masked_ce, padded_ce, standard_metrics


def build_caption_model(
    vocab_size: int,
    max_length: int = 34,
    decoder_type: str = "lstm",
    encoder_mode: str = "frozen",
    embedding_dim: int = 256,
    units: int = 256,
    dropout: float = 0.5,
    embedding_matrix=None,
    train_embeddings: bool = True,
    cnn_weights: str | None = "imagenet",
    n_layers: int = 3,
    n_heads: int = 4,
    ff_dim: int = 1024,
) -> keras.Model:
    """
    max_length is the length of an encoded caption including <start>/<end>
    and padding. The model reads tokens[:-1] and predicts tokens[1:], so its
    sequence length is max_length - 1.
    """
    seq_len = max_length - 1
    encoder = build_image_encoder(encoder_mode, units, dropout, cnn_weights)
    from src.models.decoders import ATTENTION_DECODERS
    if (decoder_type in ATTENTION_DECODERS) != (encoder_mode == "grid"):
        raise ValueError(f"decoder {decoder_type!r} needs encoder_mode 'grid'; classic decoders "
                         f"need 'frozen' or 'finetune' (got {encoder_mode!r})")
    decoder = build_decoder(decoder_type, vocab_size, seq_len, embedding_dim, units,
                            dropout, embedding_matrix, train_embeddings, n_layers, n_heads, ff_dim)

    img = keras.Input(shape=encoder.input.shape[1:], name="image")
    tokens = keras.Input(shape=(seq_len,), dtype="int32", name="tokens")
    probs = decoder([encoder(img), tokens])
    return keras.Model([img, tokens], probs, name=f"captioner_{decoder_type}_{encoder_mode}")


def compile_caption_model(model: keras.Model, learning_rate: float = 1e-3,
                          mask_padding_in_loss: bool = True, clipnorm: float | None = 5.0):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm),
        loss=masked_ce if mask_padding_in_loss else padded_ce,
        metrics=standard_metrics(),
    )
    return model


def encoder_and_decoder(model: keras.Model) -> tuple[keras.Model, keras.Model]:
    return model.get_layer("image_encoder"), model.get_layer("text_decoder")


def count_params(model: keras.Model) -> dict:
    import numpy as np
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    total = int(sum(np.prod(w.shape) for w in model.weights))
    return {"trainable": trainable, "total": total}


if __name__ == "__main__":
    # Smoke test: builds every decoder x encoder combination (random CNN weights).
    for enc, decs in [("frozen", ["lstm", "gru", "hybrid"]), ("finetune", ["lstm", "gru", "hybrid"]),
                      ("grid", ["attention", "transformer"])]:
        for dec in decs:
            m = build_caption_model(3000, 34, dec, enc, cnn_weights=None)
            print(f"{dec:11s} {enc:8s} {count_params(m)}")
