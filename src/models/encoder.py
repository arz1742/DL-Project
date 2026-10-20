"""
src/models/encoder.py

Image side of the captioning model.

  frozen   : input = precomputed pooled ResNet50 vector (2048,)
             -> bridge (Dense + dropout) -> decoder initial state
  finetune : input = precomputed ResNet50 stage-4 map (14, 14, 1024)
             -> ResNet50 conv5 block (TRAINABLE, ImageNet-initialised)
             -> global average pool (2048,) -> bridge -> decoder initial state

  grid     : input = precomputed ResNet50 final map (7, 7, 2048)
             -> 49 region vectors -> Dense projection (49, units)
             Used by the attention decoders (Part 2): the decoder sees every
             region and decides which to look at for each word.

The frozen and finetune paths produce the same 2048-d vector before the
bridge, so a model trained with the frozen encoder can be loaded straight
into the fine-tune model and training continues from there (two-stage
fine-tuning).
"""

import keras
from keras import layers

CONV4_LAYER = "conv4_block6_out"


def build_bridge_layer(units: int, feature_dim: int = 2048, dropout: float = 0.5) -> keras.Model:
    """Projects the 2048-d CNN vector to the decoder's hidden size."""
    feature_input = keras.Input(shape=(feature_dim,), name="cnn_feature_input")
    x = layers.Dense(units, activation="relu", name="bridge_dense")(feature_input)
    x = layers.Dropout(dropout, name="bridge_dropout")(x)
    return keras.Model(feature_input, x, name="encoder_bridge")


def build_conv5_head(weights: str | None = "imagenet") -> keras.Model:
    """
    ResNet50's last block (conv5_x) + global average pooling, as a standalone
    model taking the stage-4 map (14, 14, 1024) and returning (2048,).
    Weights come from ImageNet. BatchNorm layers stay frozen (standard practice
    when fine-tuning with small batches); conv kernels are trainable.
    """
    from keras.applications.resnet50 import ResNet50

    full = ResNet50(weights=weights, include_top=False, pooling="avg",
                    input_shape=(224, 224, 3))
    head = keras.Model(full.get_layer(CONV4_LAYER).output, full.output, name="resnet50_conv5")
    head.trainable = True
    for layer in head.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return head


def build_image_encoder(encoder_mode: str, units: int, dropout: float = 0.5,
                        cnn_weights: str | None = "imagenet") -> keras.Model:
    """Image input -> decoder initial state vector of size `units`."""
    bridge = build_bridge_layer(units, dropout=dropout)
    if encoder_mode == "frozen":
        img = keras.Input(shape=(2048,), name="image_input")
        state = bridge(img)
    elif encoder_mode == "grid":
        img = keras.Input(shape=(7, 7, 2048), name="image_input")
        x = layers.Reshape((49, 2048), name="regions")(img)
        x = layers.Dense(units, name="region_projection")(x)
        # Raw ResNet values are large and unnormalised; LayerNorm keeps the
        # attention softmax and the decoder's starting state out of saturation.
        x = layers.LayerNormalization(name="region_norm")(x)
        state = layers.Dropout(dropout, name="region_dropout")(x)
    elif encoder_mode == "finetune":
        img = keras.Input(shape=(14, 14, 1024), name="image_input")
        # training=False keeps BatchNorm on its ImageNet statistics;
        # gradients still flow into the conv kernels.
        pooled = build_conv5_head(cnn_weights)(img, training=False)
        state = bridge(pooled)
    else:
        raise ValueError(f"Unknown encoder_mode {encoder_mode!r} (expected 'frozen', 'finetune' or 'grid')")
    return keras.Model(img, state, name="image_encoder")
