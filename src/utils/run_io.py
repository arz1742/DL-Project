"""
src/utils/run_io.py

Configs, run folders and model loading shared by train.py, evaluate.py,
inference.py and the dashboard.

Every training run lives in its own folder:

    runs/<config_name>/seed<seed>/
        config.yaml         the exact settings used
        state.json          progress: last epoch, best epoch, early-stopping counter, done flag
        history.json        per-epoch train/val loss and accuracies
        last.weights.h5     latest epoch (used to resume after a Colab disconnect)
        best.weights.h5     lowest validation loss (used for evaluation)
        eval/               written by evaluate.py
"""

import os
import json

import numpy as np
import yaml

DEFAULTS = {
    "config_name": None,
    "decoder": "lstm",              # lstm | gru | hybrid | attention | transformer
    "encoder_mode": "frozen",       # frozen | finetune | grid (attention decoders)
    "max_length": 34,
    "embedding_dim": 256,
    "units": 256,
    "dropout": 0.5,
    "batch_size": 32,
    "epochs": 40,
    "learning_rate": 0.001,
    "early_stopping_patience": 5,   # null = train all epochs (paper protocol)
    "mask_padding_in_loss": True,   # False = paper protocol (padding counted)
    "init_from": None,              # fine-tune only: frozen config to start from (same seed)
    "embeddings": "learned",        # learned | glove
    "train_embeddings": True,
    "train_fraction": 1.0,          # S2 learning curve: fraction of training IMAGES used
    "transformer_layers": 3,        # Transformer only
    "transformer_heads": 4,
    "transformer_ff_dim": 1024,
}


def load_config(path: str, overrides: dict | None = None) -> dict:
    with open(path) as f:
        cfg = {**DEFAULTS, **(yaml.safe_load(f) or {})}
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
    if not cfg["config_name"]:
        cfg["config_name"] = os.path.splitext(os.path.basename(path))[0].removeprefix("config_")
    return cfg


def run_dir(runs_dir: str, config_name: str, seed: int) -> str:
    return os.path.join(runs_dir, config_name, f"seed{seed}")


def read_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)  # atomic, so a disconnect never leaves a half-written file


def load_embedding_matrix(cfg: dict, processed_dir: str):
    if cfg["embeddings"] == "learned":
        return None
    if cfg["embeddings"] == "glove":
        path = os.path.join(processed_dir, f"glove_{cfg['embedding_dim']}d_matrix.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing. Run scripts/build_glove_matrix.py first.")
        return np.load(path)
    raise ValueError(f"Unknown embeddings option {cfg['embeddings']!r}")


def build_model_from_config(cfg: dict, vocab_size: int, processed_dir: str = "data/processed",
                            cnn_weights: str | None = "imagenet"):
    from src.models.caption_model import build_caption_model
    return build_caption_model(
        vocab_size=vocab_size,
        max_length=cfg["max_length"],
        decoder_type=cfg["decoder"],
        encoder_mode=cfg["encoder_mode"],
        embedding_dim=cfg["embedding_dim"],
        units=cfg["units"],
        dropout=cfg["dropout"],
        embedding_matrix=load_embedding_matrix(cfg, processed_dir),
        train_embeddings=cfg["train_embeddings"],
        cnn_weights=cnn_weights,
        n_layers=cfg["transformer_layers"],
        n_heads=cfg["transformer_heads"],
        ff_dim=cfg["transformer_ff_dim"],
    )


def load_trained_model(rdir: str, vocab_size: int, processed_dir: str = "data/processed",
                       which: str = "best", cnn_weights: str | None = None):
    """Rebuilds a run's model from its saved config and loads its weights."""
    with open(os.path.join(rdir, "config.yaml")) as f:
        cfg = {**DEFAULTS, **yaml.safe_load(f)}
    if cfg["embeddings"] != "learned":
        cfg = {**cfg, "embeddings": "learned"}  # weights file already holds the trained embeddings
    model = build_model_from_config(cfg, vocab_size, processed_dir, cnn_weights=cnn_weights)
    model.load_weights(os.path.join(rdir, f"{which}.weights.h5"))
    return model, cfg
