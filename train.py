"""
train.py

Trains one ablation config with one random seed.

    python train.py --config experiments/config_lstm_frozen.yaml --seed 1
    python train.py --config experiments/config_lstm_finetuned.yaml --seed 1   # needs lstm_frozen seed1 first

Everything for the run is written to runs/<config_name>/seed<seed>/ (see
src/utils/run_io.py). The script is safe to stop at any time: re-running the
same command resumes from the last finished epoch, and a finished run is
skipped. Point --runs_dir at Google Drive on Colab so nothing is lost when a
session disconnects.

Training details:
  * teacher forcing over the whole caption in one pass
  * Adam, loss = cross-entropy on real words (padding ignored) unless the
    config sets mask_padding_in_loss: false (paper-protocol replication)
  * early stopping on validation loss (patience from the config); the
    weights from the best epoch are kept in best.weights.h5
  * fine-tuned configs start from the trained frozen model of the same seed
    (init_from) and then also update ResNet50's last block, at a lower
    learning rate — standard two-stage fine-tuning
"""

import os
import sys
import json
import time
import shutil
import argparse

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import keras  # noqa: E402

from src.tokenizer import CaptionTokenizer  # noqa: E402
from src.dataset import encode_split, CaptionBatches  # noqa: E402
from src.feature_extraction import load_features  # noqa: E402
from src.models.caption_model import compile_caption_model, encoder_and_decoder, count_params  # noqa: E402
from src.utils.run_io import (load_config, run_dir, read_json, write_json,  # noqa: E402
                              build_model_from_config)

FEATURE_MODE = {"frozen": "pooled", "finetune": "conv4", "grid": "grid"}


class RunTracker(keras.callbacks.Callback):
    """
    Saves last/best weights every epoch, keeps history, does early stopping,
    and writes state.json so a disconnected run can resume exactly.
    """

    def __init__(self, rdir: str, state: dict, history: dict, patience: int | None):
        super().__init__()
        self.rdir, self.state, self.history, self.patience = rdir, state, history, patience
        self.t0 = None

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = {k: float(v) for k, v in (logs or {}).items()}
        logs["epoch_seconds"] = round(time.time() - self.t0, 1)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.model.save_weights(os.path.join(self.rdir, "last.weights.h5"))
        val_loss = logs.get("val_loss", logs["loss"])
        if val_loss < self.state["best_val_loss"]:
            self.state.update(best_val_loss=val_loss, best_epoch=epoch, wait=0)
            self.model.save_weights(os.path.join(self.rdir, "best.weights.h5"))
        else:
            self.state["wait"] += 1
        self.state["last_epoch"] = epoch

        stop = self.patience is not None and self.state["wait"] >= self.patience
        if stop:
            self.model.stop_training = True
            self.state["stopped_early"] = True
        write_json(os.path.join(self.rdir, "history.json"), self.history)
        write_json(os.path.join(self.rdir, "state.json"), self.state)
        print(f"  epoch {epoch + 1}: val_loss={val_loss:.4f} "
              f"(best {self.state['best_val_loss']:.4f} @ epoch {self.state['best_epoch'] + 1}, "
              f"wait {self.state['wait']}){'  -> early stop' if stop else ''}")


def select_train_images(captions: dict, fraction: float) -> dict:
    """Nested, deterministic subsets for the data-size learning curve (S2)."""
    if fraction >= 1.0:
        return captions
    names = sorted(captions)
    np.random.default_rng(1234).shuffle(names)
    keep = names[:max(1, int(round(len(names) * fraction)))]
    return {n: captions[n] for n in keep}


def warm_start_from_frozen(model, cfg, args, vocab_size):
    """Copies bridge + decoder weights from the trained frozen run of the same seed."""
    src_dir = run_dir(args.runs_dir, cfg["init_from"], args.seed)
    src_state = read_json(os.path.join(src_dir, "state.json"), {})
    if not src_state.get("done"):
        raise RuntimeError(f"{cfg['config_name']} starts from {cfg['init_from']} seed {args.seed}, "
                           f"which has not finished training yet ({src_dir}). Train it first.")
    with open(os.path.join(src_dir, "config.yaml")) as f:
        src_cfg = yaml.safe_load(f)
    for key in ["decoder", "embedding_dim", "units", "max_length"]:
        if src_cfg[key] != cfg[key]:
            raise ValueError(f"init_from mismatch on {key}: {src_cfg[key]} vs {cfg[key]}")

    frozen = build_model_from_config({**cfg, **src_cfg, "embeddings": "learned"}, vocab_size,
                                     args.processed_dir, cnn_weights=None)
    frozen.load_weights(os.path.join(src_dir, "best.weights.h5"))
    f_enc, f_dec = encoder_and_decoder(frozen)
    t_enc, t_dec = encoder_and_decoder(model)
    t_enc.get_layer("encoder_bridge").set_weights(f_enc.get_layer("encoder_bridge").get_weights())
    t_dec.set_weights(f_dec.get_weights())
    print(f"Warm-started bridge + decoder from {src_dir}/best.weights.h5")


def main():
    p = argparse.ArgumentParser(description="Train one image captioning config.")
    p.add_argument("--config", required=True, help="experiments/config_*.yaml")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--features_dir", default="data/features")
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--epochs", type=int, default=None, help="Override the config's epoch count.")
    p.add_argument("--limit_images", type=int, default=None, help="Quick test on N train images.")
    p.add_argument("--restart", action="store_true", help="Delete this run's folder and start over.")
    p.add_argument("--cnn_weights", default="imagenet", help="'imagenet' or 'none' (tests only).")
    args = p.parse_args()

    cfg = load_config(args.config, {"epochs": args.epochs})
    rdir = run_dir(args.runs_dir, cfg["config_name"], args.seed)
    if args.restart and os.path.exists(rdir):
        shutil.rmtree(rdir)
    os.makedirs(rdir, exist_ok=True)

    state = read_json(os.path.join(rdir, "state.json"))
    if state and state.get("done"):
        print(f"[skip] {cfg['config_name']} seed {args.seed} already finished "
              f"(best epoch {state['best_epoch'] + 1}). Use --restart to retrain.")
        return
    with open(os.path.join(rdir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    keras.utils.set_random_seed(args.seed)
    print(f"=== {cfg['config_name']} | seed {args.seed} | {cfg['decoder']} decoder | "
          f"{cfg['encoder_mode']} encoder ===")

    # ---- data ----
    tokenizer = CaptionTokenizer.load(os.path.join(args.processed_dir, "tokenizer.json"))
    feats, name_to_row = load_features(args.features_dir, FEATURE_MODE[cfg["encoder_mode"]],
                                       mmap=cfg["encoder_mode"] != "frozen")
    with open(os.path.join(args.processed_dir, "train_captions.json")) as f:
        train_caps = select_train_images(json.load(f), cfg["train_fraction"])
    with open(os.path.join(args.processed_dir, "val_captions.json")) as f:
        val_caps = json.load(f)
    if args.limit_images:
        train_caps = dict(sorted(train_caps.items())[:args.limit_images])
        val_caps = dict(sorted(val_caps.items())[:max(10, args.limit_images // 5)])

    tr_rows, tr_tok, _ = encode_split(train_caps, tokenizer, cfg["max_length"], name_to_row)
    va_rows, va_tok, _ = encode_split(val_caps, tokenizer, cfg["max_length"], name_to_row)
    train_data = CaptionBatches(feats, tr_rows, tr_tok, cfg["batch_size"], True, args.seed)
    val_data = CaptionBatches(feats, va_rows, va_tok, 256, False)
    print(f"Train: {len(train_caps)} images / {len(tr_rows)} captions | "
          f"Val: {len(val_caps)} images / {len(va_rows)} captions | vocab {tokenizer.vocab_size}")

    # ---- model ----
    cnn_weights = None if args.cnn_weights == "none" else args.cnn_weights
    model = build_model_from_config(cfg, tokenizer.vocab_size, args.processed_dir, cnn_weights)
    compile_caption_model(model, cfg["learning_rate"], cfg["mask_padding_in_loss"])

    history = read_json(os.path.join(rdir, "history.json"), {})
    if state:  # resume
        model.load_weights(os.path.join(rdir, "last.weights.h5"))
        start_epoch = state["last_epoch"] + 1
        print(f"Resuming from epoch {start_epoch + 1} (optimizer state restarts)")
    else:
        if cfg["init_from"]:
            warm_start_from_frozen(model, cfg, args, tokenizer.vocab_size)
        state = {"last_epoch": -1, "best_epoch": -1, "best_val_loss": float("inf"),
                 "wait": 0, "stopped_early": False, "done": False,
                 "params": count_params(model),
                 "train_images": len(train_caps), "train_captions": int(len(tr_rows)),
                 "gpu": [d.name for d in __import__("tensorflow").config.list_physical_devices("GPU")]}
        start_epoch = 0
        if cfg["init_from"]:  # record the starting point so the comparison is fair
            init = model.evaluate(val_data, verbose=0, return_dict=True)
            state["val_at_start"] = {k: float(v) for k, v in init.items()}
            src_best = read_json(os.path.join(run_dir(args.runs_dir, cfg["init_from"], args.seed),
                                              "state.json"))["best_val_loss"]
            state["init_from_best_val_loss"] = src_best
            gap = abs(state["val_at_start"]["loss"] - src_best) / src_best
            print(f"Validation before fine-tuning: {state['val_at_start']} "
                  f"(frozen model's best val_loss: {src_best:.4f})")
            if gap > 0.02 and cnn_weights is not None:
                print(f"WARNING: starting val_loss differs from the frozen model's by {gap:.1%}. "
                      "The conv4 features and the pooled features may come from different "
                      "ResNet50 weights; re-extract both with the same settings.")

    tracker = RunTracker(rdir, state, history, cfg["early_stopping_patience"])
    t0 = time.time()
    if start_epoch < cfg["epochs"] and not state.get("stopped_early"):
        model.fit(train_data, validation_data=val_data, epochs=cfg["epochs"],
                  initial_epoch=start_epoch, callbacks=[tracker], verbose=2)

    state["done"] = True
    state["train_seconds"] = round(state.get("train_seconds", 0) + time.time() - t0, 1)
    write_json(os.path.join(rdir, "state.json"), state)
    print(f"Done. Best epoch {state['best_epoch'] + 1}, val_loss {state['best_val_loss']:.4f}. "
          f"Weights: {rdir}/best.weights.h5")


if __name__ == "__main__":
    main()
