# GPU Training Handoff Instructions

If you're reading this because you've been asked to run the training step
on your GPU machine — thanks! Here's exactly what to do.

## What you need from me
- This whole repo (clone it, or I'll zip and send it)
- The `data/features/*.pkl` files (precomputed CNN features — I'll generate and send these, they're a few hundred MB, not the full image dataset)
- The `data/processed/*.json` files (captions + tokenizer — small, included in the repo transfer)

You do **not** need the raw Flickr8k images — feature extraction already happened on my end.

## Setup on your machine

```bash
pip install -r requirements.txt
```

Make sure your GPU is detected:
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
You should see your GPU listed (e.g., an entry mentioning your RTX 4050). If the list is empty, your CUDA/cuDNN setup needs attention before training will actually use the GPU (it'll silently fall back to CPU otherwise).

## Running training — do this 4 times, once per config

```bash
python train.py --decoder lstm --features data/features/resnet50_frozen_features.pkl   --config_name lstm_frozen
python train.py --decoder lstm --features data/features/resnet50_finetune_features.pkl --config_name lstm_finetuned
python train.py --decoder gru  --features data/features/resnet50_frozen_features.pkl   --config_name gru_frozen
python train.py --decoder gru  --features data/features/resnet50_finetune_features.pkl --config_name gru_finetuned
```

Each one trains for 30 epochs by default (edit `--epochs` to change).

## Pausing / stopping

It's completely safe to stop any of these runs at any point (Ctrl+C, or just closing the terminal). Re-running the exact same command later will automatically pick up from the last completed epoch — nothing is lost except whatever epoch was in progress.

## When it's done

Send back:
- The whole `checkpoints/` folder (the trained weight files)
- The whole `experiments/results/` folder (metrics JSON files)

That's everything I need to run evaluation and generate the report on my end — no further GPU work required after this.

## Rough time estimate on an RTX 4050 (6GB)
Each config should take somewhere in the range of a few minutes to ~30 minutes total for 30 epochs, depending on batch size and dataset size — Flickr8k is small enough that this shouldn't be a long commitment. If any single config seems to be taking multiple hours, something's likely misconfigured (e.g., not actually using the GPU) — worth double-checking the GPU detection step above.
