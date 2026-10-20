
# 🖼️ Image Captioning with CNN–LSTM/GRU

<p align="center">
  <b>A deep learning system that looks at a photo and writes a sentence describing it.</b><br>
  Deep Learning Course (702AI0C008) · SVKM's NMIMS, MPSTME · AY 2026–27
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white">
  <img alt="Status" src="https://img.shields.io/badge/Status-In%20Progress-yellow">
</p>

---

## 🤔 What Is This Project?

Imagine showing a computer a photo of a dog running in a park — and it writes back, in plain English: *"a brown dog running through the grass."* That's **image captioning** — teaching a machine to both *see* an image and *describe* it in a full sentence, not just label it.

This project builds that system from scratch, combining two types of neural networks:

```
🖼️  A photo goes in
     ↓
🧠  A CNN (ResNet50) looks at it and extracts what's visually important
     ↓
🔤  An RNN (LSTM or GRU) turns that visual information into a sentence, one word at a time
     ↓
💬  "a brown dog running through the grass"
```

**Why this project is more than a copy-paste tutorial:** most image captioning demos stop at "build one model, see if it works." This project instead runs a **controlled experiment** — training **6 different versions** of the same system (each with 3 random seeds), changing one design choice at a time, to actually measure what matters:

1. Does an **LSTM**, a **GRU**, or Paper 1's **hybrid LSTM→GRU** make the best decoder?
2. Does **fine-tuning** the CNN (letting it adapt during training) actually help, or is a plain pretrained CNN good enough?

This comparison — LSTM / GRU / hybrid × frozen / fine-tuned — is something neither of the two research papers this project is based on actually tests directly. That comparison is this project's real contribution.

---

## 📊 Where Things Stand Right Now

### ✅ Done
- **Dataset pipeline** — Flickr8k (8,091 images, 40,455 captions), split by image into **6,091 / 1,000 / 1,000** (train / val / test, matching Kavitha & Karpagam 2025), with an automatic check that no image appears in two splits
- **Rebuilt training pipeline** (Day 1 of the 2-week plan):
  - whole-caption teacher forcing (~12× fewer forward passes than before)
  - **real fine-tuning**: ResNet50's last block (conv5_x) is trained on precomputed stage-4 maps
  - config-driven runs with random seeds, early stopping and automatic resume after a Colab disconnect
  - batched greedy + beam search (minutes instead of ~1 hour per evaluation)
  - full metric suite: COCO BLEU-1–4, METEOR, ROUGE-L, CIDEr, masked / unmasked / top-5 token accuracy, key-object recall, caption diversity, and Paper 1's 40-image protocol
- **25 automated tests** (`pytest tests/`)

### ⏳ Next
- Run the full study on a Colab L4: 6 configs × 3 seeds + the paper-protocol replication (`notebooks/00_colab_pipeline.ipynb`)
- Showpiece analyses (Grad-CAM word heatmaps, learning curve, t-SNE, beam tree, error taxonomy) and the final dashboard

> **Earlier results (old pipeline, superseded):** `lstm_frozen` reached BLEU-1 0.4846 / BLEU-4 0.1286 (average smoothed sentence BLEU, ~607-image test split). The old "fine-tuned" configs reused the frozen features, so they did not actually fine-tune. All configs are retrained with the new pipeline.

---

## 🎯 The Goal

Build a working CNN + LSTM/GRU image captioning system grounded in two recent research papers, then go one step further than either paper does: directly and fairly compare **LSTM vs. GRU** and **frozen vs. fine-tuned CNN** — under identical conditions, on the same data. Whichever way the results land, that's a real, defensible finding neither source paper provides on its own.

---

## 📚 Built On

| Paper | What It Did |
|---|---|
| **Ahmad, Azhar & Sattar (2023)** — CNN+GRU with semantic reconstructor ([arXiv:2301.02440](https://arxiv.org/abs/2301.02440)) | GRU decoder + a module that reconstructs the image from the caption, to improve semantic accuracy |
| **Kavitha & Karpagam (2025)** — ResNet50 + Hybrid LSTM–GRU + Beam Search ([Automatika](https://doi.org/10.1080/00051144.2025.2485695)) | Compared multiple CNNs, then combined LSTM+GRU into one hybrid decoder with beam search, on Flickr8k |

📄 Full write-up: [`docs/literature_review.md`](docs/literature_review.md) · [`docs/proposal.pdf`](docs/proposal.pdf)

---

## 🗂️ Project Structure

```
DL-Project/
├── notebooks/00_colab_pipeline.ipynb   ▶ run EVERYTHING on Colab (start here)
├── train.py                            train one config + seed (resumable)
├── evaluate.py                         all metrics + predictions.csv for one run
├── inference.py                        caption any image
├── build_tokenizer.py                  vocabulary from the training captions
├── experiments/config_*.yaml           one file per config (see below)
├── scripts/run_experiments.py          train + evaluate many configs/seeds in order
├── scripts/aggregate_results.py        → results/ tables A, B, C + summary.json
├── src/
│   ├── data_loader.py                  captions.txt → image-level splits
│   ├── tokenizer.py                    cleaning + vocabulary
│   ├── feature_extraction.py           ResNet50 → pooled (2048) and conv4 (14×14×1024) features
│   ├── dataset.py                      batches of (image feature, caption)
│   ├── evaluation.py                   BLEU, METEOR, ROUGE-L, CIDEr, key-object recall, diversity
│   ├── models/encoder.py               frozen bridge / trainable conv5 block
│   ├── models/decoders.py              LSTM, GRU, hybrid LSTM→GRU
│   ├── models/metrics.py               masked / unmasked / top-5 accuracy, losses
│   └── utils/beam_search.py            batched greedy + beam search, beam traces
├── tests/                              pytest suite (CPU, no dataset needed)
└── docs/                               proposal, literature review
```

Every run writes to `runs/<config>/seed<N>/` (config, history, best/last weights, `eval/`).

---

## 🧪 The Experiments

| ID | Config | Decoder | ResNet50 encoder |
|---|---|:---:|:---:|
| C1 | `lstm_frozen` | LSTM | frozen |
| C2 | `gru_frozen` | GRU | frozen |
| C3 | `hybrid_frozen` | LSTM → GRU | frozen |
| C4 | `lstm_finetuned` | LSTM | last block fine-tuned |
| C5 | `gru_finetuned` | GRU | last block fine-tuned |
| C6 | `hybrid_finetuned` | LSTM → GRU | last block fine-tuned |
| — | `paper_protocol` | LSTM → GRU | frozen, trained exactly like Paper 1 |

Each core config runs with **3 seeds** so differences can be reported as mean ± std.
Fine-tuned configs start from the frozen model of the same seed (two-stage fine-tuning).

### Why there is a "paper protocol" run
Kavitha & Karpagam report **0.8932 accuracy** but only **~0.60 BLEU-1**. Their accuracy counts
padding tokens: most of a 34-token caption is padding, which is trivial to predict. The
`paper_protocol` run reproduces that training setup so the report can show both numbers
side by side — the same model, measured honestly and measured the paper's way.

---

## 🚀 How to Run

**On Colab (recommended):** open `notebooks/00_colab_pipeline.ipynb`, choose an **L4 GPU**, run all cells.

**Locally:**
```bash
pip install -r requirements.txt
python src/data_loader.py                        # needs data/raw/captions.txt + data/raw/Images/
python build_tokenizer.py
python src/feature_extraction.py --mode pooled
python src/feature_extraction.py --mode conv4
python scripts/run_experiments.py --stage frozen --seeds 1,2,3
python scripts/run_experiments.py --stage finetuned --seeds 1,2,3
python scripts/run_experiments.py --stage paper --seeds 1
python scripts/aggregate_results.py
python inference.py --image photo.jpg --config_name hybrid_frozen --seed 1 --show_probs
pytest tests/
```

---

## 🙏 Acknowledgements

Built for the **Deep Learning** course at **SVKM's NMIMS — Mukesh Patel School of Technology Management & Engineering**.

