# 🖼️ Image Captioning with CNN–LSTM/GRU

<p align="center">
  <b>Teaching a machine to look at a photo and describe it in a sentence.</b><br>
  A CNN–RNN hybrid deep learning project · Deep Learning Course (702AI0C008) · SVKM's NMIMS, MPSTME · AY 2026–27
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
  <img alt="Status" src="https://img.shields.io/badge/Status-In%20Progress-yellow">
</p>

---

## 📖 Overview

**Image captioning** sits at the intersection of two of the hardest problems in AI: *understanding what's in an image* (computer vision) and *describing it in fluent language* (natural language generation). This project builds a complete, working system that does exactly that — give it any photo, and it generates a natural-language sentence describing what's happening in it.

```
🖼️  Image  →  🧠 ResNet50 (CNN Encoder)  →  📐 2048-d feature vector  →  🔤 LSTM / GRU Decoder  →  💬 "a brown dog running through the grass"
```

A pretrained **ResNet50** convolutional network looks at the image and compresses it into a rich numerical representation. That representation is then handed to a **recurrent decoder — either an LSTM or a GRU** — which generates the caption one word at a time, learning to connect visual patterns to language.

What makes this more than a tutorial rebuild: the project runs a **controlled ablation study**, systematically comparing:
- 🔁 **LSTM vs. GRU** as the decoder
- ❄️ **Frozen vs. fine-tuned** CNN encoder

...under otherwise identical conditions — a direct, isolated comparison that **neither of the two mother papers this project is grounded in actually performs.**

---

## 📊 Project Status

> This is an **active, in-progress course project**. Here's exactly what's real and working right now, and what's still ahead.

### ✅ Completed & Verified

| Component | Status |
|---|---|
| Dataset loading & train/val/test split | ✅ Verified on full Flickr8k — **8,091 images, 40,455 captions** |
| Vocabulary construction | ✅ Verified — **2,740-token vocabulary** |
| ResNet50 feature extraction (GPU-accelerated) | ✅ Verified working on a demo-scale run |
| LSTM & GRU decoder architectures | ✅ Implemented, builds & compiles correctly |
| Training loop with checkpoint save/resume | ✅ Verified — loss decreases correctly across epochs, checkpoints persist correctly |
| End-to-end inference (image → caption) | ✅ Verified working, both greedy & beam search decoding |
| Unit tests (tokenizer, data loader) | ✅ All passing |

### ⏳ Pending

- 🔲 Full-scale training on the **complete** Flickr8k dataset (validated so far only on a small demo subset)
- 🔲 The full **4-way ablation study** (LSTM/GRU × frozen/fine-tuned) trained to completion
- 🔲 BLEU / METEOR evaluation across the full test set
- 🔲 Final results analysis, comparison charts, and written report

---

## 🎯 Goal

Build a working CNN–LSTM/GRU image captioning system grounded in recent literature, then go one step further than either mother paper does: run a **controlled ablation** isolating decoder type and encoder training strategy as independent variables. Neither reference paper runs this specific comparison — this project's contribution is turning *"Paper A did X, Paper B did Y"* into a direct, measured **X vs. Y**, on the same data, under the same conditions.

---

## 📚 Grounded in Recent Literature

| 📄 Paper | 💡 Contribution | 🔍 Gap this project addresses |
|---|---|---|
| **Ahmad, Azhar & Sattar (2023)**<br>*CNN+GRU with semantic reconstructor*<br>[arXiv:2301.02440](https://arxiv.org/abs/2301.02440) | GRU decoder + attribute vector + caption-to-image semantic reconstructor, evaluated on MS COCO | Doesn't isolate GRU's own contribution from the reconstructor module |
| **Kavitha & Karpagam (2025)**<br>*ResNet50 + Hybrid LSTM–GRU + Beam Search*<br>[Automatika, DOI](https://doi.org/10.1080/00051144.2025.2485695) | Systematic CNN encoder comparison + hybrid LSTM-GRU decoder + beam search, on Flickr8k | Doesn't test LSTM-only vs. GRU-only as a controlled standalone comparison |

📄 **Full literature review:** [`docs/literature_review.md`](docs/literature_review.md)
📄 **Project proposal:** [`docs/proposal.pdf`](docs/proposal.pdf)

---

## 🏗️ Architecture

```
                     ┌─────────────────────┐
    Input Image ───▶ │   ResNet50 Encoder   │ ───▶  2048-d feature vector
                     │  (frozen/fine-tuned) │
                     └─────────────────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │   Dense Bridge Layer │  ───▶  256-d embedding
                     └─────────────────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │  LSTM / GRU Decoder  │  ───▶  word-by-word generation
                     └─────────────────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │   Softmax over vocab │  ───▶  "a", "dog", "runs", ...
                     └─────────────────────┘
```

---

## 🗂️ Repository Structure

```
DL-Project/
├── 📁 docs/                       Proposal, literature review, GPU handoff guide
├── 📁 data/
│   ├── raw/                       Flickr8k images + captions.txt  (not committed — see Setup)
│   ├── processed/                  Train/val/test splits + tokenizer  (generated)
│   └── features/                    Precomputed ResNet50 feature vectors  (generated)
├── 📁 src/
│   ├── data_loader.py              Parses & splits the Flickr8k dataset
│   ├── tokenizer.py                 Vocabulary building, caption encode/decode
│   ├── feature_extraction.py         CNN feature extraction (CPU-friendly)
│   ├── models/                        Encoder bridge · LSTM decoder · GRU decoder · combined model
│   └── utils/                          Checkpoint save/resume · beam search decoding
├── build_tokenizer.py               Builds vocabulary from training captions
├── train.py                          🔥 Training loop — the one GPU-bound step
├── evaluate.py                        BLEU evaluation on the test set
├── inference.py                        Generate a caption for any image
├── 📁 experiments/                       YAML configs for each ablation run + results
├── 📁 notebooks/                          Step-by-step exploratory notebooks
├── 📁 outputs/                             Sample captions, comparison charts
└── 📁 tests/                                Unit tests
```

---

## ⚙️ Setup

```bash
git clone https://github.com/arz1742/DL-Project.git
cd DL-Project
pip install -r requirements.txt
```

Download **Flickr8k** (e.g. from Kaggle) and place it as:
```
data/raw/
├── Images/          ← all .jpg files
└── captions.txt
```

---

## 🚀 Running the Pipeline

Everything except training runs comfortably on a **CPU-only machine** — no GPU required until step 5.

| Step | Command | GPU? |
|:---:|---|:---:|
| 1️⃣ | `python src/data_loader.py` — split the dataset | ❌ |
| 2️⃣ | `python build_tokenizer.py` — build vocabulary | ❌ |
| 3️⃣ | `python src/feature_extraction.py --mode frozen` — extract CNN features | ❌ |
| 4️⃣ | `python src/feature_extraction.py --mode finetune` — tag features for fine-tuning | ❌ |
| 5️⃣ | `python train.py --decoder lstm --features data/features/resnet50_frozen_features.pkl --config_name lstm_frozen` — **train** (×4 configs, see `experiments/`) | ✅ |
| 6️⃣ | `python evaluate.py --decoder lstm --config_name lstm_frozen --checkpoint_epoch 29` — BLEU score | ❌ |
| 7️⃣ | `python inference.py --image your_photo.jpg --decoder lstm --config_name lstm_frozen --checkpoint_epoch 29` — caption any image | ❌ |

> 💡 **Tip:** add `--limit N` to `feature_extraction.py` to run a fast partial pipeline on just N images — great for quick demos or sanity checks before a full run.

Training (step 5️⃣) is the only GPU-bound stage, and can run on **any CUDA-capable machine** — a local GPU, a lab machine, or a cloud GPU runtime like Google Colab. It supports **automatic pause/resume**: an interrupted run picks up exactly where it left off, no progress lost.

---

## 🧪 The Ablation Study

| Config | Decoder | Encoder | Config File |
|---|:---:|:---:|---|
| `lstm_frozen` | LSTM | ResNet50 (frozen) | [`config_lstm_frozen.yaml`](experiments/config_lstm_frozen.yaml) |
| `lstm_finetuned` | LSTM | ResNet50 (fine-tuned) | [`config_lstm_finetuned.yaml`](experiments/config_lstm_finetuned.yaml) |
| `gru_frozen` | GRU | ResNet50 (frozen) | [`config_gru_frozen.yaml`](experiments/config_gru_frozen.yaml) |
| `gru_finetuned` | GRU | ResNet50 (fine-tuned) | [`config_gru_finetuned.yaml`](experiments/config_gru_finetuned.yaml) |

Results (BLEU-1 → BLEU-4, loss/accuracy curves) land in `experiments/results/` once training and evaluation are run — not pre-populated with placeholder numbers.

---

## ✅ Testing

```bash
pytest tests/
```

Covers tokenizer round-tripping, vocabulary frequency filtering, and dataset split logic — all runnable instantly, no dataset or GPU needed.

---

## 🔭 Future Work

**Attention mechanisms** and **Transformer-based architectures** (e.g., *Show, Attend and Tell*) are deliberately out of scope here — they sit outside the CNN/RNN/LSTM/GRU fundamentals this project is built around. Flagged here as a clear next step, not an oversight.

---

## 🙏 Acknowledgements

Built for the **Deep Learning** course at **SVKM's NMIMS — Mukesh Patel School of Technology Management & Engineering**.
