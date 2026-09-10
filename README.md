
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

**Why this project is more than a copy-paste tutorial:** most image captioning demos stop at "build one model, see if it works." This project instead runs a **controlled experiment** — training **4 different versions** of the same system, changing one design choice at a time, to actually measure what matters:

1. Does an **LSTM** or a **GRU** make a better decoder?
2. Does **fine-tuning** the CNN (letting it adapt during training) actually help, or is a plain pretrained CNN good enough?

This 2×2 comparison — LSTM/GRU × frozen/fine-tuned — is something neither of the two research papers this project is based on actually tests directly. That comparison is this project's real contribution.

---

## 📊 Where Things Stand Right Now

### ✅ Fully Done
- **Dataset pipeline** — Flickr8k (8,091 images, 40,455 captions) loaded, cleaned, and split into train/val/test
- **Vocabulary** — 2,740-word vocabulary built from training captions
- **Config 1: LSTM + Frozen CNN** — ✅ **trained for the full 30 epochs and evaluated**

  | Metric | Result |
  |---|---|
  | Final training accuracy | 44.5% |
  | Final validation accuracy | 44.3% (no overfitting — train/val nearly identical) |
  | BLEU-1 | **0.4846** |
  | BLEU-2 | 0.2866 |
  | BLEU-3 | 0.1892 |
  | BLEU-4 | 0.1286 |

  **Real example output** from this model:
  > 🖼️ *(a photo of a black-and-white dog on grass)*
  > **Predicted:** "a black and white dog is running on the grass"
  > **Reference (human-written):** "a dog jumps off a wooden porch"

  The captions are grammatically correct and often get the right general subject (dog, grass) — sometimes the exact action/scene doesn't match, which is expected at this stage and matches known limitations described in both source papers.

### 🔄 Currently Running
- **Config 2: LSTM + Fine-tuned CNN** — actively training (true end-to-end fine-tuning, where the CNN itself adjusts during training, not just the decoder)
  - Currently past epoch 8/30
  - Loss steadily decreasing every epoch (5.03 → 4.17 and still dropping)
  - Validation accuracy tracking slightly *above* training accuracy — a good sign of healthy generalization, no overfitting
  - Training across multiple sessions using automatic checkpoint save/resume, so no progress is ever lost between sessions

### ⏳ Not Started Yet
- **Config 3: GRU + Frozen CNN**
- **Config 4: GRU + Fine-tuned CNN**
- Final BLEU comparison across all 4 configs
- Written results analysis and report

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
├── src/                  Core code: data loading, tokenizer, CNN feature extraction, model architectures
├── train.py                Trains the FROZEN-CNN configs (fast — CNN features precomputed once)
├── train_finetune.py        Trains the FINE-TUNED-CNN configs (slower — CNN learns too, in real time)
├── evaluate.py                Scores a trained model using BLEU
├── inference.py                 Generate a caption for any image you give it
├── experiments/                  One config file per ablation run (LSTM/GRU × frozen/fine-tuned)
├── docs/                           Proposal, literature review
└── tests/                           Automated checks for the core code
```

---

## 🧪 The Four-Way Comparison

| Config | Decoder | CNN Encoder | Status |
|---|:---:|:---:|:---:|
| `lstm_frozen` | LSTM | Frozen | ✅ Done — BLEU-1: 0.4846 |
| `lstm_finetuned` | LSTM | Fine-tuned | 🔄 Training (epoch 8+/30) |
| `gru_frozen` | GRU | Frozen | ⏳ Pending |
| `gru_finetuned` | GRU | Fine-tuned | ⏳ Pending |

---

## 🔭 What's Next

1. Finish training `lstm_finetuned`
2. Train `gru_frozen` and `gru_finetuned`
3. Run BLEU evaluation on all 4, compare them side by side
4. Write up the findings — which decoder wins, whether fine-tuning was worth the extra compute cost

---

## 🙏 Acknowledgements

Built for the **Deep Learning** course at **SVKM's NMIMS — Mukesh Patel School of Technology Management & Engineering**.

