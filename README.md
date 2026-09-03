# Image Captioning with CNN–LSTM/GRU

A deep learning system that generates natural-language captions for images, built as a course project for **Deep Learning (702AI0C008)**, SVKM's NMIMS — Mukesh Patel School of Technology Management & Engineering, AY 2026–27.

A pretrained **ResNet50** CNN extracts visual features from an input image, which are decoded into a caption word-by-word by an **LSTM or GRU** recurrent decoder. The project's core contribution is a **controlled ablation study**: LSTM vs. GRU decoders, and frozen vs. fine-tuned CNN encoders, compared under identical conditions on the **Flickr8k** dataset.

```
[Image] → ResNet50 (frozen or fine-tuned) → feature vector → LSTM/GRU decoder → caption
```

## Project Status

This is a course project actively in progress, not a finished system.

### Completed

- Dataset loading and train/val/test splitting (verified on the full Flickr8k dataset — 8,091 images, 40,455 captions)
- Vocabulary building (verified — 2,740-token vocabulary from training captions)
- CNN feature extraction pipeline using ResNet50 (verified working with GPU acceleration on a demo-scale subset)
- Model architecture for both LSTM and GRU decoder variants, with a shared encoder bridge (implemented and verified to build/compile correctly)
- Training loop with automatic checkpoint save and resume support (verified — a demo training run showed loss decreasing across epochs, and checkpoints saved correctly after each epoch)
- Inference pipeline for generating a caption on any new image, with both greedy and beam search decoding (verified working end-to-end)
- Unit tests for the tokenizer and data loader (passing)

### Pending

- Full-scale training on the complete Flickr8k dataset (training has only been validated on a small subset so far, for pipeline testing — not yet run at full scale or for a full epoch count)
- The four-way ablation study comparing LSTM vs. GRU decoders and frozen vs. fine-tuned CNN encoders (architecture supports all four configurations, but none have been trained to completion yet)
- BLEU/METEOR evaluation on the full test set
- Results analysis, comparison charts, and the final written report

## Goal

The goal of this project is to build a working CNN-LSTM/GRU image captioning system grounded in two recent mother papers, and to run a controlled ablation comparing decoder type (LSTM vs. GRU) and encoder training strategy (frozen vs. fine-tuned) — a comparison neither mother paper performs directly. The system should generate a natural-language caption for any input image end-to-end, from raw pixels through a CNN encoder to an RNN decoder. The ablation results are intended to isolate which of these two design choices matters more for caption quality on Flickr8k.

## Motivation & Literature

This project is grounded in two recent papers, and directly addresses a gap neither of them isolates: a controlled, side-by-side comparison of decoder type and encoder training strategy.

| Paper | Contribution | Limitation this project probes |
|---|---|---|
| Ahmad, Azhar & Sattar (2023) — *CNN+GRU with semantic reconstructor* ([arXiv:2301.02440](https://arxiv.org/abs/2301.02440)) | GRU decoder + attribute vector + caption-to-image semantic reconstructor, evaluated on MS COCO | Doesn't isolate GRU's contribution from the reconstructor's |
| Kavitha & Karpagam (2025) — *ResNet50 + Hybrid LSTM–GRU + Beam Search* ([Automatika, DOI](https://doi.org/10.1080/00051144.2025.2485695)) | Systematic CNN encoder comparison + hybrid LSTM-GRU decoder + beam search, on Flickr8k | Doesn't run LSTM-only vs. GRU-only as a controlled standalone comparison |

Full literature review and proposal: see [`docs/proposal.pdf`](docs/proposal.pdf) and [`docs/literature_review.md`](docs/literature_review.md).

## Repository Structure

```
├── docs/                    Proposal, literature review, final report
├── data/
│   ├── raw/                 Flickr8k images + captions.txt (not included — see Setup)
│   ├── processed/           Train/val/test split JSONs + tokenizer (generated)
│   └── features/            Precomputed ResNet50 feature vectors (generated)
├── src/
│   ├── data_loader.py       Parses & splits the Flickr8k dataset
│   ├── tokenizer.py         Vocabulary building, caption encode/decode
│   ├── feature_extraction.py CNN feature extraction (CPU-friendly)
│   ├── models/               Encoder bridge, LSTM decoder, GRU decoder, combined model
│   └── utils/                Checkpoint save/resume, beam search decoding
├── build_tokenizer.py       Builds vocabulary from training captions
├── train.py                 Training loop — THE STEP THAT NEEDS A GPU
├── evaluate.py               BLEU-score evaluation on the test set
├── inference.py               Generate a caption for any single image
├── experiments/               YAML configs for each of the 4 ablation runs + results
├── notebooks/                  Step-by-step exploratory notebooks
├── outputs/                    Sample generated captions, comparison charts
└── tests/                      Unit tests for tokenizer & data loader
```

## Setup

```bash
git clone <this-repo-url>
cd image-captioning-cnn-lstm-gru
pip install -r requirements.txt
```

Download **Flickr8k** (e.g., from Kaggle) and place it as:
```
data/raw/
├── Images/            (all .jpg files)
└── captions.txt
```

## Pipeline — what needs a GPU and what doesn't

Everything except the actual training loop runs comfortably on a CPU-only machine.

| Step | Command | Needs GPU? |
|---|---|---|
| 1. Split dataset | `python src/data_loader.py` | No |
| 2. Build vocabulary | `python build_tokenizer.py` | No |
| 3. Extract CNN features (frozen) | `python src/feature_extraction.py --mode frozen` | No |
| 4. Extract CNN features (finetune-tagged) | `python src/feature_extraction.py --mode finetune` | No |
| 5. **Train each ablation config** | `python train.py --decoder lstm --features data/features/resnet50_frozen_features.pkl --config_name lstm_frozen` (repeat ×4, see `experiments/*.yaml`) | **Yes** |
| 6. Evaluate (BLEU) | `python evaluate.py --decoder lstm --features ... --config_name lstm_frozen --checkpoint_epoch 29` | No |
| 7. Caption any new image | `python inference.py --image your_photo.jpg --decoder lstm --config_name lstm_frozen --checkpoint_epoch 29` | No |

Step 5 is the only GPU-bound step — it's designed to be handed off as a standalone script and can be run on any available CUDA-capable machine (a local GPU, a lab machine, or a cloud GPU runtime such as Google Colab), and supports **pause/resume**: if training is interrupted, re-running the same command automatically resumes from the last saved epoch instead of restarting.

## Ablation Configurations

| Config | Decoder | Encoder | Config file |
|---|---|---|---|
| `lstm_frozen` | LSTM | ResNet50 (frozen) | `experiments/config_lstm_frozen.yaml` |
| `lstm_finetuned` | LSTM | ResNet50 (fine-tuned) | `experiments/config_lstm_finetuned.yaml` |
| `gru_frozen` | GRU | ResNet50 (frozen) | `experiments/config_gru_frozen.yaml` |
| `gru_finetuned` | GRU | ResNet50 (fine-tuned) | `experiments/config_gru_finetuned.yaml` |

Results (BLEU-1 to BLEU-4, loss/accuracy curves) are written to `experiments/results/` after training and evaluation — populated once training has actually been run, not included as placeholders in this repo.

## Testing

```bash
pytest tests/
```

Covers tokenizer encode/decode round-tripping, vocabulary frequency filtering, and dataset split logic — all runnable without the dataset or a GPU.

## Future Work

Attention-based decoding (e.g., *Show, Attend and Tell*) and Transformer-based architectures are explicitly out of scope for this project, since they fall outside the CNN/RNN/LSTM/GRU fundamentals covered in the source course — noted here as a natural next step rather than an oversight.

## Acknowledgements

Built for the Deep Learning course at SVKM's NMIMS, Mukesh Patel School of Technology Management & Engineering.
