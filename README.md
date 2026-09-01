# Image Captioning with CNN–LSTM/GRU

A deep learning system that generates natural-language captions for images, built as a course project for **Deep Learning (702AI0C008)**, SVKM's NMIMS — Mukesh Patel School of Technology Management & Engineering, AY 2026–27.

A pretrained **ResNet50** CNN extracts visual features from an input image, which are decoded into a caption word-by-word by an **LSTM or GRU** recurrent decoder. The project's core contribution is a **controlled ablation study**: LSTM vs. GRU decoders, and frozen vs. fine-tuned CNN encoders, compared under identical conditions on the **Flickr8k** dataset.

```
[Image] → ResNet50 (frozen or fine-tuned) → feature vector → LSTM/GRU decoder → caption
```

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

Everything except the actual training loop runs comfortably on a CPU-only machine (tested on an i7, no GPU).

| Step | Command | Needs GPU? |
|---|---|---|
| 1. Split dataset | `python src/data_loader.py` | No |
| 2. Build vocabulary | `python build_tokenizer.py` | No |
| 3. Extract CNN features (frozen) | `python src/feature_extraction.py --mode frozen` | No |
| 4. Extract CNN features (finetune-tagged) | `python src/feature_extraction.py --mode finetune` | No |
| 5. **Train each ablation config** | `python train.py --decoder lstm --features data/features/resnet50_frozen_features.pkl --config_name lstm_frozen` (repeat ×4, see `experiments/*.yaml`) | **Yes** |
| 6. Evaluate (BLEU) | `python evaluate.py --decoder lstm --features ... --config_name lstm_frozen --checkpoint_epoch 29` | No |
| 7. Caption any new image | `python inference.py --image your_photo.jpg --decoder lstm --config_name lstm_frozen --checkpoint_epoch 29` | No |

Step 5 is the only GPU-bound step — it's designed to be handed off (e.g., run on a friend's GPU machine) as a standalone script, and supports **pause/resume**: if training is interrupted, re-running the same command automatically resumes from the last saved epoch instead of restarting.

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
