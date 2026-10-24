# Image Captioning Lab: local dashboard

A Streamlit dashboard with **live captioning** (upload or webcam), per-word attention heatmaps,
beam-search trees, and every result from the project.

## 1. Get the bundle (once)

In `notebooks/00_colab_pipeline.ipynb`, run cells 1–6, then **cell 21 (dashboard bundle)**. It writes
`MyDrive/DL-Project/dashboard_bundle.zip` (~100 MB): trained models, results and photo thumbnails.

Download it from Google Drive.

## 2. Set up on your laptop (once)

You need **Python 3.10, 3.11 or 3.12** (TensorFlow doesn't support newer versions yet).

```bash
git clone https://github.com/arz1742/DL-Project.git
cd DL-Project
# unzip dashboard_bundle.zip HERE, into the DL-Project folder
#   (it creates runs/, results/, dashboard/data/ and data/processed/tokenizer.json)

python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r dashboard/requirements.txt
```

Optional, for the **CLIP model** (best captions) and **BLIP-large**:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
```

## 3. Run

```bash
streamlit run dashboard/app.py
```

It opens at http://localhost:8501. The first photo takes ~30 s (it downloads ResNet50's ImageNet
weights, ~100 MB, once); after that each photo takes a few seconds on a CPU.

## Pages

| Page | What it shows |
| --- | --- |
| Live captioning | Upload/webcam → captions from every model, read-aloud, heatmap per word, top-5 guesses, beam tree |
| Overview | The five findings and the model line-up |
| Results | Every metric for every model (mean ± sd over seeds), greedy vs beam |
| Which differences are real? | Paired-bootstrap confidence intervals and p-values |
| Accuracy illusion | Why Paper 1's 0.89 accuracy is mostly padding |
| Attention explorer | Precomputed heatmaps for 40 test photos (A1, A2, Grad-CAM) |
| Caption gallery | All 1000 test photos, every model's caption, filters and search |
| Errors & hallucination | CHAIR hallucination rates and error types |
| t-SNE map | What ResNet50 "sees"; click a dot to see the photo |
| Beam search | Beam-search trees |
| Training curves | Validation loss/accuracy per epoch for every seed |

**Live models:** C3 Hybrid (classic, with Grad-CAM), A1 LSTM + attention, A2 Transformer, and
Hybrid + CLIP (needs torch). The fine-tuned C6 is left out of the bundle because its weights are
~200 MB for a 1–2% gain over C3.

## Troubleshooting

- **"No trained models found"**: the bundle wasn't unzipped into the project folder. You should see
  `runs/attention/seed1/best.weights.h5`.
- **Webcam doesn't start**: allow camera access for `localhost` in the browser.
- **Slow first run**: ResNet50 and (if enabled) CLIP/BLIP download once and are cached.
