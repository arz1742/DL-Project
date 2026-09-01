# Literature Review

## Paper 1: An Image Captioning Algorithm Based on the Hybrid Deep Learning Technique (CNN+GRU)

**Authors:** Rana Adnan Ahmad, Muhammad Azhar, Hina Sattar — COMSATS University Islamabad
**Reference:** arXiv:2301.02440 (2023)

### Core Idea
Proposes a CNN encoder + GRU decoder framework, extended with a **semantic attribute vector** (extracted via a Multiple Instance Learning model, following Yao et al.) and a **semantic reconstructor** module. The reconstructor uses the decoder's hidden states to reconstruct the image's semantic representation, and this "reconstruction score" is combined with the standard likelihood score during both training and caption selection at inference. The architecture builds on the "LSTM-A5" template (Yao et al.) but swaps in GRU for the recurrent decoder.

### Input → Output
Input: an image. Output: a caption sentence, selected via beam search + position reset, scored by a combination of log-likelihood and semantic reconstruction score.

### Dataset
MS COCO (130,000 images, 4 captions each for training; 5,000 for validation, 5,000 for testing).

### Reported Results
BLEU@1–4 = 0.751 / 0.578 / 0.439 / 0.335, METEOR 0.259, ROUGE-L 0.545, CIDEr 1.035 (offline COCO test set) — outperforming their primary baseline, LSTM-A5.

### Limitations
- Standard CNN-LSTM pairs are noted as suffering from poor time/space complexity efficiency; this paper's GRU + reconstructor approach partially addresses this but adds its own complexity (attribute extraction model + reconstructor module).
- The approach compensates for weak scene understanding via reconstruction scoring rather than allowing the model to dynamically attend to specific image regions during generation (no attention mechanism).
- Requires a separately-trained attribute extraction model, adding a non-trivial additional training pipeline beyond the core CNN-GRU encoder-decoder.

---

## Paper 2: Image Captioning Deep Learning Model Using ResNet50 Encoder and Hybrid LSTM–GRU Decoder Optimized with Beam Search

**Authors:** P. V. Kavitha, V. Karpagam — Sri Ramakrishna Engineering College, Coimbatore
**Reference:** Automatika, Vol. 66, No. 3, pp. 394–410, 2025. DOI: 10.1080/00051144.2025.2485695

### Core Idea
Systematically compares CNN encoders (VGG16, InceptionV3, ResNet50, DenseNet121) with an LSTM decoder on Flickr8k, finding ResNet50 gives the lowest loss and highest accuracy. Then compares LSTM against several LSTM/GRU variants (Stacked LSTM, Bi-LSTM, GRU, Stacked GRU, Bi-GRU) as decoders, before proposing a **hybrid LSTM–GRU decoder** (LSTM output feeding into a GRU layer) combined with **beam search** decoding (beam width 5, chosen after testing widths 3/5/10) as the best-performing configuration.

### Input → Output
Input: an image. Output: a caption sentence generated via beam search over the hybrid LSTM–GRU decoder's output distribution.

### Dataset
Flickr8k (8,092 images, 5 captions each; 1,000 for validation, 1,000 for testing) — same dataset used in this project.

### Reported Results
- ResNet50 + LSTM: accuracy 0.8459, loss 0.5959 (best single encoder-decoder baseline)
- ResNet50 + Hybrid LSTM–GRU: accuracy 0.8932, loss 0.4013 (best overall)
- BLEU-1 with beam search: 0.6034 (vs. 0.5142 with greedy search on the same model)
- METEOR 0.3124, SPICE 0.2404 — highest among all models compared, including a DenseNet121+LSTM baseline and an M²-Transformer reference point

### Limitations
- Beam search adds inference-time computational overhead compared to greedy decoding.
- The hybrid decoder increases architectural complexity without addressing the deeper issue shared with Paper 1: the CNN feature is a static input, with no attention mechanism for dynamic region focus.
- The paper's own conclusion notes scalability concerns for larger datasets (MSCOCO, Flickr30k) and reduced feasibility for real-time applications due to the computational cost of the hybrid LSTM+GRU combination.
- Does not isolate LSTM-only vs. GRU-only performance as a controlled, standalone comparison — it evaluates the hybrid combination directly.

---

## Gap This Project Addresses

Both papers share the same core limitation: the CNN-extracted image feature is used as a static input, so neither model can dynamically re-focus on different image regions while generating different words. Neither paper runs a controlled, isolated comparison of the specific design choices they each made independently.

This project does not close the attention gap (explicitly deferred to future work, see README), but directly runs the missing controlled comparison:

1. **LSTM vs. GRU decoder** — Paper 1 uses GRU, Paper 2's best baseline decoder is LSTM (before hybridizing); neither directly compares LSTM-only vs. GRU-only under identical conditions. This project's `lstm_frozen` vs. `gru_frozen` configs answer this directly.
2. **Frozen vs. fine-tuned CNN encoder** — neither paper runs this comparison explicitly. This project's `*_frozen` vs. `*_finetuned` configs answer this directly.

Results are reported in `experiments/results/` after training, evaluated with BLEU-1 through BLEU-4 on the Flickr8k test set, consistent with the metric both mother papers use.
