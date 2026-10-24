"""
dashboard/engine.py

Live captioning for the dashboard: loads the trained models once and runs any
photo through all of them. No Streamlit code here, so it can be tested alone.

ResNet50 runs ONCE per photo and gives all three feature types the models need
(conv4 maps for the fine-tuned model, the 7x7 grid for the attention models,
the pooled vector for frozen models). CLIP and BLIP are optional (PyTorch).
"""

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# Models offered for live captioning: config -> (label, needs, heatmap method)
LIVE_MODELS = {
    "hybrid_frozen": ("C3 Hybrid (classic, no attention)", "resnet", "Grad-CAM"),
    "hybrid_finetuned": ("C6 Hybrid, fine-tuned", "resnet", None),   # used only if its weights are present
    "attention": ("A1 LSTM + attention", "resnet", "attention"),
    "transformer": ("A2 Transformer", "resnet", "attention"),
    "hybrid_clip": ("Hybrid + CLIP (best trained model)", "clip", None),
}
BLIP_ID = "Salesforce/blip-image-captioning-large"


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def available_models(runs_dir: str) -> dict:
    out = {}
    for cfg, (label, needs, heat) in LIVE_MODELS.items():
        if not os.path.exists(os.path.join(runs_dir, cfg, "seed1", "best.weights.h5")):
            continue
        if needs == "clip" and not torch_available():
            continue
        out[cfg] = {"label": label, "needs": needs, "heatmap": heat}
    return out


class ResNetFeatures:
    """One ResNet50 pass -> {'conv4', 'grid', 'pooled'} for a single image."""

    def __init__(self, weights: str | None = "imagenet"):
        import keras
        from keras.applications.resnet50 import ResNet50

        base = ResNet50(weights=weights, include_top=False, pooling="avg", input_shape=(224, 224, 3))
        self.model = keras.Model(base.input, [base.get_layer("conv4_block6_out").output,
                                              base.get_layer("conv5_block3_out").output, base.output])

    def __call__(self, path: str) -> dict:
        from src.feature_extraction import load_and_preprocess
        c4, grid, pooled = self.model.predict_on_batch(load_and_preprocess(path)[None])
        return {"conv4": c4[0], "grid": grid[0], "pooled": pooled[0]}


class LiveModel:
    def __init__(self, cfg: str, runs_dir: str, processed_dir: str, tokenizer):
        from src.utils.run_io import load_trained_model, feature_mode
        from src.utils.beam_search import CaptionGenerator

        self.cfg = cfg
        self.tok = tokenizer
        self.model, self.config = load_trained_model(os.path.join(runs_dir, cfg, "seed1"),
                                                     tokenizer.vocab_size, processed_dir)
        self.mode = feature_mode(self.config)
        self.gen = CaptionGenerator(self.model, tokenizer, self.config["max_length"])

    def state(self, feats: dict) -> np.ndarray:
        x = feats[self.mode]
        return self.gen.encode_images(np.asarray(x, np.float32)[None])[0]

    def run(self, feats: dict, beam_width: int = 5) -> dict:
        from src import analysis as an

        st = self.state(feats)
        greedy = self.gen.greedy(st[None])[0]
        beam = greedy if beam_width == 1 else self.gen.beam(st[None], beam_width)[0]
        out = {"greedy": greedy, "beam": beam, "probs": self.gen.word_probabilities(st, beam)}
        heat = LIVE_MODELS[self.cfg][2]
        if heat == "attention":
            out["maps"] = self.gen.attention_maps(st, beam)
        elif heat == "Grad-CAM" and "grid" in feats:
            out["maps"] = an.gradcam_maps(self.model, feats["grid"], beam, self.tok, self.config["max_length"])
        out["trace"] = self.gen.beam_trace(st, 3)
        out["trace"]["greedy"] = greedy
        return out


class Blip:
    def __init__(self, model_id: str = BLIP_ID):
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id).to(self.device).eval()

    def __call__(self, path: str, num_beams: int = 3) -> str:
        from PIL import Image
        inputs = self.processor(images=Image.open(path).convert("RGB"), return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            ids = self.model.generate(**inputs, num_beams=num_beams, max_new_tokens=30)
        return self.processor.decode(ids[0], skip_special_tokens=True).strip()


def heat_overlay(image, att_map: np.ndarray, longest: int = 480, alpha: float = 0.55):
    """PIL image with a 7x7 map upsampled and blended on top (aspect ratio kept)."""
    from PIL import Image
    import matplotlib

    img = image.convert("RGB")
    img.thumbnail((longest, longest))
    m = np.asarray(att_map, np.float32).reshape(7, 7)
    m = m / m.max() if m.max() > 0 else m
    up = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize(img.size, Image.BICUBIC), np.float32) / 255
    colored = (matplotlib.colormaps["inferno"](up)[..., :3] * 255).astype(np.uint8)
    return Image.blend(img, Image.fromarray(colored), alpha)
