"""
scripts/eval_pretrained.py

Part 2, W3: captions the 1000 test images with a PRETRAINED modern captioner
(BLIP, Salesforce 2022) — no training — and scores it with exactly the same
metrics as our own models, so it shows up in Table D as a "ceiling".

    python scripts/eval_pretrained.py --model base  --image_dir /content/data_local/raw/Images
    python scripts/eval_pretrained.py --model large --image_dir /content/data_local/raw/Images

Writes runs/blip_<model>/seed0/eval/metrics.json and predictions.csv.

BLIP was pretrained on ~129M web images and fine-tuned on COCO captions, so it
is not a like-for-like comparison: it shows how far a large modern model is from
models trained only on 6,091 Flickr8k images. Needs PyTorch + transformers
(both preinstalled on Colab). About 3-6 minutes on an L4.
"""

import os
import sys
import json
import faulthandler
import time
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.scoring import score_captions, print_summary, write_outputs  # noqa: E402  (no TensorFlow)
from src.utils.run_io import run_dir, write_json  # noqa: E402

MODELS = {"base": "Salesforce/blip-image-captioning-base",
          "large": "Salesforce/blip-image-captioning-large"}


def blip_captions(image_paths: list[str], model_id: str, batch_size: int = 32, num_beams: int = 3,
                  max_new_tokens: int = 30) -> list[str]:
    import torch
    from PIL import Image
    from transformers import BlipForConditionalGeneration, BlipProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()

    out = []
    for i in range(0, len(image_paths), batch_size):
        images = [Image.open(p).convert("RGB") for p in image_paths[i:i + batch_size]]
        inputs = processor(images=images, return_tensors="pt").to(device, dtype)
        with torch.no_grad():
            ids = model.generate(**inputs, num_beams=num_beams, max_new_tokens=max_new_tokens)
        out += [c.strip() for c in processor.batch_decode(ids, skip_special_tokens=True)]
        print(f"  {min(i + batch_size, len(image_paths))}/{len(image_paths)}", end="\r", flush=True)
    print()
    return out


def main():
    faulthandler.enable()  # a hard crash prints where it happened instead of dying silently
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODELS), default="base")
    p.add_argument("--image_dir", default="data/raw/Images")
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--num_beams", type=int, default=3)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no_meteor", action="store_true")
    args = p.parse_args()

    with open(os.path.join(args.processed_dir, "test_captions.json")) as f:
        refs = json.load(f)
    with open(os.path.join(args.processed_dir, "train_captions.json")) as f:
        train_caps = json.load(f)
    names = sorted(refs)[:args.limit] if args.limit else sorted(refs)
    refs = {n: refs[n] for n in names}

    config_name = f"blip_{args.model}"
    rdir = run_dir(args.runs_dir, config_name, 0)
    out_dir = os.path.join(rdir, "eval")
    os.makedirs(out_dir, exist_ok=True)
    write_json(os.path.join(rdir, "state.json"), {"done": True, "pretrained": MODELS[args.model]})

    print(f"=== {MODELS[args.model]} on {len(names)} test images (beam {args.num_beams}) ===")
    t0 = time.time()
    caps = blip_captions([os.path.join(args.image_dir, n) for n in names], MODELS[args.model],
                         num_beams=args.num_beams)
    secs = time.time() - t0
    print(f"Captioned {len(caps)} images in {secs:.0f}s, e.g. {names[0]}: \"{caps[0]}\". Scoring...")
    hyps = dict(zip(names, caps))

    dec = f"beam{args.num_beams}"
    m, rows = score_captions(refs, hyps, train_caps, dec, use_meteor=not args.no_meteor)
    m.update(beam_width=args.num_beams, decode_seconds=round(secs, 1))
    print_summary(dec, m, secs)
    metrics = {"config_name": config_name, "seed": 0, "split": "test", "n_images": len(names),
               "best_epoch": None, "train_seconds": None, "pretrained_model": MODELS[args.model],
               "token_metrics": None, "decodings": {dec: m}}
    write_outputs(out_dir, metrics, rows)


if __name__ == "__main__":
    main()
