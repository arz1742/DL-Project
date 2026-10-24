"""
scripts/rescore_meteor.py

Adds METEOR to evaluated runs that are missing it (e.g. if Java was not
available when evaluate.py ran). Reads each run's predictions.csv, so no
model or GPU is needed.

    python scripts/rescore_meteor.py
"""

import os
import sys
import glob
import json
import argparse

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation import meteor, _tok  # noqa: E402
from src.utils.run_io import write_json  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--processed_dir", default="data/processed")
    args = p.parse_args()

    with open(os.path.join(args.processed_dir, "test_captions.json")) as f:
        refs = json.load(f)

    for mpath in sorted(glob.glob(os.path.join(args.runs_dir, "*", "seed*", "eval", "metrics.json"))):
        with open(mpath) as f:
            m = json.load(f)
        missing = [d for d, v in m["decodings"].items() if "meteor" not in v["coco"]]
        if not missing:
            continue
        preds = pd.read_csv(os.path.join(os.path.dirname(mpath), "predictions.csv"), keep_default_na=False)
        for dec in missing:
            sub = preds[preds.decoding == dec]
            gts = {r.image: [" ".join(_tok(c)) for c in refs[r.image]] for r in sub.itertuples()}
            res = {r.image: [" ".join(_tok(r.caption)) or "<empty>"] for r in sub.itertuples()}
            score = meteor(gts, res)
            if not score:
                print("METEOR still failing; stopping.")
                return
            m["decodings"][dec]["coco"].update(score)
            print(f"{m['config_name']} seed {m['seed']} {dec}: METEOR {score['meteor']:.4f}")
        write_json(mpath, m)


if __name__ == "__main__":
    main()
