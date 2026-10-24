"""
scripts/run_experiments.py

Trains and evaluates a set of configs x seeds in the right order, skipping
anything already finished. Safe to re-run after a Colab disconnect: it picks
up exactly where it stopped.

    python scripts/run_experiments.py --stage frozen    --seeds 1,2,3
    python scripts/run_experiments.py --stage finetuned --seeds 1,2,3   # after frozen
    python scripts/run_experiments.py --stage paper     --seeds 1
    python scripts/run_experiments.py --configs hybrid_frozen --seeds 1 --decodings greedy,beam1,beam3,beam5,beam10

Each train/evaluate call runs as its own process, so GPU memory is released
between runs.
"""

import os
import sys
import time
import argparse
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.utils.run_io import run_dir, read_json  # noqa: E402

STAGES = {
    "frozen": ["lstm_frozen", "gru_frozen", "hybrid_frozen"],
    "finetuned": ["lstm_finetuned", "gru_finetuned", "hybrid_finetuned"],
    "paper": ["paper_protocol"],
    "attention": ["attention", "transformer"],   # Part 2, needs --mode grid features
    "encoders": ["hybrid_clip", "hybrid_vgg16"],  # needs --mode clip / vgg16 features
}


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=list(STAGES) + ["all"], default=None)
    p.add_argument("--configs", default=None, help="Comma-separated config names (overrides --stage).")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--decodings", default="greedy,beam3,beam5")
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--extra", default="", help="Extra arguments passed to train.py, e.g. '--epochs 2'.")
    args = p.parse_args()

    if args.configs:
        configs = args.configs.split(",")
    elif args.stage == "all":
        configs = STAGES["frozen"] + STAGES["finetuned"] + STAGES["paper"]
    elif args.stage:
        configs = STAGES[args.stage]
    else:
        p.error("give --stage or --configs")
    seeds = [int(s) for s in args.seeds.split(",")]

    t0 = time.time()
    for cfg in configs:
        for seed in seeds:
            rdir = run_dir(args.runs_dir, cfg, seed)
            if not read_json(os.path.join(rdir, "state.json"), {}).get("done"):
                run([sys.executable, "train.py", "--config", f"experiments/config_{cfg}.yaml",
                     "--seed", str(seed), "--runs_dir", args.runs_dir] + args.extra.split())
            eval_file = os.path.join(rdir, "eval", "metrics.json")
            wanted = set(args.decodings.split(","))
            have = set(read_json(eval_file, {}).get("decodings", {}))
            if not args.skip_eval and not wanted <= have:
                run([sys.executable, "evaluate.py", "--config_name", cfg, "--seed", str(seed),
                     "--runs_dir", args.runs_dir, "--decodings", args.decodings])
            else:
                print(f"[done] {cfg} seed {seed}")
    print(f"\nAll requested runs finished in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
