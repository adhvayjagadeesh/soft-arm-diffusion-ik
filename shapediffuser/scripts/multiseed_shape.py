#!/usr/bin/env python3
"""Multi-seed statistics for the shape-conditioned variant (cond_type: shape).

The core PCC results already have 3-seed statistics (multiseed.py); the
shape-conditioned comparison (results_shape.json, cond_type: shape) was only
ever run on a single seed. This brings it to the same rigor: diffusion+mlp
only (mdn was already skipped in the single-seed shape run to save time, kept
consistent here), reusing the existing seed-0 checkpoints_shape/ (trained
with train.py's seed=0 default) rather than retraining it.

    python scripts/multiseed_shape.py --seeds 0,1,2
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--config", default="configs/shape.yaml")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="multiseed_shape_results.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    py = sys.executable

    per_seed = {}
    for seed in seeds:
        if seed == 0 and os.path.exists("checkpoints_shape/diffusion.pt") \
                and os.path.exists("results_shape.json"):
            print(f"seed 0: reusing existing checkpoints_shape/ and results_shape.json")
            per_seed[seed] = json.load(open("results_shape.json"))
            continue
        existing = f"results_shape_seed{seed}.json"
        if seed != 0 and os.path.exists(existing):
            print(f"seed {seed}: reusing existing {existing}")
            per_seed[seed] = json.load(open(existing))
            continue
        ckpt_dir = f"checkpoints_shape_seed{seed}"
        for model in ("diffusion", "mlp"):
            run([py, "scripts/train.py", "--model", model, "--config", args.config,
                 "--data", args.data, "--seed", str(seed), "--ckpt_dir", ckpt_dir])
        out_path = f"results_shape_seed{seed}.json"
        run([py, "scripts/evaluate_shape.py", "--config", args.config,
             "--ckpt_dir", ckpt_dir, "--out", out_path])
        per_seed[seed] = json.load(open(out_path))

    metric_paths = [
        ("accuracy", "shape_err_best_of_K_mm"),
        ("accuracy", "success_rate_best_of_K"),
        ("multimodality", "diversity"),
        ("timing", "ms_per_target_batchK"),
    ]
    agg = {}
    for model in ("diffusion", "mlp"):
        agg[model] = {}
        for section, key in metric_paths:
            vals = [per_seed[s][model][section][key] for s in seeds
                    if model in per_seed[s] and key in per_seed[s][model].get(section, {})]
            if vals:
                agg[model][f"{section}.{key}"] = {
                    "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                    "n": len(vals), "values": vals,
                }

    with open(args.out, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nWrote {args.out}")
    for model in ("diffusion", "mlp"):
        print(f"\n{model}:")
        for k, v in agg[model].items():
            print(f"  {k}: {v['mean']:.4f} +/- {v['std']:.4f} (n={v['n']})")


if __name__ == "__main__":
    main()
