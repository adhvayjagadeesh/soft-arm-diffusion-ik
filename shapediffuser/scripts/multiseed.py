#!/usr/bin/env python3
"""Train + evaluate diffusion/mlp/mdn across multiple seeds and aggregate.

Reuses one fixed dataset (data generation isn't reseeded) and only varies
model initialization + minibatch order per seed, so all seeds are compared on
the exact same held-out data and the exact same evaluation targets
(sample_reachable_targets uses fixed default seeds regardless of --seed) -
an apples-to-apples paired comparison of training-run variance.

    python scripts/multiseed.py --seeds 0,1,2
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
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="multiseed_results.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    py = sys.executable

    per_seed = {}
    for seed in seeds:
        ckpt_dir = f"checkpoints_seed{seed}"
        for model in ("diffusion", "mlp", "mdn"):
            run([py, "scripts/train.py", "--model", model, "--config", args.config,
                 "--data", args.data, "--seed", str(seed), "--ckpt_dir", ckpt_dir])
        out_path = f"results_seed{seed}.json"
        run([py, "scripts/evaluate.py", "--config", args.config,
             "--ckpt_dir", ckpt_dir, "--out", out_path])
        per_seed[seed] = json.load(open(out_path))

    metric_paths = [
        ("accuracy", "tip_err_best_of_K_mm"),
        ("accuracy", "success_rate_best_of_K"),
        ("multimodality", "mode_recall"),
        ("multimodality", "diversity"),
        ("obstacle_task", "success_rate"),
        ("timing", "ms_per_target_batchK"),
    ]
    agg = {}
    for model in ("diffusion", "mlp", "mdn"):
        agg[model] = {}
        for section, key in metric_paths:
            vals = [per_seed[s][model][section][key] for s in seeds
                    if model in per_seed[s] and key in per_seed[s][model].get(section, {})]
            if vals:
                agg[model][f"{section}.{key}"] = {
                    "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                    "n": len(vals), "values": vals,
                }
    agg["gt_modes_mean"] = {
        "mean": float(np.mean([per_seed[s]["gt_modes_mean"] for s in seeds])),
        "values": [per_seed[s]["gt_modes_mean"] for s in seeds],
    }

    with open(args.out, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nWrote {args.out}")
    for model in ("diffusion", "mlp", "mdn"):
        print(f"\n{model}:")
        for k, v in agg[model].items():
            print(f"  {k}: {v['mean']:.4f} +/- {v['std']:.4f} (n={v['n']})")


if __name__ == "__main__":
    main()
