#!/usr/bin/env python3
"""Ablate the diffusion sampler's guidance weight and DDIM step count.

Reuses the already-trained diffusion checkpoint (no retraining): for each
value in a sweep, resamples the E1 accuracy targets and the E2 mode targets
(ground-truth modes are enumerated once and reused across the whole sweep,
since they don't depend on guidance/steps) and reports accuracy, mode
recall/diversity, and wall-clock timing.

    python scripts/ablate.py --sweep guidance --values 0.5,1.0,1.5,2.0,3.0,4.0
    python scripts/ablate.py --sweep ddim_steps --values 5,10,20,50,100
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shapediffuser import PCCArm  # noqa: E402
from shapediffuser.metrics import (  # noqa: E402
    tip_error, diversity, curvature_features, enumerate_modes, mode_recall,
    sample_reachable_targets,
)
from evaluate import load_model, make_sampler  # noqa: E402


def run_one(arm, model, norm, device, ddim_steps, guidance, tol, K,
            acc_targets, mode_targets, gt_modes):
    sampler = make_sampler(model, norm, device, ddim_steps, guidance)

    qs = sampler(acc_targets, K)
    errs = tip_error(arm, qs, acc_targets.unsqueeze(1))
    accuracy = {
        "tip_err_best_of_K_mm": errs.min(dim=1).values.mean().item() * 1000,
        "tip_err_single_mm": errs[:, 0].mean().item() * 1000,
        "success_rate_best_of_K": (errs.min(dim=1).values < tol).float().mean().item(),
    }

    recalls, divs = [], []
    for ti, t in enumerate(mode_targets):
        if gt_modes[ti].shape[0] == 0:
            continue
        qm = sampler(t.unsqueeze(0), K)[0]
        errm = tip_error(arm, qm, t.unsqueeze(0).expand(K, -1))
        ok = errm < tol
        feats = curvature_features(arm, qm)
        recalls.append(mode_recall(gt_modes[ti], feats, ok, radius=15.0))
        divs.append(diversity(feats[ok]))
    multimodality = {
        "mode_recall": float(np.nanmean(recalls)) if recalls else float("nan"),
        "diversity": float(np.mean(divs)) if divs else float("nan"),
    }

    _ = sampler(acc_targets[:2], 2)  # warm-up
    t0 = time.time()
    _ = sampler(acc_targets[:64], K)
    ms_per_target = (time.time() - t0) / 64 * 1000

    return {"accuracy": accuracy, "multimodality": multimodality,
            "ms_per_target_batchK": ms_per_target}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--sweep", required=True, choices=["guidance", "ddim_steps"])
    ap.add_argument("--values", required=True, help="comma-separated sweep values")
    ap.add_argument("--n_targets_accuracy", type=int, default=128,
                    help="subset of E1 targets for speed (default config uses 512)")
    ap.add_argument("--n_targets_modes", type=int, default=10,
                    help="subset of E2 targets for speed (default config uses 20)")
    ap.add_argument("--out", default="ablation_results.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ev = cfg["eval"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arm = PCCArm(**cfg["arm"])
    tol = ev["success_tol"]
    K = ev["n_samples_per_target"]

    ckpt_path = os.path.join(args.ckpt_dir, "diffusion.pt")
    model, norm, ck = load_model(ckpt_path, device)

    acc_targets = sample_reachable_targets(arm, args.n_targets_accuracy)
    mode_targets = sample_reachable_targets(arm, args.n_targets_modes, seed=777)

    print(f"Enumerating ground-truth modes once for {args.n_targets_modes} targets "
          f"(pool={ev['mode_pool']}, reused across the whole sweep) ...")
    gt_modes = [
        enumerate_modes(arm, t, pool_size=ev["mode_pool"], tol=tol,
                        dbscan_eps=ev["dbscan_eps"],
                        min_samples=ev.get("dbscan_min_samples", 10))
        for t in mode_targets
    ]
    print("gt modes per target:", [g.shape[0] for g in gt_modes])

    values = [float(v) for v in args.values.split(",")]
    results = {}
    for v in values:
        if args.sweep == "guidance":
            steps, guidance = ev["ddim_steps"], v
        else:
            steps, guidance = int(v), ev["guidance"]
        out = run_one(arm, model, norm, device, steps, guidance, tol, K,
                      acc_targets, mode_targets, gt_modes)
        results[str(v)] = out
        print(f"{args.sweep}={v}: {out}")

    with open(args.out, "w") as f:
        json.dump({"sweep": args.sweep, "results": results}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
