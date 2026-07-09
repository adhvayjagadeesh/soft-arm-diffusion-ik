#!/usr/bin/env python3
"""Full evaluation of the classical optimization-based IK baseline (grad-IK).

The external critique's biggest scientific gap: the paper compared diffusion
only against learned regressors, not against classical optimization-based IK
- the field's default approach for redundant manipulators. grad_ik
(pcc_arm.py) is exactly that: gradient descent through the differentiable
PCC model from random restarts, giving multimodal candidates without any
learning. This evaluates it with the SAME protocol as the learned methods:

  E1 accuracy   - n_targets_accuracy held-out targets, K=32 restarts,
                  best-of-K error and success (grad-IK's "single sample" is
                  its best restart by construction; noted as such).
  E2 mode recall- restart solutions as candidates vs the same enumerated
                  ground-truth modes, same curvature-space radius.
  E3 obstacles  - identical trial generation (seeds 2026/999) to
                  evaluate.py; success iff any restart is accurate AND
                  collision-free.
  E4 timing     - wall-clock per target.

grad-IK requires the true simulator at query time (it optimizes through the
model), so it has no train/test split issue - but it also cannot run where
only samples of the plant are available, which the paper should note.

    python scripts/evaluate_gradik.py --out results_gradik.json
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

from shapediffuser import PCCArm, grad_ik  # noqa: E402
from shapediffuser.metrics import (  # noqa: E402
    tip_error, diversity, curvature_features, enumerate_modes, mode_recall,
    sample_reachable_targets,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n_restarts", type=int, default=32,
                    help="matches the learned methods' K=32 samples per target")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--n_targets_accuracy", type=int, default=None,
                    help="default: config value (512)")
    ap.add_argument("--n_targets_modes", type=int, default=None)
    ap.add_argument("--n_obstacle_trials", type=int, default=None)
    ap.add_argument("--skip_modes", action="store_true")
    ap.add_argument("--out", default="results_gradik.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ev = cfg["eval"]
    arm = PCCArm(**cfg["arm"])
    tol = ev["success_tol"]
    K = args.n_restarts
    n_acc = args.n_targets_accuracy or ev["n_targets_accuracy"]
    n_modes = args.n_targets_modes or ev["n_targets_modes"]
    n_obs = args.n_obstacle_trials or ev["n_obstacle_trials"]

    results = {"n_restarts": K, "iters": args.iters}

    # ---------------- E1: accuracy ---------------- #
    targets = sample_reachable_targets(arm, n_acc)
    t0 = time.time()
    errs_best, errs_all = [], []
    for i, t in enumerate(targets):
        qs = grad_ik(arm, t, n_restarts=K, iters=args.iters)  # sorted best-first
        e = tip_error(arm, qs, t.unsqueeze(0).expand(K, -1))
        errs_best.append(e.min().item())
        errs_all.append(e.numpy())
        if (i + 1) % 50 == 0:
            print(f"E1 {i + 1}/{n_acc} (mean best so far "
                  f"{np.mean(errs_best) * 1000:.2f}mm)", flush=True)
    e1_time = time.time() - t0
    errs_best = np.array(errs_best)
    results["accuracy"] = {
        "tip_err_best_of_K_mm": float(errs_best.mean() * 1000),
        "success_rate_best_of_K": float((errs_best < tol).mean()),
        "note": "grad-IK's output is inherently best-of-restarts; no single-sample analogue",
    }
    results["timing"] = {"ms_per_target_batchK": e1_time / n_acc * 1000, "K": K}
    print("E1:", results["accuracy"], flush=True)
    print("E4:", results["timing"], flush=True)

    # ---------------- E2: multimodality ---------------- #
    if not args.skip_modes:
        mode_targets = sample_reachable_targets(arm, n_modes, seed=777)
        recalls, divs = [], []
        for ti, t in enumerate(mode_targets):
            gt = enumerate_modes(arm, t, pool_size=ev["mode_pool"], tol=tol,
                                 dbscan_eps=ev["dbscan_eps"],
                                 min_samples=ev.get("dbscan_min_samples", 10))
            if gt.shape[0] == 0:
                continue
            qs = grad_ik(arm, t, n_restarts=K, iters=args.iters)
            e = tip_error(arm, qs, t.unsqueeze(0).expand(K, -1))
            ok = e < tol
            feats = curvature_features(arm, qs)
            recalls.append(mode_recall(gt, feats, ok, radius=ev["mode_match_radius"]))
            divs.append(diversity(feats[ok]))
            print(f"E2 target {ti + 1}/{n_modes}: {gt.shape[0]} gt modes, "
                  f"recall {recalls[-1]:.2f}", flush=True)
        results["multimodality"] = {
            "mode_recall": float(np.nanmean(recalls)) if recalls else float("nan"),
            "diversity": float(np.mean(divs)) if divs else float("nan"),
        }
        print("E2:", results["multimodality"], flush=True)

    # ---------------- E3: obstacle task (identical generation to evaluate.py) --- #
    rng = np.random.default_rng(2026)
    reach = sample_reachable_targets(arm, n_obs, seed=999)
    total_len = cfg["arm"]["n_segments"] * cfg["arm"]["seg_length"]
    succ = 0
    for i in range(n_obs):
        t = reach[i]
        obs = []
        while len(obs) < ev["n_obstacles"]:
            c = rng.uniform(-0.6, 0.6, size=3) * total_len
            c[2] = rng.uniform(0.1, 0.9) * total_len
            if np.linalg.norm(c - t.numpy()) > ev["obstacle_radius"] + 0.03:
                obs.append(np.concatenate([c, [ev["obstacle_radius"]]]))
        obstacles = torch.as_tensor(np.stack(obs), dtype=torch.float32)
        qs = grad_ik(arm, t, n_restarts=K, iters=args.iters)
        e = tip_error(arm, qs, t.unsqueeze(0).expand(K, -1))
        with torch.no_grad():
            bb = arm.forward(qs)["backbone"]
        coll = arm.collides(bb, obstacles)
        if ((e < tol) & (~coll)).any():
            succ += 1
        if (i + 1) % 25 == 0:
            print(f"E3 {i + 1}/{n_obs} (success so far {succ / (i + 1):.3f})", flush=True)
    results["obstacle_task"] = {"success_rate": succ / n_obs}
    print("E3:", results["obstacle_task"], flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
