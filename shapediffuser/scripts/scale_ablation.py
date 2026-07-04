#!/usr/bin/env python3
"""Dataset-size scaling ablation for the diffusion model.

Trains only the diffusion model (the model the paper's claims are about) at
several training-set sizes, taking prefixes of the existing data/train.npz
(no regeneration needed). Ground-truth modes and evaluation targets are
computed once and reused across every size, since they depend only on the
arm/target distribution, not on the trained model, so E1/E2/E3/E4 numbers are
directly comparable to the main results.json (same targets, same tolerances).

    python scripts/scale_ablation.py --sizes 5000,20000,50000,100000,200000,500000
"""

import argparse
import json
import os
import subprocess
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


def evaluate_diffusion(arm, model, norm, device, cfg, acc_targets, mode_targets,
                       gt_modes, obstacle_targets, obstacles_per_trial):
    ev = cfg["eval"]
    tol = ev["success_tol"]
    K = ev["n_samples_per_target"]
    sampler = make_sampler(model, norm, device, ev["ddim_steps"], ev["guidance"])

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
        recalls.append(mode_recall(gt_modes[ti], feats, ok, radius=ev["mode_match_radius"]))
        divs.append(diversity(feats[ok]))
    multimodality = {
        "mode_recall": float(np.nanmean(recalls)) if recalls else float("nan"),
        "diversity": float(np.mean(divs)) if divs else float("nan"),
    }

    succ = 0
    for t, obstacles in zip(obstacle_targets, obstacles_per_trial):
        qo = sampler(t.unsqueeze(0), K)[0]
        erro = tip_error(arm, qo, t.unsqueeze(0).expand(K, -1))
        with torch.no_grad():
            bb = arm.forward(qo)["backbone"]
        coll = arm.collides(bb, obstacles)
        if ((erro < tol) & (~coll)).any():
            succ += 1
    obstacle_task = {"success_rate": succ / len(obstacle_targets)}

    _ = sampler(acc_targets[:2], 2)
    t0 = time.time()
    _ = sampler(acc_targets[:64], K)
    timing = {"ms_per_target_batchK": (time.time() - t0) / 64 * 1000}

    return {"accuracy": accuracy, "multimodality": multimodality,
            "obstacle_task": obstacle_task, "timing": timing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data")
    ap.add_argument("--sizes", default="5000,20000,50000,100000,200000,500000")
    ap.add_argument("--out", default="scale_ablation_results.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ev = cfg["eval"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arm = PCCArm(**cfg["arm"])
    tol = ev["success_tol"]

    print("Precomputing shared eval targets + ground-truth modes once ...")
    acc_targets = sample_reachable_targets(arm, ev["n_targets_accuracy"])
    mode_targets = sample_reachable_targets(arm, ev["n_targets_modes"], seed=777)
    gt_modes = [
        enumerate_modes(arm, t, pool_size=ev["mode_pool"], tol=tol,
                        dbscan_eps=ev["dbscan_eps"],
                        min_samples=ev.get("dbscan_min_samples", 10))
        for t in mode_targets
    ]
    print("gt modes per target:", [g.shape[0] for g in gt_modes])

    rng = np.random.default_rng(2026)
    trials = ev["n_obstacle_trials"]
    obstacle_targets = sample_reachable_targets(arm, trials, seed=999)
    total_len = cfg["arm"]["n_segments"] * cfg["arm"]["seg_length"]
    obstacles_per_trial = []
    for i in range(trials):
        t = obstacle_targets[i]
        obs = []
        while len(obs) < ev["n_obstacles"]:
            c = rng.uniform(-0.6, 0.6, size=3) * total_len
            c[2] = rng.uniform(0.1, 0.9) * total_len
            if np.linalg.norm(c - t.numpy()) > ev["obstacle_radius"] + 0.03:
                obs.append(np.concatenate([c, [ev["obstacle_radius"]]]))
        obstacles_per_trial.append(torch.as_tensor(np.stack(obs), dtype=torch.float32))

    py = sys.executable
    sizes = [int(s) for s in args.sizes.split(",")]
    results = {}
    for n in sizes:
        ckpt_dir = f"checkpoints_scale{n}"
        print(f"\n=== n_train={n} ===", flush=True)
        subprocess.run(
            [py, "scripts/train.py", "--model", "diffusion", "--config", args.config,
             "--data", args.data, "--n_train", str(n), "--ckpt_dir", ckpt_dir],
            check=True,
        )
        model, norm, _ = load_model(os.path.join(ckpt_dir, "diffusion.pt"), device)
        out = evaluate_diffusion(arm, model, norm, device, cfg, acc_targets,
                                 mode_targets, gt_modes, obstacle_targets,
                                 obstacles_per_trial)
        results[n] = out
        print(f"n_train={n}: {out}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")
    print(f"\n{'n_train':>10} {'best-of-K mm':>13} {'success@5mm':>12} "
          f"{'mode_recall':>12} {'diversity':>10} {'ms/target':>10}")
    for n in sizes:
        r = results[n]
        print(f"{n:>10} {r['accuracy']['tip_err_best_of_K_mm']:>13.3f} "
              f"{r['accuracy']['success_rate_best_of_K']:>12.3f} "
              f"{r['multimodality']['mode_recall']:>12.3f} "
              f"{r['multimodality']['diversity']:>10.2f} "
              f"{r['timing']['ms_per_target_batchK']:>10.2f}")


if __name__ == "__main__":
    main()
