#!/usr/bin/env python3
"""Evaluate shape-conditioned models (cond_type: shape in configs/shape.yaml).

evaluate.py assumes a 3-D tip target and tip-distance success criteria, which
don't apply once the model is conditioned on the full 24-D whole-body shape:
specifying the whole shape (not just the tip) is expected to remove most of
the actuation redundancy that motivates diffusion for tip-conditioning, so
this script measures whether that collapse actually happens, alongside
reconstruction accuracy.

Metrics:
  accuracy - mean per-point L2 error (mm) between the achieved and target
             backbone (8 points, matching data.py's shape encoding), and
             success rate at eval.success_tol.
  multimodality - diversity (curvature-feature spread, see metrics.py) among
             successful samples for the same shape target: expected to be
             much smaller than the tip-conditioned case if redundancy really
             does collapse.
  timing   - wall-clock ms per target for a batch of K samples.

No obstacle task (E3): avoiding obstacles while matching an exact, fully
specified whole-body shape is a different and less natural downstream task
than tip-conditioned obstacle avoidance, so it's out of scope here.

    python scripts/evaluate_shape.py --config configs/shape.yaml
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
from shapediffuser.metrics import curvature_features, diversity  # noqa: E402
from evaluate import load_model, make_sampler  # noqa: E402


def sample_reachable_shapes(arm, n_targets, shape_points=8, seed=123):
    rng = np.random.default_rng(seed)
    q = torch.as_tensor(rng.uniform(0, 1, size=(n_targets, arm.q_dim)), dtype=torch.float32)
    with torch.no_grad():
        bb = arm.forward(q)["backbone"]
    idx = torch.linspace(0, bb.shape[1] - 1, shape_points).long()
    return bb[:, idx, :].reshape(n_targets, -1)


def shape_error(arm, q, target_shape, shape_points=8):
    """q: (..., q_dim) pressures; target_shape: (..., 3*shape_points), broadcastable
    against q's leading dims. Returns (...,) mean per-point L2 in metres."""
    flat = q.reshape(-1, arm.q_dim)
    with torch.no_grad():
        bb = arm.forward(flat)["backbone"]
    idx = torch.linspace(0, bb.shape[1] - 1, shape_points).long()
    pred = bb[:, idx, :].reshape(*q.shape[:-1], shape_points, 3)
    tgt = target_shape.reshape(*target_shape.shape[:-1], shape_points, 3)
    return (pred - tgt).norm(dim=-1).mean(dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/shape.yaml")
    ap.add_argument("--ckpt_dir", default="checkpoints_shape")
    ap.add_argument("--out", default="results_shape.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ev, d_cfg = cfg["eval"], cfg["data"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arm = PCCArm(**cfg["arm"])
    tol = ev["success_tol"]
    K = ev["n_samples_per_target"]
    shape_points = d_cfg["shape_points"]

    samplers = {}
    for name in ("diffusion", "mlp", "mdn"):
        path = os.path.join(args.ckpt_dir, f"{name}.pt")
        if os.path.exists(path):
            model, norm, ck = load_model(path, device)
            if ck["cond_type"] != "shape":
                print(f"WARNING: {name} checkpoint was trained with cond_type="
                      f"{ck['cond_type']!r}, not 'shape' - skipping.")
                continue
            samplers[name] = make_sampler(model, norm, device, ev["ddim_steps"], ev["guidance"])
        else:
            print(f"(no checkpoint for {name} in {args.ckpt_dir}, skipping)")

    targets = sample_reachable_shapes(arm, ev["n_targets_accuracy"], shape_points)
    results = {}
    for name, sampler in samplers.items():
        qs = sampler(targets, K)  # (B, K, q_dim)
        errs = shape_error(arm, qs, targets.unsqueeze(1), shape_points)  # (B, K)
        results.setdefault(name, {})["accuracy"] = {
            "shape_err_best_of_K_mm": errs.min(dim=1).values.mean().item() * 1000,
            "shape_err_single_mm": errs[:, 0].mean().item() * 1000,
            "success_rate_best_of_K": (errs.min(dim=1).values < tol).float().mean().item(),
        }
        print(name, "shape accuracy:", results[name]["accuracy"])

        divs = []
        n_sub = min(64, targets.shape[0])
        for i in range(n_sub):
            t = targets[i]
            qi = sampler(t.unsqueeze(0), K)[0]
            ei = shape_error(arm, qi, t.unsqueeze(0).expand(K, -1), shape_points)
            ok = ei < tol
            feats = curvature_features(arm, qi)
            divs.append(diversity(feats[ok]))
        results[name]["multimodality"] = {"diversity": float(np.mean(divs))}
        print(name, "diversity (curvature space, lower = less redundancy):",
              results[name]["multimodality"])

        _ = sampler(targets[:2], 2)
        t0 = time.time()
        _ = sampler(targets[:64], K)
        results[name]["timing"] = {
            "ms_per_target_batchK": (time.time() - t0) / 64 * 1000, "K": K, "device": device,
        }
        print(name, "timing:", results[name]["timing"])

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
