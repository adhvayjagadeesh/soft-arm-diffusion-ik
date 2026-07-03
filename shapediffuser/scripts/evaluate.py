#!/usr/bin/env python3
"""Run the full evaluation suite and write results.json.

Experiments (mapping to the paper's claims):
  E1 accuracy       — tip error over held-out reachable targets, best-of-K and
                      single-sample, all methods (+ grad-IK reference).
  E2 multimodality  — ground-truth mode enumeration per target; mode recall and
                      solution diversity per method. THE headline experiment.
  E3 obstacle task  — same tip target, random obstacles; success iff any of K
                      sampled solutions is accurate AND collision-free.
  E4 efficiency     — wall-clock sampling time per batch of K solutions.
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

from shapediffuser import PCCArm, BabblingDataset, Normalizer, build_model, grad_ik  # noqa: E402
from shapediffuser.metrics import (  # noqa: E402
    tip_error, diversity, enumerate_modes, mode_recall, sample_reachable_targets,
)


def load_model(path: str, device: str):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["model_name"], ck["cond_dim"], ck["q_dim"], ck["model_cfg"])
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, Normalizer.from_state(ck["cond_norm"]), ck


def make_sampler(model, norm, device, ddim_steps, guidance):
    """Returns f(target_tips (B,3), K) -> pressures (B, K, q_dim) in [0,1]."""

    def f(targets: torch.Tensor, k: int) -> torch.Tensor:
        cond = norm.encode(targets.to(device))
        kwargs = {}
        if hasattr(model, "alphas_cumprod"):  # diffusion
            kwargs = {"steps": ddim_steps, "guidance": guidance}
        q_scaled = model.sample(cond, n_samples=k, **kwargs)
        return BabblingDataset.q_to_pressures(q_scaled).cpu()

    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--skip_modes", action="store_true",
                    help="skip E2 (slow mode-enumeration sweep)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ev = cfg["eval"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arm = PCCArm(**cfg["arm"])
    tol = ev["success_tol"]
    K = ev["n_samples_per_target"]

    samplers = {}
    for name in ("diffusion", "mdn", "mlp"):
        path = os.path.join(args.ckpt_dir, f"{name}.pt")
        if os.path.exists(path):
            model, norm, ck = load_model(path, device)
            if ck["cond_type"] != "tip":
                print(f"NOTE: {name} was trained shape-conditioned; E2/E3 assume tip.")
            samplers[name] = make_sampler(model, norm, device,
                                          ev["ddim_steps"], ev["guidance"])
        else:
            print(f"(no checkpoint for {name}, skipping)")

    results = {}

    # ---------------- E1: accuracy ---------------- #
    targets = sample_reachable_targets(arm, ev["n_targets_accuracy"])
    for name, sampler in samplers.items():
        qs = sampler(targets, K)  # (B, K, q_dim)
        errs = tip_error(arm, qs, targets.unsqueeze(1))  # (B, K)
        results.setdefault(name, {})["accuracy"] = {
            "tip_err_best_of_K_mm": errs.min(dim=1).values.mean().item() * 1000,
            "tip_err_single_mm": errs[:, 0].mean().item() * 1000,
            "success_rate_best_of_K": (errs.min(dim=1).values < tol).float().mean().item(),
        }
        print(name, "E1:", results[name]["accuracy"])

    # grad-IK reference (not learned; slow)
    gi_err = []
    for t in targets[:32]:
        qs = grad_ik(arm, t, n_restarts=16, iters=200)
        gi_err.append(tip_error(arm, qs[:1], t.unsqueeze(0)).item())
    results["grad_ik"] = {"accuracy": {"tip_err_best_mm": float(np.mean(gi_err)) * 1000}}

    # ---------------- E2: multimodality ---------------- #
    if not args.skip_modes:
        mode_targets = sample_reachable_targets(arm, ev["n_targets_modes"], seed=777)
        agg = {name: {"recall": [], "diversity": []} for name in samplers}
        n_modes_all = []
        for ti, t in enumerate(mode_targets):
            gt = enumerate_modes(arm, t, pool_size=ev["mode_pool"], tol=tol,
                                 dbscan_eps=ev["dbscan_eps"])
            n_modes_all.append(gt.shape[0])
            print(f"target {ti}: {gt.shape[0]} ground-truth modes")
            if gt.shape[0] == 0:
                continue
            for name, sampler in samplers.items():
                qs = sampler(t.unsqueeze(0), K)[0]  # (K, q_dim)
                errs = tip_error(arm, qs, t.unsqueeze(0).expand(K, -1))
                ok = errs < tol
                agg[name]["recall"].append(
                    mode_recall(gt, qs, ok, radius=ev["mode_match_radius"]))
                agg[name]["diversity"].append(diversity(qs[ok]))
        results["gt_modes_mean"] = float(np.mean(n_modes_all)) if n_modes_all else 0.0
        for name in samplers:
            results[name]["multimodality"] = {
                "mode_recall": float(np.nanmean(agg[name]["recall"])),
                "diversity": float(np.mean(agg[name]["diversity"])),
            }
            print(name, "E2:", results[name]["multimodality"])

    # ---------------- E3: obstacle-conditioned selection ---------------- #
    rng = np.random.default_rng(2026)
    succ = {name: 0 for name in samplers}
    trials = ev["n_obstacle_trials"]
    reach = sample_reachable_targets(arm, trials, seed=999)
    total_len = cfg["arm"]["n_segments"] * cfg["arm"]["seg_length"]
    for i in range(trials):
        t = reach[i]
        obs = []
        while len(obs) < ev["n_obstacles"]:
            c = rng.uniform(-0.6, 0.6, size=3) * total_len
            c[2] = rng.uniform(0.1, 0.9) * total_len
            if np.linalg.norm(c - t.numpy()) > ev["obstacle_radius"] + 0.03:
                obs.append(np.concatenate([c, [ev["obstacle_radius"]]]))
        obstacles = torch.as_tensor(np.stack(obs), dtype=torch.float32)
        for name, sampler in samplers.items():
            qs = sampler(t.unsqueeze(0), K)[0]
            errs = tip_error(arm, qs, t.unsqueeze(0).expand(K, -1))
            with torch.no_grad():
                bb = arm.forward(qs)["backbone"]
            coll = arm.collides(bb, obstacles)
            if ((errs < tol) & (~coll)).any():
                succ[name] += 1
    for name in samplers:
        results[name]["obstacle_task"] = {"success_rate": succ[name] / trials}
        print(name, "E3:", results[name]["obstacle_task"])

    # ---------------- E4: efficiency ---------------- #
    t64 = sample_reachable_targets(arm, 64, seed=5)
    for name, sampler in samplers.items():
        _ = sampler(t64[:2], 2)  # warm-up
        t0 = time.time()
        _ = sampler(t64, K)
        results[name]["timing"] = {
            "ms_per_target_batchK": (time.time() - t0) / 64 * 1000, "K": K,
            "device": device,
        }
        print(name, "E4:", results[name]["timing"])

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
