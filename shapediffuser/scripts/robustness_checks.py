#!/usr/bin/env python3
"""Four robustness checks closing holes from the second adversarial pass.

  A. grad-IK compute ablation - iterations x restarts grid: is the paper's
     "10x slower" query-time claim robust, or does a cheaper grad-IK
     configuration match diffusion's accuracy at similar cost?
  B. grad-IK restart-RNG variance - 3 torch seeds on a target subset: how
     stable is the single-run number in Table I?
  C. Mode-recall hyperparameter sensitivity - DBSCAN eps x match radius grid
     on the standard 20 mode targets, for diffusion (seed-0 checkpoint) and
     grad-IK candidates: does the E2 conclusion depend on the calibrated
     values (eps=12, radius=15)?
  D. Elastica settle-time convergence - random actuations settled for 1.5s
     vs 3.0s: is the quasi-static assumption converged?
  E. torque_gain calibration sweep (fourth pass) - committed rerun of the
     ad-hoc check from the original transfer study (previously recorded only
     in a commit message): does any Elastica couple gain spanning two orders
     of magnitude around the default align the two simulators' tips for
     shared actuations, or is the mismatch structural/directional?

    python scripts/robustness_checks.py --out robustness_checks.json
    python scripts/robustness_checks.py --checks E   # merge one check into an existing out-file
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

from shapediffuser import PCCArm, BabblingDataset, grad_ik  # noqa: E402
from shapediffuser.elastica_arm import ElasticaArm  # noqa: E402
from shapediffuser.metrics import (  # noqa: E402
    tip_error, curvature_features, mode_recall, sample_reachable_targets,
)
from shapediffuser.metrics import enumerate_modes as _enumerate  # noqa: E402
from evaluate import load_model, make_sampler  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n_targets_compute", type=int, default=128)
    ap.add_argument("--out", default="robustness_checks.json")
    ap.add_argument("--checks", default="ABCDE",
                    help="which checks to run; existing out-file keys are kept")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ev = cfg["eval"]
    arm = PCCArm(**cfg["arm"])
    tol = ev["success_tol"]
    results = {}
    if os.path.exists(args.out):
        results = json.load(open(args.out))

    if set("AB") & set(args.checks):
        targets = sample_reachable_targets(arm, args.n_targets_compute)

    # ---------- A. grad-IK compute ablation ---------- #
    if "A" in args.checks:
      print("=== A: grad-IK compute ablation ===", flush=True)
      results["gradik_compute"] = {}
      for iters in (100, 300):
        for restarts in (8, 32):
            torch.manual_seed(0)
            t0 = time.time()
            best = []
            for t in targets:
                qs = grad_ik(arm, t, n_restarts=restarts, iters=iters)
                best.append(tip_error(arm, qs, t.unsqueeze(0).expand(restarts, -1)).min().item())
            dt = (time.time() - t0) / len(targets) * 1000
            best = np.array(best)
            key = f"iters{iters}_restarts{restarts}"
            results["gradik_compute"][key] = {
                "tip_err_best_mm": float(best.mean() * 1000),
                "success": float((best < tol).mean()),
                "ms_per_target": dt,
            }
            print(key, results["gradik_compute"][key], flush=True)

    # ---------- B. grad-IK restart-RNG variance ---------- #
    if "B" in args.checks:
      print("=== B: grad-IK RNG variance (300 iters, 32 restarts) ===", flush=True)
      per_seed = []
      for s in (0, 1, 2):
        torch.manual_seed(s)
        best = []
        for t in targets:
            qs = grad_ik(arm, t, n_restarts=32, iters=300)
            best.append(tip_error(arm, qs, t.unsqueeze(0).expand(32, -1)).min().item())
        per_seed.append(float(np.mean(best) * 1000))
        print(f"seed {s}: {per_seed[-1]:.3f}mm", flush=True)
      results["gradik_rng_variance"] = {
        "tip_err_best_mm_mean": float(np.mean(per_seed)),
        "tip_err_best_mm_std": float(np.std(per_seed)),
        "values": per_seed,
      }

    # ---------- C. mode-recall hyperparameter sensitivity ---------- #
    if "C" in args.checks:
      print("=== C: E2 hyperparameter sensitivity ===", flush=True)
      device = "cpu"
      model, norm, _ = load_model("checkpoints_seed0/diffusion.pt", device)
      sampler = make_sampler(model, norm, device, ev["ddim_steps"], ev["guidance"])
      mode_targets = sample_reachable_targets(arm, ev["n_targets_modes"], seed=777)
      K = ev["n_samples_per_target"]

      # candidates (fixed across the grid): diffusion samples + grad-IK restarts
      cands = {}
      for ti, t in enumerate(mode_targets):
        qd = sampler(t.unsqueeze(0), K)[0]
        ed = tip_error(arm, qd, t.unsqueeze(0).expand(K, -1))
        torch.manual_seed(0)
        qg = grad_ik(arm, t, n_restarts=32, iters=300)
        eg = tip_error(arm, qg, t.unsqueeze(0).expand(32, -1))
        cands[ti] = (curvature_features(arm, qd), ed < tol,
                     curvature_features(arm, qg), eg < tol)

      results["e2_sensitivity"] = {}
      for eps in (10.0, 12.0, 14.0):
        gts = [_enumerate(arm, t, pool_size=ev["mode_pool"], tol=tol,
                          dbscan_eps=eps, min_samples=ev.get("dbscan_min_samples", 10))
               for t in mode_targets]
        for radius in (10.0, 15.0, 20.0):
            rec_d, rec_g = [], []
            for ti in range(len(mode_targets)):
                if gts[ti].shape[0] == 0:
                    continue
                fd, okd, fg, okg = cands[ti]
                rec_d.append(mode_recall(gts[ti], fd, okd, radius=radius))
                rec_g.append(mode_recall(gts[ti], fg, okg, radius=radius))
            key = f"eps{eps:g}_radius{radius:g}"
            results["e2_sensitivity"][key] = {
                "diffusion_recall": float(np.nanmean(rec_d)),
                "gradik_recall": float(np.nanmean(rec_g)),
                "mean_gt_modes": float(np.mean([g.shape[0] for g in gts])),
            }
            print(key, results["e2_sensitivity"][key], flush=True)

    # ---------- D. Elastica settle-time convergence ---------- #
    if "D" in args.checks:
      print("=== D: settle-time convergence ===", flush=True)
      rng = np.random.default_rng(3)
      q = torch.as_tensor(rng.uniform(0, 1, size=(8, arm.q_dim)), dtype=torch.float32)
      tips = {}
      for settle in (1.5, 3.0):
        ela = ElasticaArm(n_segments=cfg["arm"]["n_segments"],
                          seg_length=cfg["arm"]["seg_length"],
                          base_radius=cfg["arm"]["rod_radius"], settle_time=settle)
        tips[settle] = ela.forward(q)["tip"]
      d = (tips[1.5] - tips[3.0]).norm(dim=-1) * 1000
      results["settle_convergence"] = {
        "max_tip_shift_mm_1p5_vs_3p0": float(d.max()),
        "mean_tip_shift_mm": float(d.mean()),
        "n_actuations": 8,
      }
      print(results["settle_convergence"], flush=True)

    # ---------- E. torque_gain calibration sweep ---------- #
    if "E" in args.checks:
      print("=== E: torque_gain calibration sweep ===", flush=True)
      rng = np.random.default_rng(7)
      q = torch.as_tensor(rng.uniform(0, 1, size=(8, arm.q_dim)), dtype=torch.float32)
      pcc_tips = arm.forward(q)["tip"]
      results["gain_calibration_sweep"] = {
          "note": "8 shared random actuations; per gain, mean/max tip discrepancy "
                  "between ElasticaArm(torque_gain=g) and the PCC arm, plus each "
                  "simulator's mean tip radius (workspace scale)",
          "pcc_mean_tip_radius_m": float(pcc_tips.norm(dim=-1).mean()),
          "gains": {},
      }
      for gain in (5.0e-4, 1.5e-3, 5.0e-3, 1.5e-2, 5.0e-2):
        ela = ElasticaArm(n_segments=cfg["arm"]["n_segments"],
                          seg_length=cfg["arm"]["seg_length"],
                          base_radius=cfg["arm"]["rod_radius"], torque_gain=gain)
        et = ela.forward(q)["tip"]
        d = (et - pcc_tips).norm(dim=-1) * 1000
        results["gain_calibration_sweep"]["gains"][f"{gain:g}"] = {
            "mean_tip_discrepancy_mm": float(d.mean()),
            "max_tip_discrepancy_mm": float(d.max()),
            "min_tip_discrepancy_mm": float(d.min()),
            "elastica_mean_tip_radius_m": float(et.norm(dim=-1).mean()),
        }
        print(gain, results["gain_calibration_sweep"]["gains"][f"{gain:g}"], flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
