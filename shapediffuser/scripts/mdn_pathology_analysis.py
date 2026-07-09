#!/usr/bin/env python3
"""Demonstrate (not assert) why the MDN underperforms: per-component analysis.

The external critique: "MDNs are notoriously unstable to train (collapsing
variances, NLL blow-ups); before trusting this result, I'd want per-component
error histograms or training curves ruling out a training pathology."

This checks, per component count k in {4,8,16,32} on the swept checkpoints:
  1. mixture-weight health   - effective components (perplexity of pi)
  2. variance health         - sigma statistics (collapsing variances show up
                               as sigma -> exp(-6) clamp floor)
  3. per-component accuracy  - tip error of each component MEAN, per target:
                               if even the best component mean is far from
                               tolerance, the failure is imprecision, not
                               collapse or sampling noise
  4. training-curve health   - parses the sweep log for NLL blow-ups
                               (pass --train_log)

    python scripts/mdn_pathology_analysis.py --train_log <sweep log path>
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shapediffuser import PCCArm, BabblingDataset  # noqa: E402
from shapediffuser.metrics import sample_reachable_targets, tip_error  # noqa: E402
from evaluate import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="4,8,16,32")
    ap.add_argument("--n_targets", type=int, default=128)
    ap.add_argument("--train_log", default=None,
                    help="sweep stdout containing '[mdn] epoch ...' lines")
    ap.add_argument("--out", default="mdn_pathology.json")
    args = ap.parse_args()

    arm = PCCArm(n_segments=4, seg_length=0.06, curvature_gain=25.0,
                 points_per_seg=10, rod_radius=0.012)
    targets = sample_reachable_targets(arm, args.n_targets, seed=321)
    tol = 0.005

    results = {}
    for k in (int(x) for x in args.ks.split(",")):
        model, norm, ck = load_model(f"checkpoints_mdn_k{k}/mdn.pt", "cpu")
        cond = norm.encode(targets)
        with torch.no_grad():
            log_pi, mu, log_sigma = model._params(cond)  # (N,k), (N,k,12), (N,k,12)
        pi = log_pi.exp()
        eff = torch.exp(-(pi * log_pi).sum(-1))
        sigma = log_sigma.exp()

        # tip error of every component MEAN for every target
        press = BabblingDataset.q_to_pressures(mu.reshape(-1, 12))
        errs = tip_error(arm, press, targets.repeat_interleave(k, dim=0)).view(-1, k)
        best_comp = errs.min(dim=1).values  # (N,) best component mean per target

        results[k] = {
            "effective_components_mean": float(eff.mean()),
            "sigma_mean": float(sigma.mean()),
            "sigma_min": float(sigma.min()),
            "sigma_frac_at_clamp_floor": float((log_sigma <= -5.99).float().mean()),
            "component_mean_tip_err_median_mm": float(errs.median() * 1000),
            "best_component_tip_err_mean_mm": float(best_comp.mean() * 1000),
            "best_component_success_rate": float((best_comp < tol).float().mean()),
            "frac_components_within_tol": float((errs < tol).float().mean()),
        }
        r = results[k]
        print(f"k={k}: eff_comp={r['effective_components_mean']:.2f}  "
              f"sigma(mean/min)={r['sigma_mean']:.3f}/{r['sigma_min']:.4f}  "
              f"clamp_frac={r['sigma_frac_at_clamp_floor']:.3f}  "
              f"median comp err={r['component_mean_tip_err_median_mm']:.1f}mm  "
              f"BEST comp err={r['best_component_tip_err_mean_mm']:.1f}mm  "
              f"best-comp success={r['best_component_success_rate']:.3f}", flush=True)

    if args.train_log and os.path.exists(args.train_log):
        curves, cur = {}, None
        for line in open(args.train_log, errors="ignore"):
            m = re.search(r"=== mdn_k=(\d+) ===", line)
            if m:
                cur = int(m.group(1))
                curves[cur] = {"train": [], "val": []}
            m = re.search(r"\[mdn\] epoch \d+/\d+ train ([\d.]+) val ([\d.]+)", line)
            if m and cur is not None:
                curves[cur]["train"].append(float(m.group(1)))
                curves[cur]["val"].append(float(m.group(2)))
        for k, c in curves.items():
            tr = np.array(c["train"])
            monotone_frac = float((np.diff(tr) <= 0.05).mean())  # allow tiny bumps
            results[k]["train_curve"] = {
                "n_epochs": len(tr), "first": float(tr[0]), "last": float(tr[-1]),
                "max_epoch_increase": float(np.diff(tr).max()) if len(tr) > 1 else 0.0,
                "frac_nonincreasing_steps": monotone_frac,
            }
            print(f"k={k} training: {tr[0]:.2f} -> {tr[-1]:.2f} over {len(tr)} epochs, "
                  f"max single-epoch increase {results[k]['train_curve']['max_epoch_increase']:.3f}",
                  flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
