#!/usr/bin/env python3
"""Does per-target improvement from candidate selection correlate with
anything measurable on the PCC side alone?

query_budget_study.py found that best-of-32 vs best-of-1 improvement under
Elastica ranges from ~0mm to ~74mm depending on the target, with no
explanation offered. This tests two purely-PCC-side predictors (no new
Elastica simulation needed - Elastica errors are already cached in
query_budget_results.json):

  diversity  - spread of the 32 sampled candidates in curvature-feature
               space (metrics.diversity/curvature_features): how many
               genuinely different PCC-valid solutions did the model
               actually propose for this target?
  min_sv     - smallest singular value of the PCC tip-position Jacobian at
               the target's best-PCC candidate (PCC is differentiable, so
               this is nearly free via autograd): how locally sensitive is
               PCC's own map at this point? (Note: PCC's Jacobian is a
               (3,12) matrix, generically rank 3, so there's always a
               large exact null space - min_sv here characterizes the
               *tightest* locally-sensitive direction, not the null space
               itself.)

Candidates here are freshly resampled from the same 20 targets (same seed
as query_budget_study.py), not the literal candidates from that run, since
diffusion sampling isn't seeded - diversity/Jacobian structure are treated
as properties of the (fixed, trained) model at a given target, not of one
specific stochastic draw, so this is a valid independent measurement.

    python scripts/query_budget_correlation.py
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shapediffuser import PCCArm  # noqa: E402
from shapediffuser.metrics import (  # noqa: E402
    sample_reachable_targets, curvature_features, diversity,
)
from evaluate import load_model, make_sampler  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--query_budget_results", default="query_budget_results.json")
    ap.add_argument("--out", default="query_budget_correlation.json")
    ap.add_argument("--fig", default="figures/query_budget_correlation.png")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ev = cfg["eval"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pcc = PCCArm(**cfg["arm"])

    qb = json.load(open(args.query_budget_results))
    err = np.array(qb["per_target_elastica_err_mm"])  # (n_targets, k), PCC-ordered
    improvement = err[:, 0] - err.min(axis=1)  # (n_targets,) mm
    n_targets = err.shape[0]

    model, norm, ck = load_model(os.path.join(args.ckpt_dir, "diffusion.pt"), device)
    sampler = make_sampler(model, norm, device, ev["ddim_steps"], ev["guidance"])
    targets = sample_reachable_targets(pcc, n_targets, seed=args.seed)

    def tip_fn(q_flat):
        return pcc.forward(q_flat.unsqueeze(0))["tip"].squeeze(0)

    div_vals, min_sv_vals = [], []
    for ti, t in enumerate(targets):
        qs = sampler(t.unsqueeze(0), args.k)[0]  # (k, q_dim)
        feats = curvature_features(pcc, qs)
        div_vals.append(diversity(feats))

        with torch.no_grad():
            pcc_tip = pcc.forward(qs)["tip"]
        best_idx = (pcc_tip - t).norm(dim=-1).argmin().item()
        q_best = qs[best_idx].clone().requires_grad_(True)
        J = torch.autograd.functional.jacobian(tip_fn, q_best)
        sv = torch.linalg.svdvals(J)
        min_sv_vals.append(sv.min().item())
        print(f"target {ti + 1}/{n_targets}: diversity={div_vals[-1]:.3f} "
              f"min_sv={min_sv_vals[-1]:.4f} improvement={improvement[ti]:.1f}mm")

    div_vals, min_sv_vals = np.array(div_vals), np.array(min_sv_vals)

    results = {"diversity": div_vals.tolist(), "min_singular_value": min_sv_vals.tolist(),
              "improvement_mm": improvement.tolist()}
    for name, x in [("diversity", div_vals), ("min_singular_value", min_sv_vals)]:
        r_pearson, p_pearson = stats.pearsonr(x, improvement)
        r_spearman, p_spearman = stats.spearmanr(x, improvement)
        results[f"{name}_vs_improvement"] = {
            "pearson_r": float(r_pearson), "pearson_p": float(p_pearson),
            "spearman_r": float(r_spearman), "spearman_p": float(p_spearman),
        }
        print(f"\n{name} vs improvement: pearson r={r_pearson:.3f} (p={p_pearson:.3f}), "
              f"spearman r={r_spearman:.3f} (p={p_spearman:.3f})")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, name, x in [(axes[0], "diversity (curvature-space spread)", div_vals),
                        (axes[1], "min singular value (PCC Jacobian)", min_sv_vals)]:
        ax.scatter(x, improvement, color="tab:blue")
        r = results[f"{name.split(' ')[0]}_vs_improvement"]["pearson_r"] \
            if "diversity" in name else results["min_singular_value_vs_improvement"]["pearson_r"]
        ax.set_xlabel(name)
        ax.set_ylabel("improvement, m=1->m=32 (mm)")
        ax.set_title(f"pearson r = {r:.2f}")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.fig) or ".", exist_ok=True)
    fig.savefig(args.fig, dpi=200)
    print(f"Wrote {args.fig}")


if __name__ == "__main__":
    main()
