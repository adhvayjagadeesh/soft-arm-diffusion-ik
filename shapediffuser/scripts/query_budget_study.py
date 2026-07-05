#!/usr/bin/env python3
"""Query-efficient candidate selection under PCC->Elastica model mismatch.

Core pivot experiment: a deterministic IK model only ever proposes one
actuation candidate, so if it's wrong under simulator/model mismatch there's
nothing to select among. A diffusion model proposes K diverse candidates in
the (cheap, idealized) PCC simulator it was trained on. This measures how
much of the sim-to-sim transfer gap (see transfer_study.py: diffusion
collapses from 0.44mm/100% on PCC to 140.6mm/0% on Elastica) can be recovered
by spending a small budget of m <= K expensive Elastica evaluations to
*select* among those candidates, versus having no candidates to select from
at all.

Two candidate orderings are supported (--order):
  pcc       ascending PCC-predicted error (the realistic "try your most
            simulator-confident solution first" strategy) - the original,
            default ordering.
  diversity greedy farthest-point selection in curvature-feature space,
            seeded from the same PCC-best candidate (so budget=1 is identical
            between orderings, isolating the effect of *ordering* the
            remaining budget rather than changing the first guess): does
            spending your query budget on maximally-different candidates
            reach the same accuracy with a smaller m than PCC-confidence
            order?

All K Elastica evaluations are cached once per target (in --order's chosen
sequence) so every budget size is computed post-hoc from the same cached
array, no re-simulation needed.

    python scripts/query_budget_study.py --n_targets 45 --k 32 --order pcc
    python scripts/query_budget_study.py --n_targets 45 --k 32 --order diversity
"""

import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shapediffuser import PCCArm  # noqa: E402
from shapediffuser.elastica_arm import ElasticaArm  # noqa: E402
from shapediffuser.metrics import sample_reachable_targets, curvature_features  # noqa: E402
from evaluate import load_model, make_sampler  # noqa: E402


def diversity_order(feats: torch.Tensor, seed_idx: int) -> list:
    """Greedy farthest-point ordering in feature space, starting from seed_idx."""
    n = feats.shape[0]
    remaining = set(range(n))
    order = [seed_idx]
    remaining.discard(seed_idx)
    min_dist = torch.cdist(feats[seed_idx:seed_idx + 1], feats).squeeze(0)
    while remaining:
        rem_list = list(remaining)
        rem_dists = min_dist[rem_list]
        next_idx = rem_list[int(torch.argmax(rem_dists).item())]
        order.append(next_idx)
        remaining.discard(next_idx)
        d_new = torch.cdist(feats[next_idx:next_idx + 1], feats).squeeze(0)
        min_dist = torch.minimum(min_dist, d_new)
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--n_targets", type=int, default=20)
    ap.add_argument("--k", type=int, default=32, help="candidates sampled per target")
    ap.add_argument("--budgets", default="1,2,4,8,16,32",
                    help="comma-separated Elastica-query budgets m <= k to evaluate")
    ap.add_argument("--seed", type=int, default=4242,
                    help="matches transfer_study.py's default for comparable targets")
    ap.add_argument("--order", default="pcc", choices=["pcc", "diversity"])
    ap.add_argument("--out", default="query_budget_results.json")
    ap.add_argument("--fig", default="figures/query_budget_curve.png")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ev = cfg["eval"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pcc = PCCArm(**cfg["arm"])
    elastica = ElasticaArm(
        n_segments=cfg["arm"]["n_segments"],
        seg_length=cfg["arm"]["seg_length"],
        base_radius=cfg["arm"]["rod_radius"],
    )
    tol = ev["success_tol"]
    budgets = [int(b) for b in args.budgets.split(",") if int(b) <= args.k]

    model, norm, ck = load_model(os.path.join(args.ckpt_dir, "diffusion.pt"), device)
    sampler = make_sampler(model, norm, device, ev["ddim_steps"], ev["guidance"])

    targets = sample_reachable_targets(pcc, args.n_targets, seed=args.seed)

    # per_target_elastica_err[i] = (k,) Elastica tip errors for target i's k
    # candidates, already sorted by ascending PCC-predicted error.
    per_target_elastica_err = []
    t_start = time.time()
    for ti, t in enumerate(targets):
        qs = sampler(t.unsqueeze(0), args.k)[0]  # (k, q_dim)

        with torch.no_grad():
            pcc_tip = pcc.forward(qs)["tip"]
        pcc_err = (pcc_tip - t).norm(dim=-1)
        pcc_order = torch.argsort(pcc_err)

        if args.order == "pcc":
            order = pcc_order
        else:
            feats = curvature_features(pcc, qs)
            order = torch.as_tensor(diversity_order(feats, int(pcc_order[0].item())))
        qs_ordered = qs[order]

        elastica_tip = elastica.forward(qs_ordered)["tip"]
        elastica_err = (elastica_tip - t).norm(dim=-1).numpy()
        per_target_elastica_err.append(elastica_err)
        print(f"target {ti + 1}/{len(targets)}: "
              f"pcc-best {pcc_err.min().item() * 1000:.2f}mm  "
              f"elastica errs ({args.order}-order) [mm] = "
              f"{np.array2string(elastica_err * 1000, precision=1)}", flush=True)

    per_target_elastica_err = np.stack(per_target_elastica_err)  # (n_targets, k)
    n = per_target_elastica_err.shape[0]

    results = {"n_targets": args.n_targets, "k": args.k, "order": args.order, "budgets": {},
              "per_target_elastica_err_mm": (per_target_elastica_err * 1000).tolist()}
    for m in budgets:
        best_of_m = per_target_elastica_err[:, :m].min(axis=1)  # (n_targets,)
        mm = best_of_m * 1000
        results["budgets"][m] = {
            "elastica_tip_err_best_of_m_mm": float(mm.mean()),
            "elastica_tip_err_sem_mm": float(mm.std(ddof=1) / np.sqrt(n)),
            "elastica_success_rate_best_of_m": float((best_of_m < tol).mean()),
        }
        print(f"budget m={m}: {results['budgets'][m]}")

    results["wall_clock_s"] = time.time() - t_start
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")

    ms = sorted(results["budgets"].keys())
    mean_err = [results["budgets"][m]["elastica_tip_err_best_of_m_mm"] for m in ms]
    sem_err = [results["budgets"][m]["elastica_tip_err_sem_mm"] for m in ms]
    success = [results["budgets"][m]["elastica_success_rate_best_of_m"] for m in ms]

    mlp_results = {}
    mlp_path = "transfer_study_results.json"
    if os.path.exists(mlp_path):
        mlp_results = json.load(open(mlp_path)).get("mlp", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.errorbar(ms, mean_err, yerr=sem_err, marker="o", color="tab:blue",
                capsize=4, label=f"diffusion (best-of-m, {args.order}-order)")
    if "elastica_tip_err_best_of_K_mm" in mlp_results:
        ax1.axhline(mlp_results["elastica_tip_err_best_of_K_mm"], color="tab:red",
                   linestyle="--", label="mlp (1 candidate only)")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(ms)
    ax1.set_xticklabels([str(m) for m in ms])
    ax1.set_xlabel("Elastica query budget m")
    ax1.set_ylabel("mean best-of-m tip error under Elastica (mm)")
    ax1.set_title("Headline: continuous error, mean +/- SEM")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(ms, success, marker="o", color="tab:blue", label="diffusion")
    if "elastica_success_rate_best_of_K" in mlp_results:
        ax2.axhline(mlp_results["elastica_success_rate_best_of_K"], color="tab:red",
                   linestyle="--", label="mlp")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(ms)
    ax2.set_xticklabels([str(m) for m in ms])
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("Elastica query budget m")
    ax2.set_ylabel("success rate (tol=5mm)")
    ax2.set_title("Secondary: binary success (tol likely too strict\nfor this scale of mismatch)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    os.makedirs(os.path.dirname(args.fig) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.fig, dpi=200)
    print(f"Wrote {args.fig}")


if __name__ == "__main__":
    main()
