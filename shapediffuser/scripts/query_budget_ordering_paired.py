#!/usr/bin/env python3
"""Properly paired comparison of pcc-order vs diversity-order candidate selection.

query_budget_study.py's --order pcc and --order diversity runs are two
*independent* draws of K candidates per target, so the ~2-5.6mm advantage
diversity-order showed at intermediate budgets could just be between-run
sampling noise (each run's own SEM is ~9-10mm). This settles it properly: draw
K=32 candidates ONCE per target, evaluate all 32 under Elastica ONCE (so this
costs *less* total Elastica compute than running both orderings separately),
then apply both orderings post-hoc to the exact same cached per-candidate
error array - a paired comparison with no extra between-run noise.

    python scripts/query_budget_ordering_paired.py --n_targets 45 --k 32
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
from shapediffuser.elastica_arm import ElasticaArm  # noqa: E402
from shapediffuser.metrics import sample_reachable_targets, curvature_features  # noqa: E402
from evaluate import load_model, make_sampler  # noqa: E402
from query_budget_study import diversity_order  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--n_targets", type=int, default=45)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--budgets", default="1,2,4,8,16,32")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", default="query_budget_ordering_paired.json")
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

    pcc_err_by_target, div_err_by_target = [], []
    t_start = time.time()
    for ti, t in enumerate(targets):
        qs = sampler(t.unsqueeze(0), args.k)[0]  # (k, q_dim) - ONE draw per target

        with torch.no_grad():
            pcc_tip = pcc.forward(qs)["tip"]
        pcc_error = (pcc_tip - t).norm(dim=-1)
        pcc_ord = torch.argsort(pcc_error)

        feats = curvature_features(pcc, qs)
        div_ord = torch.as_tensor(diversity_order(feats, int(pcc_ord[0].item())))

        # ONE Elastica pass over all k candidates, in original (arbitrary) index
        # order; both orderings below just reindex this same cached array.
        elastica_tip = elastica.forward(qs)["tip"]
        elastica_err = (elastica_tip - t).norm(dim=-1).numpy()

        pcc_err_by_target.append(elastica_err[pcc_ord.numpy()])
        div_err_by_target.append(elastica_err[div_ord.numpy()])
        print(f"target {ti + 1}/{len(targets)}: pcc-best {pcc_error.min().item()*1000:.2f}mm "
              f"(single Elastica pass, both orderings reindex the same {args.k} errors)",
              flush=True)

    pcc_err_by_target = np.stack(pcc_err_by_target) * 1000  # (n, k) mm
    div_err_by_target = np.stack(div_err_by_target) * 1000
    n = pcc_err_by_target.shape[0]

    results = {"n_targets": args.n_targets, "k": args.k, "budgets": {}}
    for m in budgets:
        pcc_best = pcc_err_by_target[:, :m].min(axis=1)
        div_best = div_err_by_target[:, :m].min(axis=1)
        delta = pcc_best - div_best  # positive = diversity-order better
        results["budgets"][m] = {
            "pcc_order_mm": float(pcc_best.mean()),
            "diversity_order_mm": float(div_best.mean()),
            "paired_delta_mean_mm": float(delta.mean()),
            "paired_delta_sem_mm": float(delta.std(ddof=1) / np.sqrt(n)),
            "paired_delta_t": float(delta.mean() / (delta.std(ddof=1) / np.sqrt(n) + 1e-12)),
            "n_targets_diversity_better": int((delta > 0).sum()),
            "n_targets_pcc_better": int((delta < 0).sum()),
        }
        print(f"budget m={m}: {results['budgets'][m]}")

    results["wall_clock_s"] = time.time() - t_start
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
