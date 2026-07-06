#!/usr/bin/env python3
"""Does diversity-ordering combine with transfer-guided sampling?

Two independent mechanisms were each confirmed to help (see README): guided
sampling changes what candidates get generated (t=2.27 on held-out targets),
diversity-ordering changes which of a fixed candidate set you spend your
query budget on first (t=3.34/3.21 at moderate budgets). This tests whether
they stack: does diversity-ordering *guided* candidates beat diversity-
ordering *unguided* candidates by roughly the same margin the guided-sampling
result alone would predict, more, or not at all?

Full 2x2 factorial from ONE paired Elastica evaluation pass per target (not
four separate reruns): for each of --n_targets held-out targets, samples K
unguided AND K guided candidates, evaluates all 2K under Elastica once, then
computes both pcc-order and diversity-order best-of-m curves for both
candidate sets from the same cached per-candidate errors.

    python scripts/combined_adaptation_ordering.py --n_targets 15 --k 16
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

from shapediffuser import PCCArm, BabblingDataset, TransferRegressor  # noqa: E402
from shapediffuser.elastica_arm import ElasticaArm  # noqa: E402
from shapediffuser.metrics import sample_reachable_targets, curvature_features  # noqa: E402
from evaluate import load_model  # noqa: E402
from query_budget_study import diversity_order  # noqa: E402


def orderings_for(qs, pcc, t):
    """qs: (k, q_dim) pressures. Returns dict of {name: index order (k,)}."""
    with torch.no_grad():
        pcc_tip = pcc.forward(qs)["tip"]
    pcc_err = (pcc_tip - t).norm(dim=-1)
    pcc_ord = torch.argsort(pcc_err)
    feats = curvature_features(pcc, qs)
    div_ord = torch.as_tensor(diversity_order(feats, int(pcc_ord[0].item())))
    return {"pcc": pcc_ord.numpy(), "diversity": div_ord.numpy()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--regressor_ckpt", default="checkpoints/transfer_regressor.pt")
    ap.add_argument("--n_targets", type=int, default=15)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=9999, help="same held-out set as evaluate_adaptation.py")
    ap.add_argument("--guidance_scale", type=float, default=0.3)
    ap.add_argument("--budgets", default="1,2,4,8,16")
    ap.add_argument("--out", default="combined_adaptation_ordering_results.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ev = cfg["eval"]
    tol = ev["success_tol"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pcc = PCCArm(**cfg["arm"])
    elastica = ElasticaArm(
        n_segments=cfg["arm"]["n_segments"],
        seg_length=cfg["arm"]["seg_length"],
        base_radius=cfg["arm"]["rod_radius"],
    )
    budgets = [int(b) for b in args.budgets.split(",") if int(b) <= args.k]

    model, norm, ck = load_model(os.path.join(args.ckpt_dir, "diffusion.pt"), device)
    reg_ck = torch.load(args.regressor_ckpt, map_location=device, weights_only=False)
    regressor = TransferRegressor(q_dim=reg_ck["q_dim"], cond_dim=reg_ck["cond_dim"]).to(device)
    regressor.load_state_dict(reg_ck["model"])
    regressor.eval()

    targets = sample_reachable_targets(pcc, args.n_targets, seed=args.seed)

    # cached[cond_name][ordering_name] -> (n_targets, k) mm, reordered per-target
    cached = {"unguided": {"pcc": [], "diversity": []}, "guided": {"pcc": [], "diversity": []}}
    t_start = time.time()
    for ti, t in enumerate(targets):
        cond = norm.encode(t.unsqueeze(0).to(device))

        with torch.no_grad():
            q_base = model.sample(cond, n_samples=args.k, steps=ev["ddim_steps"], guidance=ev["guidance"])[0]
        q_guided = model.sample_with_transfer_guidance(
            cond, regressor, transfer_guidance_scale=args.guidance_scale,
            n_samples=args.k, steps=ev["ddim_steps"], guidance=ev["guidance"])[0]

        p_base = BabblingDataset.q_to_pressures(q_base).cpu()
        p_guided = BabblingDataset.q_to_pressures(q_guided).cpu()

        err_base = (elastica.forward(p_base)["tip"] - t).norm(dim=-1).numpy() * 1000
        err_guided = (elastica.forward(p_guided)["tip"] - t).norm(dim=-1).numpy() * 1000

        ord_base = orderings_for(p_base, pcc, t)
        ord_guided = orderings_for(p_guided, pcc, t)
        for name, ordr in ord_base.items():
            cached["unguided"][name].append(err_base[ordr])
        for name, ordr in ord_guided.items():
            cached["guided"][name].append(err_guided[ordr])

        print(f"target {ti + 1}/{len(targets)}: unguided-best {err_base.min():.1f}mm  "
              f"guided-best {err_guided.min():.1f}mm", flush=True)

    results = {"n_targets": args.n_targets, "k": args.k, "guidance_scale": args.guidance_scale,
              "budgets": {}}
    for cond_name in ("unguided", "guided"):
        for ord_name in ("pcc", "diversity"):
            arr = np.stack(cached[cond_name][ord_name])  # (n, k)
            key = f"{cond_name}_{ord_name}"
            for m in budgets:
                best = arr[:, :m].min(axis=1)
                results["budgets"].setdefault(m, {})[key] = {
                    "mean_mm": float(best.mean()),
                    "sem_mm": float(best.std(ddof=1) / np.sqrt(len(best))),
                    "success_rate": float((best < tol * 1000).mean()),
                }

    results["wall_clock_s"] = time.time() - t_start
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}\n")

    print(f"{'m':>4} {'unguided+pcc':>14} {'unguided+div':>14} {'guided+pcc':>12} {'guided+div':>12}")
    for m in budgets:
        b = results["budgets"][m]
        print(f"{m:>4} {b['unguided_pcc']['mean_mm']:>14.1f} {b['unguided_diversity']['mean_mm']:>14.1f} "
              f"{b['guided_pcc']['mean_mm']:>12.1f} {b['guided_diversity']['mean_mm']:>12.1f}")


if __name__ == "__main__":
    main()
