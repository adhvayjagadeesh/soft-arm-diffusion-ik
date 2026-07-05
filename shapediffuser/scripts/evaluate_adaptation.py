#!/usr/bin/env python3
"""Does transfer-guided sampling generalize to held-out targets?

Compares plain diffusion sampling against sample_with_transfer_guidance
(steered by the TransferRegressor from train_transfer_regressor.py) on
targets the regressor never saw during its own training - this held-out
check is the whole point: if guidance only helps on the targets it was
fit to, that's memorization, not adaptation.

    python scripts/evaluate_adaptation.py --n_targets 15 --k 16 --guidance_scale 0.3
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
from shapediffuser.metrics import sample_reachable_targets  # noqa: E402
from evaluate import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--regressor_ckpt", default="checkpoints/transfer_regressor.pt")
    ap.add_argument("--n_targets", type=int, default=15)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=9999,
                    help="distinct from train_transfer_regressor.py's seed=8888 - held-out targets")
    ap.add_argument("--guidance_scale", type=float, default=0.3)
    ap.add_argument("--out", default="adaptation_results.json")
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

    model, norm, ck = load_model(os.path.join(args.ckpt_dir, "diffusion.pt"), device)

    reg_ck = torch.load(args.regressor_ckpt, map_location=device, weights_only=False)
    regressor = TransferRegressor(q_dim=reg_ck["q_dim"], cond_dim=reg_ck["cond_dim"]).to(device)
    regressor.load_state_dict(reg_ck["model"])
    regressor.eval()

    targets = sample_reachable_targets(pcc, args.n_targets, seed=args.seed)

    baseline_errs, guided_errs = [], []
    t_start = time.time()
    for ti, t in enumerate(targets):
        cond = norm.encode(t.unsqueeze(0).to(device))

        with torch.no_grad():
            q_base = model.sample(cond, n_samples=args.k,
                                  steps=ev["ddim_steps"], guidance=ev["guidance"])[0]
        q_guided = model.sample_with_transfer_guidance(
            cond, regressor, transfer_guidance_scale=args.guidance_scale,
            n_samples=args.k, steps=ev["ddim_steps"], guidance=ev["guidance"])[0]

        p_base = BabblingDataset.q_to_pressures(q_base).cpu()
        p_guided = BabblingDataset.q_to_pressures(q_guided).cpu()

        err_base = (elastica.forward(p_base)["tip"] - t).norm(dim=-1).numpy() * 1000
        err_guided = (elastica.forward(p_guided)["tip"] - t).norm(dim=-1).numpy() * 1000
        baseline_errs.append(err_base)
        guided_errs.append(err_guided)
        print(f"target {ti + 1}/{len(targets)}: baseline best {err_base.min():.1f}mm  "
              f"guided best {err_guided.min():.1f}mm", flush=True)

    baseline_errs = np.stack(baseline_errs)  # (n, k) mm
    guided_errs = np.stack(guided_errs)
    n = baseline_errs.shape[0]

    base_best = baseline_errs.min(axis=1)
    guided_best = guided_errs.min(axis=1)
    delta = base_best - guided_best  # positive = guidance better

    results = {
        "n_targets": args.n_targets, "k": args.k, "guidance_scale": args.guidance_scale,
        "baseline_best_of_k_mean_mm": float(base_best.mean()),
        "guided_best_of_k_mean_mm": float(guided_best.mean()),
        "baseline_success_rate": float((base_best < tol * 1000).mean()),
        "guided_success_rate": float((guided_best < tol * 1000).mean()),
        "paired_delta_mean_mm": float(delta.mean()),
        "paired_delta_sem_mm": float(delta.std(ddof=1) / np.sqrt(n)),
        "paired_delta_t": float(delta.mean() / (delta.std(ddof=1) / np.sqrt(n) + 1e-12)),
        "n_targets_guided_better": int((delta > 0).sum()),
        "n_targets_baseline_better": int((delta < 0).sum()),
        "wall_clock_s": time.time() - t_start,
    }
    print(f"\n{results}")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
