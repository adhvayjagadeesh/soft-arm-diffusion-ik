#!/usr/bin/env python3
"""PCC -> Elastica sim-to-sim transfer study.

Takes actuations sampled from a model trained purely on the fast PCC
simulator's inverse kinematics and executes them through the high-fidelity
PyElastica Cosserat-rod simulator instead, measuring tip error there. This
probes robustness to model mismatch between the simulator the model was
trained on and a more physically realistic one - reviewers like this because
the paper's claims are about learning the inverse map of a *given* simulator,
and this shows how much of that advantage survives when the map is wrong.

ElasticaArm has no batching/parallelism (~4s per sample, unbatched Cosserat
integration), so n_targets/K are kept small relative to the main evaluate.py
suite: this is deliberately a smaller high-fidelity spot-check, not a
full-scale evaluation.

    pip install pyelastica
    python scripts/transfer_study.py --n_targets 20 --k 8
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
from shapediffuser.metrics import sample_reachable_targets  # noqa: E402
from evaluate import load_model, make_sampler  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--n_targets", type=int, default=20)
    ap.add_argument("--k", type=int, default=8,
                    help="candidate actuations per target (each costs one Elastica sim)")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", default="transfer_study_results.json")
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

    samplers = {}
    for name in ("diffusion", "mlp"):
        path = os.path.join(args.ckpt_dir, f"{name}.pt")
        if os.path.exists(path):
            model, norm, ck = load_model(path, device)
            samplers[name] = make_sampler(model, norm, device, ev["ddim_steps"], ev["guidance"])
        else:
            print(f"(no checkpoint for {name}, skipping)")

    targets = sample_reachable_targets(pcc, args.n_targets, seed=args.seed)

    results = {}
    for name, sampler in samplers.items():
        pcc_errs, elastica_errs = [], []
        t_start = time.time()
        for ti, t in enumerate(targets):
            qs = sampler(t.unsqueeze(0), args.k)[0]  # (k, q_dim) pressures in [0,1]

            with torch.no_grad():
                pcc_tip = pcc.forward(qs)["tip"]
            pcc_err = (pcc_tip - t).norm(dim=-1)
            pcc_errs.append(pcc_err.min().item())

            elastica_tip = elastica.forward(qs)["tip"]
            elastica_err = (elastica_tip - t).norm(dim=-1)
            elastica_errs.append(elastica_err.min().item())
            print(f"  [{name}] target {ti + 1}/{len(targets)}: "
                  f"pcc best {pcc_err.min().item() * 1000:.2f}mm  "
                  f"elastica best {elastica_err.min().item() * 1000:.2f}mm", flush=True)

        pcc_errs, elastica_errs = np.array(pcc_errs), np.array(elastica_errs)
        results[name] = {
            "pcc_tip_err_best_of_K_mm": float(pcc_errs.mean() * 1000),
            "pcc_success_rate_best_of_K": float((pcc_errs < tol).mean()),
            "elastica_tip_err_best_of_K_mm": float(elastica_errs.mean() * 1000),
            "elastica_success_rate_best_of_K": float((elastica_errs < tol).mean()),
            "n_targets": args.n_targets, "K": args.k,
            "wall_clock_s": time.time() - t_start,
        }
        print(name, ":", results[name])

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
