#!/usr/bin/env python3
"""Direct test of the mechanism claimed for the Blind model.

Claim: with the morphology withheld, the model learns the MARGGINAL
    p(q | tip) = integral p(q | tip, theta) p(theta) dtheta
so each of its samples is a correct actuation for SOME arm in the family,
just not (except by chance) for the arm being queried.

Prediction, testable: for a Blind sample q drawn for target t, there exists
a theta in the training box such that FK(q; theta) lands within tolerance
of t. The fraction of samples for which this holds should be high for Blind
- and low for random actuations, which is the control that shows the test
has teeth. Amortized samples should also pass (trivially: they work on the
queried theta), which is the sanity bound.

    python scripts/blind_marginal_test.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from shapediffuser import PCCArm                                    # noqa: E402
from shapediffuser.metrics import sample_reachable_targets         # noqa: E402
from evaluate import load_model                                     # noqa: E402
from evaluate_morph import make_sampler, BOX, N                     # noqa: E402

N_THETA = 20000      # random morphologies searched per sample
TOL = 0.005
K = 32
N_TARGETS = 20


def min_dist_over_family(arm, q, t, rng):
    """min over theta in the box of ||FK(q; theta) - t||, by random search.
    q: (q_dim,), t: (3,). Returns metres."""
    Ls = rng.uniform(*BOX["seg_length"], (N_THETA, N))
    Gs = rng.uniform(*BOX["curvature_gain"], (N_THETA, N))
    qb = q.view(1, -1).expand(N_THETA, -1)
    with torch.no_grad():
        tips = arm.forward(qb, gain=torch.as_tensor(Gs, dtype=torch.float32),
                           lengths=torch.as_tensor(Ls, dtype=torch.float32))["tip"]
    return (tips - t.view(1, 3)).norm(dim=1).min().item()


def main():
    cfg = yaml.safe_load(open("configs/morph_amortized.yaml"))
    ev = cfg["eval"]
    rng = np.random.default_rng(0)
    models = {}
    for label, d in (("blind", "checkpoints_morph_blind"), ("amortized", "checkpoints_morph_amortized")):
        models[label] = load_model(os.path.join(d, "diffusion.pt"), "cpu")

    # query on three arms: nominal, one in-dist, one extrap
    arms = {
        "nominal":  ([0.06] * N, [25.0] * N),
        "in_dist":  ([0.0451, 0.06, 0.0641, 0.0411], [17.96, 33.56, 16.41, 17.6]),
        "extrap_stiff": ([0.06] * N, [40.0] * N),
    }
    out = {"n_theta_searched": N_THETA, "tol_m": TOL, "arms": {}}
    print(f"For each sample: is it a valid solution on SOME arm in the box? ({N_THETA} random thetas searched)\n")
    print(f"{'arm':14}{'model':12}{'valid on queried arm':>22}{'valid on SOME arm':>20}{'valid nowhere':>16}")
    for name, (L, g) in arms.items():
        arm = PCCArm(n_segments=N, seg_length=L, curvature_gain=g)
        targets = sample_reachable_targets(arm, N_TARGETS, seed=777)
        out["arms"][name] = {}
        rows = {}
        for label, (model, norm, ck) in models.items():
            s = make_sampler(model, norm, "cpu", ev["ddim_steps"], ev["guidance"], ck["cond_type"], arm.morph)
            on_query, on_some = [], []
            for t in targets:
                with torch.no_grad():
                    qs = s(t.unsqueeze(0), K)[0]
                    tips_q = arm.forward(qs)["tip"]
                dq = (tips_q - t.view(1, 3)).norm(dim=1)
                for i in range(K):
                    on_query.append(dq[i].item() < TOL)
                    on_some.append(min_dist_over_family(arm, qs[i], t, rng) < TOL)
            rows[label] = (np.mean(on_query), np.mean(on_some))
        # control: random actuations
        on_query, on_some = [], []
        for t in targets:
            qs = torch.rand(K, arm.q_dim)
            with torch.no_grad():
                dq = (arm.forward(qs)["tip"] - t.view(1, 3)).norm(dim=1)
            for i in range(K):
                on_query.append(dq[i].item() < TOL)
                on_some.append(min_dist_over_family(arm, qs[i], t, rng) < TOL)
        rows["random q"] = (np.mean(on_query), np.mean(on_some))
        for label, (a, b) in rows.items():
            print(f"{name:14}{label:12}{a:>21.1%}{b:>20.1%}{1-b:>16.1%}")
            out["arms"][name][label] = {"valid_on_queried_arm": a, "valid_on_some_arm": b}
        print()
    with open("results_blind_marginal.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results_blind_marginal.json")


if __name__ == "__main__":
    main()
