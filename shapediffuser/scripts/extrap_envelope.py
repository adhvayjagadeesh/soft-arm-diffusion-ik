#!/usr/bin/env python3
"""Map the generalization envelope of an amortized IK model.

Sweep one morphology axis at a time - all four segment lengths together, or
all four gains together - from well inside the training box to well outside
it, and record accuracy at each point. The result is the curve a reader needs
to judge "generalizes outside the box": not a single extrapolation number but
the SHAPE of the degradation, and where it starts.

Training box: L in [0.04, 0.08], g in [15, 35].

    python scripts/extrap_envelope.py --out results_envelope.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from shapediffuser import PCCArm                          # noqa: E402
from shapediffuser.metrics import tip_error, sample_reachable_targets  # noqa: E402
from evaluate import load_model                           # noqa: E402
from evaluate_morph import make_sampler, BOX, N           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/morph_amortized.yaml")
    ap.add_argument("--models", nargs="+", default=[
        "amortized=checkpoints_morph_amortized", "blind=checkpoints_morph_blind"])
    ap.add_argument("--L", default="0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.12")
    ap.add_argument("--g", default="5,10,15,20,25,30,35,40,45,50,60")
    ap.add_argument("--n_targets", type=int, default=256)
    ap.add_argument("--out", default="results_envelope.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ev = cfg["eval"]
    tol, K = ev["success_tol"], ev["n_samples_per_target"]
    loaded = {}
    for spec in args.models:
        label, d = spec.split("=", 1)
        p = os.path.join(d, "diffusion.pt")
        if os.path.exists(p):
            loaded[label] = load_model(p, "cpu")

    def run(L, g):
        arm = PCCArm(n_segments=N, seg_length=L, curvature_gain=g,
                     points_per_seg=cfg["arm"]["points_per_seg"], rod_radius=cfg["arm"]["rod_radius"])
        targets = sample_reachable_targets(arm, args.n_targets, seed=1)
        out = {}
        for label, (model, norm, ck) in loaded.items():
            s = make_sampler(model, norm, "cpu", ev["ddim_steps"], ev["guidance"], ck["cond_type"], arm.morph)
            with torch.no_grad():
                errs = tip_error(arm, s(targets, K), targets.unsqueeze(1))
            out[label] = {"best_of_K_mm": errs.min(1).values.mean().item() * 1000,
                          "success": (errs.min(1).values < tol).float().mean().item()}
        return out

    res = {"box": BOX, "sweeps": {}}
    for axis, vals, fixed in (("seg_length", [float(x) for x in args.L.split(",")], 25.0),
                              ("curvature_gain", [float(x) for x in args.g.split(",")], 0.06)):
        lo, hi = BOX[axis]
        print(f"\n=== sweep {axis} (box {lo}-{hi}); other axis fixed at nominal ===")
        print(f"{'value':>8} {'in box':>7}   " + "   ".join(f"{l:>18}" for l in loaded))
        rows = []
        for v in vals:
            L, g = ([v] * N, [fixed] * N) if axis == "seg_length" else ([fixed] * N, [v] * N)
            r = run(L, g)
            inside = lo <= v <= hi
            rows.append({"value": v, "in_box": inside, **{k: r[k] for k in r}})
            print(f"{v:>8.3g} {'yes' if inside else 'NO':>7}   " +
                  "   ".join(f"{r[l]['best_of_K_mm']:8.2f}mm {r[l]['success']:6.1%}" for l in loaded))
        res["sweeps"][axis] = rows
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
