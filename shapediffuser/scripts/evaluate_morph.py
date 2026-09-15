#!/usr/bin/env python3
"""Evaluate IK models on HELD-OUT morphologies.

The question this answers: can one model, conditioned on morphology, solve
inverse kinematics for arms it never saw - and does it need to be TOLD the
morphology, or is training on a randomized family enough?

Three models, one test suite:

  amortized  trained on the randomized family, conditioned on [target, morph]
  blind      trained on the SAME randomized data, conditioned on [target] only
             (Paper 1's domain randomization, extended to length + gain)
  nominal    trained on the single Paper-1 arm, conditioned on [target]
             (what happens if you just use the old model on a new arm)

Test morphologies fall in three bands relative to the training box
(L in [0.04,0.08], g in [15,35], per segment):

  nominal        the Paper-1 arm - inside the box, so an interpolation test
  in-dist        random draws from the box, never seen as exact values
  extrapolation  outside the box on one or more axes

Every metric is computed on the TEST arm's own kinematics: a PCCArm is
instantiated at that morphology and the whole Paper-1 metric suite runs on it
unchanged.

    python scripts/evaluate_morph.py --out results_morph.json
    python scripts/evaluate_morph.py --modes        # + ground-truth mode recall (slow)
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

from shapediffuser import PCCArm, BabblingDataset                         # noqa: E402
from shapediffuser.metrics import (                                        # noqa: E402
    tip_error, curvature_features, diversity, enumerate_modes, mode_recall,
    sample_reachable_targets,
)
from evaluate import load_model                                            # noqa: E402

N = 4
BOX = {"seg_length": (0.04, 0.08), "curvature_gain": (15.0, 35.0)}


def test_morphologies(n_in_dist: int = 5, seed: int = 11) -> list[dict]:
    """Named test arms. Extrapolation cases sit outside the training box."""
    rng = np.random.default_rng(seed)
    ms = [{"name": "nominal", "band": "nominal", "L": [0.06] * N, "g": [25.0] * N}]
    for i in range(n_in_dist):
        ms.append({"name": f"in_dist_{i}", "band": "in_dist",
                   "L": rng.uniform(*BOX["seg_length"], N).round(4).tolist(),
                   "g": rng.uniform(*BOX["curvature_gain"], N).round(2).tolist()})
    ms += [
        {"name": "extrap_long",      "band": "extrap", "L": [0.09] * N, "g": [25.0] * N},
        {"name": "extrap_short",     "band": "extrap", "L": [0.03] * N, "g": [25.0] * N},
        {"name": "extrap_stiff",     "band": "extrap", "L": [0.06] * N, "g": [40.0] * N},
        {"name": "extrap_compliant", "band": "extrap", "L": [0.06] * N, "g": [10.0] * N},
        {"name": "extrap_mixed",     "band": "extrap",
         "L": [0.09, 0.03, 0.09, 0.03], "g": [40.0, 10.0, 40.0, 10.0]},
    ]
    return ms


def make_sampler(model, norm, device, ddim_steps, guidance, cond_type, morph_vec):
    """evaluate.make_sampler, plus: append the morphology when the model expects it.

    The same Normalizer that trained the model encodes the concatenated
    condition, so [target, morph] is z-scored exactly as during training."""
    wants_morph = cond_type.endswith("_morph")

    def f(targets: torch.Tensor, k: int) -> torch.Tensor:
        raw = targets
        if wants_morph:
            m = morph_vec.to(targets).view(1, -1).expand(targets.shape[0], -1)
            raw = torch.cat([targets, m], dim=1)
        cond = norm.encode(raw.to(device))
        kwargs = {"steps": ddim_steps, "guidance": guidance} if hasattr(model, "alphas_cumprod") else {}
        return BabblingDataset.q_to_pressures(model.sample(cond, n_samples=k, **kwargs)).cpu()

    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/morph_amortized.yaml")
    ap.add_argument("--models", nargs="+", default=[
        "amortized=checkpoints_morph_amortized",
        "blind=checkpoints_morph_blind",
        "nominal=checkpoints_seed0",
    ], help="label=ckpt_dir; each dir may hold diffusion.pt / mlp.pt / mdn.pt")
    ap.add_argument("--arch", nargs="+", default=["diffusion"],
                    help="which architectures to evaluate from each dir")
    ap.add_argument("--modes", action="store_true", help="ground-truth mode recall (slow)")
    ap.add_argument("--n_in_dist", type=int, default=5)
    ap.add_argument("--out", default="results_morph.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ev = cfg["eval"]
    device = "cpu"
    tol, K = ev["success_tol"], ev["n_samples_per_target"]

    # load every (label, arch) that exists
    loaded = {}
    for spec in args.models:
        label, ckdir = spec.split("=", 1)
        for arch in args.arch:
            p = os.path.join(ckdir, f"{arch}.pt")
            if not os.path.exists(p):
                print(f"(no {p}, skipping)")
                continue
            model, norm, ck = load_model(p, device)
            loaded[f"{label}/{arch}"] = (model, norm, ck)
            print(f"loaded {label}/{arch}: cond_type={ck['cond_type']} cond_dim={ck['cond_dim']}")
    if not loaded:
        sys.exit("nothing to evaluate")

    results = {"box": BOX, "morphologies": {}, "tol_m": tol, "K": K}
    for m in test_morphologies(args.n_in_dist):
        arm = PCCArm(n_segments=N, seg_length=m["L"], curvature_gain=m["g"],
                     points_per_seg=cfg["arm"]["points_per_seg"], rod_radius=cfg["arm"]["rod_radius"])
        morph_vec = arm.morph
        reach = float(np.linalg.norm(sample_reachable_targets(arm, 2000, seed=3).numpy(), axis=1).max())
        print(f"\n=== {m['name']} [{m['band']}]  L={m['L']} g={m['g']}  max reach {reach*1000:.0f} mm ===")
        entry = {"band": m["band"], "L": m["L"], "g": m["g"], "max_reach_mm": reach * 1000, "models": {}}

        targets = sample_reachable_targets(arm, ev["n_targets_accuracy"], seed=1)
        mode_targets = sample_reachable_targets(arm, ev["n_targets_modes"], seed=777) if args.modes else None
        gts = None
        if args.modes:
            gts = [enumerate_modes(arm, t, pool_size=ev["mode_pool"], tol=tol,
                                   dbscan_eps=ev["dbscan_eps"], min_samples=ev.get("dbscan_min_samples", 10))
                   for t in mode_targets]
            entry["gt_modes_mean"] = float(np.mean([g.shape[0] for g in gts]))
            print(f"  ground-truth modes: mean {entry['gt_modes_mean']:.2f} over {len(gts)} targets")

        for key, (model, norm, ck) in loaded.items():
            sampler = make_sampler(model, norm, device, ev["ddim_steps"], ev["guidance"], ck["cond_type"], morph_vec)
            with torch.no_grad():
                qs = sampler(targets, K)
            errs = tip_error(arm, qs, targets.unsqueeze(1))
            r = {
                "tip_err_best_of_K_mm": errs.min(dim=1).values.mean().item() * 1000,
                "tip_err_single_mm": errs[:, 0].mean().item() * 1000,
                "success_rate_best_of_K": (errs.min(dim=1).values < tol).float().mean().item(),
            }
            if args.modes:
                rec, div = [], []
                for t, gt in zip(mode_targets, gts):
                    if gt.shape[0] == 0:
                        continue
                    with torch.no_grad():
                        q1 = sampler(t.unsqueeze(0), K)[0]
                    e1 = tip_error(arm, q1, t.unsqueeze(0).expand(K, -1))
                    ok = e1 < tol
                    feats = curvature_features(arm, q1)
                    rec.append(mode_recall(gt, feats, ok, radius=ev["mode_match_radius"]))
                    div.append(diversity(feats[ok]))
                r["mode_recall"] = float(np.nanmean(rec)) if rec else float("nan")
                r["diversity"] = float(np.mean(div)) if div else float("nan")
            entry["models"][key] = r
            print(f"  {key:22} best-of-K {r['tip_err_best_of_K_mm']:6.2f} mm  "
                  f"success {r['success_rate_best_of_K']:.3f}"
                  + (f"  recall {r['mode_recall']:.2f}  div {r['diversity']:.1f}" if args.modes else ""))
        results["morphologies"][m["name"]] = entry

    # band aggregates: the numbers the paper reports
    results["by_band"] = {}
    for band in ("nominal", "in_dist", "extrap"):
        ents = [e for e in results["morphologies"].values() if e["band"] == band]
        if not ents:
            continue
        agg = {}
        for key in loaded:
            vals = [e["models"][key] for e in ents if key in e["models"]]
            agg[key] = {k: float(np.nanmean([v[k] for v in vals])) for k in vals[0]}
        results["by_band"][band] = agg

    print("\n=== by band (mean over morphologies) ===")
    print(f"{'band':9} {'model':22} {'best-of-K mm':>13} {'success':>8}" + ("  recall   div" if args.modes else ""))
    for band, agg in results["by_band"].items():
        for key, r in agg.items():
            print(f"{band:9} {key:22} {r['tip_err_best_of_K_mm']:13.2f} {r['success_rate_best_of_K']:8.3f}"
                  + (f"  {r['mode_recall']:6.2f} {r['diversity']:5.1f}" if args.modes else ""))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
