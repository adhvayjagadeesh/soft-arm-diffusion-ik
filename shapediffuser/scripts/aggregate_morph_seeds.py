#!/usr/bin/env python3
"""Run evaluate_morph across training seeds and report mean +/- std.

One protocol, three seeds, every band, mode recall included - so the
paper's central tables all come from a single consistent evaluation rather
than a patchwork of runs with different morphology subsets.

    python scripts/aggregate_morph_seeds.py --out results_morph_3seed.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--modes", action="store_true")
    ap.add_argument("--n_in_dist", type=int, default=5)
    ap.add_argument("--out", default="results_morph_3seed.json")
    ap.add_argument("--reuse", action="store_true", help="skip seeds whose per-seed JSON exists")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    per_seed = {}
    for s in seeds:
        sfx = "" if s == 0 else f"_seed{s}"
        out = f"results_morph_eval_seed{s}.json"
        if not (args.reuse and os.path.exists(out)):
            cmd = [sys.executable, os.path.join(HERE, "evaluate_morph.py"),
                   "--models", f"amortized=checkpoints_morph_amortized{sfx}",
                   f"blind=checkpoints_morph_blind{sfx}", f"nominal=checkpoints_seed{s}",
                   "--arch", "diffusion", "--n_in_dist", str(args.n_in_dist), "--out", out]
            if args.modes:
                cmd.append("--modes")
            print(f"\n##### seed {s}: {' '.join(cmd[2:])}\n", flush=True)
            subprocess.run(cmd, check=True)
        per_seed[s] = json.load(open(out))

    # aggregate by band and by morphology
    bands = list(per_seed[seeds[0]]["by_band"])
    models = list(per_seed[seeds[0]]["by_band"][bands[0]])
    metrics = list(per_seed[seeds[0]]["by_band"][bands[0]][models[0]])
    agg = {"seeds": seeds, "by_band": {}, "by_morphology": {}}
    for band in bands:
        agg["by_band"][band] = {}
        for m in models:
            agg["by_band"][band][m] = {}
            for k in metrics:
                v = np.array([per_seed[s]["by_band"][band][m][k] for s in seeds], dtype=float)
                agg["by_band"][band][m][k] = {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v)),
                                              "per_seed": v.tolist()}
    for name in per_seed[seeds[0]]["morphologies"]:
        e0 = per_seed[seeds[0]]["morphologies"][name]
        agg["by_morphology"][name] = {"band": e0["band"], "L": e0["L"], "g": e0["g"], "models": {}}
        for m in models:
            agg["by_morphology"][name]["models"][m] = {}
            for k in metrics:
                v = np.array([per_seed[s]["morphologies"][name]["models"][m][k] for s in seeds], dtype=float)
                agg["by_morphology"][name]["models"][m][k] = {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v))}

    with open(args.out, "w") as f:
        json.dump(agg, f, indent=2)

    print(f"\n===== {len(seeds)}-seed mean +/- std by band =====")
    hdr = f"{'band':9} {'model':22} {'best-of-K mm':>16} {'success':>14}"
    if args.modes:
        hdr += f" {'recall':>14} {'diversity':>14}"
    print(hdr)
    for band in bands:
        for m in models:
            r = agg["by_band"][band][m]
            line = (f"{band:9} {m:22} {r['tip_err_best_of_K_mm']['mean']:7.2f} ± {r['tip_err_best_of_K_mm']['std']:<5.2f}"
                    f" {r['success_rate_best_of_K']['mean']:6.3f} ± {r['success_rate_best_of_K']['std']:<5.3f}")
            if args.modes:
                line += (f" {r['mode_recall']['mean']:6.2f} ± {r['mode_recall']['std']:<5.2f}"
                         f" {r['diversity']['mean']:6.1f} ± {r['diversity']['std']:<5.1f}")
            print(line)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
