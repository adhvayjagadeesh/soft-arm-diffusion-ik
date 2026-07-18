#!/usr/bin/env python3
"""Morphology-generalization sweep: does the core E1/E3/E4 story hold across
different segment counts, not just the one 4-segment arm used everywhere
else?

Trains diffusion+mlp fresh on 2-segment and 6-segment PCC arms (configs/
morph_2seg.yaml, morph_6seg.yaml) with their own data directories, and
evaluates with --skip_modes (E2's ground-truth mode enumeration was
calibrated - dbscan_eps/mode_match_radius - for the 4-segment arm's
curvature scale and isn't trustworthy for other morphologies without
recalibration, out of scope for this sweep). The 4-segment point reuses the
already-trained checkpoints/ and results.json rather than retraining.

    python scripts/morphology_sweep.py
"""

import argparse
import json
import os
import subprocess
import sys


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="morphology_sweep_results.json")
    args = ap.parse_args()
    py = sys.executable

    morphologies = {
        2: {"config": "configs/morph_2seg.yaml", "data": "data_morph2seg",
            "ckpt_dir": "checkpoints_morph2seg", "results": "results_morph2seg.json"},
        4: {"config": "configs/default.yaml", "data": "data",
            "ckpt_dir": "checkpoints", "results": "results.json"},
        6: {"config": "configs/morph_6seg.yaml", "data": "data_morph6seg",
            "ckpt_dir": "checkpoints_morph6seg", "results": "results_morph6seg.json"},
        8: {"config": "configs/morph_8seg.yaml", "data": "data_morph8seg",
            "ckpt_dir": "checkpoints_morph8seg", "results": "results_morph8seg.json"},
        10: {"config": "configs/morph_10seg.yaml", "data": "data_morph10seg",
             "ckpt_dir": "checkpoints_morph10seg", "results": "results_morph10seg.json"},
    }

    per_morph = {}
    for n_seg, m in morphologies.items():
        if n_seg == 4 and os.path.exists(m["results"]):
            print(f"n_segments=4: reusing existing {m['results']}")
            per_morph[n_seg] = json.load(open(m["results"]))
            continue
        if os.path.exists(m["results"]):
            print(f"n_segments={n_seg}: reusing existing {m['results']}")
            per_morph[n_seg] = json.load(open(m["results"]))
            continue

        if not os.path.exists(os.path.join(m["data"], "train.npz")):
            run([py, "scripts/generate_data.py", "--config", m["config"], "--out", m["data"]])
        for model in ("diffusion", "mlp"):
            ckpt_path = os.path.join(m["ckpt_dir"], f"{model}.pt")
            if os.path.exists(ckpt_path):
                print(f"n_segments={n_seg} {model}: reusing existing {ckpt_path}")
                continue
            run([py, "scripts/train.py", "--model", model, "--config", m["config"],
                 "--data", m["data"], "--ckpt_dir", m["ckpt_dir"]])
        run([py, "scripts/evaluate.py", "--config", m["config"], "--ckpt_dir", m["ckpt_dir"],
             "--out", m["results"], "--skip_modes"])
        per_morph[n_seg] = json.load(open(m["results"]))

    summary = {}
    for n_seg, r in per_morph.items():
        summary[n_seg] = {}
        for model in ("diffusion", "mlp"):
            if model not in r:
                continue
            summary[n_seg][model] = {
                "tip_err_best_of_K_mm": r[model]["accuracy"]["tip_err_best_of_K_mm"],
                "success_rate_best_of_K": r[model]["accuracy"]["success_rate_best_of_K"],
                "obstacle_task_success_rate": r[model].get("obstacle_task", {}).get("success_rate"),
                "ms_per_target": r[model]["timing"]["ms_per_target_batchK"],
            }

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out}")
    print(f"\n{'n_segments':>10} {'model':>10} {'tip err mm':>12} {'success':>9} {'obstacle':>9} {'ms/target':>10}")
    for n_seg in sorted(summary.keys()):
        for model in ("diffusion", "mlp"):
            if model not in summary[n_seg]:
                continue
            d = summary[n_seg][model]
            print(f"{n_seg:>10} {model:>10} {d['tip_err_best_of_K_mm']:>12.3f} "
                  f"{d['success_rate_best_of_K']:>9.3f} "
                  f"{d['obstacle_task_success_rate']:>9.3f} {d['ms_per_target']:>10.3f}")


if __name__ == "__main__":
    main()
