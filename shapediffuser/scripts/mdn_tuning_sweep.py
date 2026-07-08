#!/usr/bin/env python3
"""MDN fairness sweep: does tuning the component count rescue the MDN?

The paper's most attackable baseline point (flagged in the hostile
self-review, and predicted by the project's original handoff notes): the MDN
was only ever run at its default 8 components. This sweeps mdn_k over
{4, 8, 16, 32}, training each fresh on the standard 500k dataset with the
standard protocol and running the full evaluation (E1 accuracy, E2 mode
recall - the question that matters - E3 obstacle task). If no component
count changes the story, the paper can claim the baseline was tuned; if one
does, the paper must say so.

    python scripts/mdn_tuning_sweep.py --ks 4,8,16,32
"""

import argparse
import json
import os
import subprocess
import sys

import yaml


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data")
    ap.add_argument("--ks", default="4,8,16,32")
    ap.add_argument("--out", default="mdn_tuning_results.json")
    args = ap.parse_args()
    ks = [int(k) for k in args.ks.split(",")]
    py = sys.executable

    base_cfg = yaml.safe_load(open(args.config))
    results = {}
    for k in ks:
        print(f"\n=== mdn_k={k} ===", flush=True)
        cfg = json.loads(json.dumps(base_cfg))  # deep copy
        cfg["model"]["mdn_k"] = k
        cfg_path = f"configs/_mdn_sweep_k{k}.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)

        ckpt_dir = f"checkpoints_mdn_k{k}"
        res_path = f"results_mdn_k{k}.json"
        if not os.path.exists(res_path):
            run([py, "scripts/train.py", "--model", "mdn", "--config", cfg_path,
                 "--data", args.data, "--ckpt_dir", ckpt_dir])
            run([py, "scripts/evaluate.py", "--config", cfg_path,
                 "--ckpt_dir", ckpt_dir, "--out", res_path])
        else:
            print(f"reusing existing {res_path}")
        results[k] = json.load(open(res_path)).get("mdn", {})
        os.remove(cfg_path)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")

    print(f"\n{'k':>4} {'tip_err_mm':>11} {'success':>8} {'mode_recall':>12} "
          f"{'diversity':>10} {'obstacle':>9}")
    for k in ks:
        r = results[k]
        print(f"{k:>4} {r['accuracy']['tip_err_best_of_K_mm']:>11.3f} "
              f"{r['accuracy']['success_rate_best_of_K']:>8.3f} "
              f"{r['multimodality']['mode_recall']:>12.3f} "
              f"{r['multimodality']['diversity']:>10.3f} "
              f"{r['obstacle_task']['success_rate']:>9.3f}")


if __name__ == "__main__":
    main()
