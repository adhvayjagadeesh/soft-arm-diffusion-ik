#!/usr/bin/env python3
"""Generate motor-babbling datasets (train + val) from the PCC arm."""

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shapediffuser import PCCArm, generate_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    arm = PCCArm(**cfg["arm"])
    d = cfg["data"]
    os.makedirs(args.out, exist_ok=True)

    gain_range = tuple(d["curvature_gain_range"]) if d.get("curvature_gain_range") else None
    if gain_range:
        print(f"Domain randomization: curvature_gain ~ Uniform{gain_range} per sample "
              f"(nominal arm.curvature_gain={cfg['arm']['curvature_gain']} unused for generation)")
    morph = d.get("morph_ranges")
    if morph:
        morph = {k: tuple(v) for k, v in morph.items()}
        print(f"Amortized morphology: per-sample seg_length ~ U{morph['seg_length']}, "
              f"curvature_gain ~ U{morph['curvature_gain']}, per segment; morph written to dataset")

    print(f"Generating {d['n_train']} train samples ({d['mode']}) ...")
    train = generate_dataset(arm, d["n_train"], mode=d["mode"],
                             babble_step=d["babble_step"],
                             shape_points=d["shape_points"], seed=d["seed"],
                             curvature_gain_range=gain_range, morph_ranges=morph)
    np.savez_compressed(os.path.join(args.out, "train.npz"), **train)

    print(f"Generating {d['n_val']} val samples (uniform) ...")
    val = generate_dataset(arm, d["n_val"], mode="uniform",
                           shape_points=d["shape_points"], seed=d["seed"] + 1,
                           curvature_gain_range=gain_range, morph_ranges=morph)
    np.savez_compressed(os.path.join(args.out, "val.npz"), **val)
    print(f"Done. Wrote {args.out}/train.npz and {args.out}/val.npz")


if __name__ == "__main__":
    main()
