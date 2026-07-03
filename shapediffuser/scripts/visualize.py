#!/usr/bin/env python3
"""Render the headline figure: one tip target, many valid diffusion-sampled arm
shapes vs. the single (mode-averaged) MLP solution.

    python scripts/visualize.py --n_show 8 --out figures/multimodality.png
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shapediffuser import PCCArm  # noqa: E402
from shapediffuser.metrics import sample_reachable_targets, tip_error  # noqa: E402
from evaluate import load_model, make_sampler  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--target_seed", type=int, default=42)
    ap.add_argument("--n_show", type=int, default=8)
    ap.add_argument("--out", default="figures/multimodality.png")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ev = cfg["eval"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arm = PCCArm(**cfg["arm"])
    target = sample_reachable_targets(arm, 1, seed=args.target_seed)[0]

    fig = plt.figure(figsize=(11, 5))
    panels = [("diffusion", "ShapeDiffuser (ours): K samples"),
              ("mlp", "MLP regressor: single solution")]
    for pi, (name, title) in enumerate(panels):
        ax = fig.add_subplot(1, 2, pi + 1, projection="3d")
        path = os.path.join(args.ckpt_dir, f"{name}.pt")
        if not os.path.exists(path):
            ax.set_title(f"{title}\n(no checkpoint)")
            continue
        model, norm, _ = load_model(path, device)
        sampler = make_sampler(model, norm, device, ev["ddim_steps"], ev["guidance"])
        qs = sampler(target.unsqueeze(0), args.n_show)[0]
        errs = tip_error(arm, qs, target.unsqueeze(0).expand(args.n_show, -1))
        with torch.no_grad():
            bbs = arm.forward(qs)["backbone"].numpy()
        cmap = plt.get_cmap("viridis")
        for i in range(args.n_show):
            ok = errs[i].item() < ev["success_tol"]
            ax.plot(bbs[i, :, 0], bbs[i, :, 1], bbs[i, :, 2],
                    color=cmap(i / max(args.n_show - 1, 1)),
                    lw=2.2 if ok else 1.0, alpha=1.0 if ok else 0.35)
        ax.scatter(*target.numpy(), color="red", s=90, marker="*",
                   label="target tip", depthshade=False)
        ax.set_title(f"{title}\nmin tip err {errs.min().item() * 1000:.1f} mm")
        ax.set_box_aspect((1, 1, 1))
        ax.legend(loc="upper left")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
