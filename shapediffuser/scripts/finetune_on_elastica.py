#!/usr/bin/env python3
"""Fine-tune the PCC-pretrained diffusion model on Elastica-simulated data.

The real-data-mixing thread: domain randomization (synthetic curvature_gain
variation) failed to close the sim-to-sim gap (see README). This tests the
more principled alternative - a modest amount of genuinely Elastica-simulated
(actuation, tip) data (elastica_finetune_data.npz, 10,000 samples,
scripts/generate_elastica_dataset.py), used to fine-tune the existing
500k-PCC-pretrained checkpoint rather than training from scratch.

Design choices (agreed before running):
  - Heavy oversampling: each prefix's fine-tune trains exclusively on that
    prefix of Elastica data (no PCC data mixed in this phase) - otherwise a
    few thousand real samples would be drowned out by 500k PCC samples and
    prove nothing either way (the natural-ratio-mixing failure mode).
  - Full prefix scaling curve (500/1000/3000/5000/10000), not just the full
    10,000 - free once the data exists (fine-tuning is cheap; only
    generation was expensive), and shows whether the effect (if any)
    saturates early like the PCC dataset-size ablation did.
  - Reuses the ORIGINAL checkpoint's Normalizer (never refit on the tiny
    Elastica set) - the pretrained denoiser network already learned to
    interpret conditioning vectors in that normalizer's coordinate system;
    refitting on ~500-10000 samples would shift statistics arbitrarily and
    misalign everything the frozen architecture already learned.
  - Low fine-tuning LR (10x smaller than pretraining) and few epochs (20),
    standard fine-tuning practice to avoid catastrophic forgetting of
    PCC-domain behavior for the sake of a small correction set.

Each prefix's fine-tune starts fresh from the same base checkpoint (not
chained across prefixes), then both PCC self-consistency (evaluate.py
--skip_modes against the standard nominal arm) and Elastica transfer
(transfer_study.py) are evaluated on a fresh held-out target seed never
touched by data generation or any earlier experiment this session.

    python scripts/finetune_on_elastica.py --prefixes 500,1000,3000,5000,10000
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shapediffuser import BabblingDataset, build_model  # noqa: E402
from evaluate import load_model  # noqa: E402


def finetune_one(base_ckpt_path, elastica_data, n_prefix, ckpt_dir, cfg,
                 epochs, lr, batch_size, device):
    model, cond_norm, base_ck = load_model(base_ckpt_path, device)
    diffusion = model  # GaussianDiffusion, already has denoiser weights loaded

    sub = {k: v[:n_prefix] for k, v in elastica_data.items()}
    ds = BabblingDataset(sub, cond_type=cfg["data"]["cond_type"], cond_norm=cond_norm)
    bs = min(batch_size, len(ds))
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=(len(ds) > bs))

    opt = torch.optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    diffusion.train()
    for epoch in range(epochs):
        tot, nb = 0.0, 0
        for q, cond in dl:
            q, cond = q.to(device), cond.to(device)
            loss = diffusion.loss(q, cond)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  n={n_prefix} epoch {epoch + 1}/{epochs} loss {tot / max(nb, 1):.5f}", flush=True)

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "diffusion.pt")
    torch.save(
        {
            "model": diffusion.state_dict(),
            "model_name": "diffusion",
            "q_dim": base_ck["q_dim"],
            "cond_dim": base_ck["cond_dim"],
            "cond_type": base_ck["cond_type"],
            "cond_norm": cond_norm.state_dict(),
            "model_cfg": base_ck["model_cfg"],
        },
        ckpt_path,
    )
    print(f"  wrote {ckpt_path}", flush=True)


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--base_ckpt", default="checkpoints/diffusion.pt")
    ap.add_argument("--elastica_data", default="elastica_finetune_data.npz")
    ap.add_argument("--prefixes", default="500,1000,3000,5000,10000")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--held_out_seed", type=int, default=55555,
                    help="fresh seed, never used by data generation or earlier experiments")
    ap.add_argument("--n_targets", type=int, default=20)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default="finetune_scaling_results.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    elastica_data = dict(np.load(args.elastica_data))
    prefixes = [int(p) for p in args.prefixes.split(",")]
    py = sys.executable

    results = {}
    for n in prefixes:
        print(f"\n=== prefix n={n} ===", flush=True)
        ckpt_dir = f"checkpoints_finetune_{n}"

        finetune_one(args.base_ckpt, elastica_data, n, ckpt_dir, cfg,
                    args.epochs, args.lr, args.batch_size, device)

        pcc_out = f"results_finetune_{n}_pcc.json"
        run([py, "scripts/evaluate.py", "--config", args.config, "--ckpt_dir", ckpt_dir,
             "--out", pcc_out, "--skip_modes"])

        transfer_out = f"transfer_study_finetune_{n}_results.json"
        run([py, "scripts/transfer_study.py", "--config", args.config, "--ckpt_dir", ckpt_dir,
             "--n_targets", str(args.n_targets), "--k", str(args.k),
             "--seed", str(args.held_out_seed), "--out", transfer_out])

        results[n] = {
            "pcc": json.load(open(pcc_out)).get("diffusion", {}),
            "transfer": json.load(open(transfer_out)).get("diffusion", {}),
        }

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")

    print(f"\n{'n':>7} {'pcc_err_mm':>10} {'pcc_succ':>9} {'elastica_err_mm':>16} {'elastica_succ':>13}")
    for n in prefixes:
        r = results[n]
        pcc_err = r["pcc"].get("accuracy", {}).get("tip_err_best_of_K_mm", float("nan"))
        pcc_succ = r["pcc"].get("accuracy", {}).get("success_rate_best_of_K", float("nan"))
        el_err = r["transfer"].get("elastica_tip_err_best_of_K_mm", float("nan"))
        el_succ = r["transfer"].get("elastica_success_rate_best_of_K", float("nan"))
        print(f"{n:>7} {pcc_err:>10.3f} {pcc_succ:>9.3f} {el_err:>16.2f} {el_succ:>13.3f}")


if __name__ == "__main__":
    main()
