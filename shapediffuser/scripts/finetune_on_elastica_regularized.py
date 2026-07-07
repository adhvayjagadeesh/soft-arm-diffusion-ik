#!/usr/bin/env python3
"""Regularized version of finetune_on_elastica.py: mix PCC data back into
fine-tuning to prevent catastrophic forgetting.

finetune_on_elastica.py's heavy-oversampling design (fine-tune exclusively
on Elastica data) confirmed real transfer signal (140.6mm -> 63.5mm Elastica
error, 0% -> 10% success at n=10,000) but at the cost of catastrophically
forgetting PCC-domain accuracy (0.44mm -> 136.3mm, 100% -> 0% success) - a
genuine Pareto tradeoff, not a usable result on its own.

This targets the identified causal mechanism directly: each fine-tuning
batch is now a ~50/50 mix of PCC and Elastica samples (via a
WeightedRandomSampler over a ConcatDataset - PCC samples individually
downweighted since there are 500k of them vs. a few thousand Elastica
samples, so the expected composition stays ~50/50 despite the size
imbalance), rather than exclusively Elastica data. Everything else (epochs,
LR, prefix size, held-out eval seed) is held identical to the n=10,000 point
of the prior sweep, isolating this one variable for a direct, fair
comparison: PCC data mixed in vs. not.

    python scripts/finetune_on_elastica_regularized.py --n_elastica 10000
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shapediffuser import BabblingDataset  # noqa: E402
from evaluate import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--base_ckpt", default="checkpoints/diffusion.pt")
    ap.add_argument("--pcc_data", default="data")
    ap.add_argument("--elastica_data", default="elastica_finetune_data.npz")
    ap.add_argument("--n_elastica", type=int, default=10000)
    ap.add_argument("--pcc_mix_frac", type=float, default=0.5,
                    help="expected fraction of each batch drawn from PCC data")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--steps_per_epoch", type=int, default=None,
                    help="default: len(elastica subset)/batch_size, matching the "
                         "unregularized run's epoch length for a fair comparison")
    ap.add_argument("--held_out_seed", type=int, default=55555,
                    help="matches finetune_on_elastica.py's held-out seed for direct comparison")
    ap.add_argument("--n_targets", type=int, default=20)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--ckpt_dir", default="checkpoints_finetune_regularized")
    ap.add_argument("--out", default="finetune_regularized_results.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, cond_norm, base_ck = load_model(args.base_ckpt, device)
    diffusion = model

    pcc_np = dict(np.load(os.path.join(args.pcc_data, "train.npz")))
    pcc_ds = BabblingDataset(pcc_np, cond_type=cfg["data"]["cond_type"], cond_norm=cond_norm)

    elastica_np = dict(np.load(args.elastica_data))
    elastica_sub = {k: v[: args.n_elastica] for k, v in elastica_np.items()}
    elastica_ds = BabblingDataset(elastica_sub, cond_type=cfg["data"]["cond_type"], cond_norm=cond_norm)

    combined = ConcatDataset([pcc_ds, elastica_ds])
    n_pcc, n_elastica = len(pcc_ds), len(elastica_ds)
    w_pcc = args.pcc_mix_frac / n_pcc
    w_elastica = (1.0 - args.pcc_mix_frac) / n_elastica
    weights = torch.cat([torch.full((n_pcc,), w_pcc), torch.full((n_elastica,), w_elastica)])

    steps_per_epoch = args.steps_per_epoch or max(1, n_elastica // args.batch_size)
    num_samples_per_epoch = steps_per_epoch * args.batch_size
    sampler = WeightedRandomSampler(weights, num_samples=num_samples_per_epoch, replacement=True)
    dl = DataLoader(combined, batch_size=args.batch_size, sampler=sampler, drop_last=True)

    opt = torch.optim.AdamW(diffusion.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print(f"n_pcc={n_pcc} n_elastica={n_elastica} target_pcc_frac={args.pcc_mix_frac} "
          f"steps_per_epoch={steps_per_epoch}", flush=True)

    diffusion.train()
    for epoch in range(args.epochs):
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
            print(f"epoch {epoch + 1}/{args.epochs} loss {tot / max(nb, 1):.5f}", flush=True)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, "diffusion.pt")
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
    print(f"wrote {ckpt_path}", flush=True)

    py = sys.executable
    pcc_out = "results_finetune_regularized_pcc.json"
    subprocess.run([py, "scripts/evaluate.py", "--config", args.config, "--ckpt_dir", args.ckpt_dir,
                    "--out", pcc_out, "--skip_modes"], check=True)

    transfer_out = "transfer_study_finetune_regularized_results.json"
    subprocess.run([py, "scripts/transfer_study.py", "--config", args.config, "--ckpt_dir", args.ckpt_dir,
                    "--n_targets", str(args.n_targets), "--k", str(args.k),
                    "--seed", str(args.held_out_seed), "--out", transfer_out], check=True)

    results = {
        "pcc_mix_frac": args.pcc_mix_frac, "n_elastica": args.n_elastica,
        "pcc": json.load(open(pcc_out)).get("diffusion", {}),
        "transfer": json.load(open(transfer_out)).get("diffusion", {}),
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")
    pcc_err = results["pcc"].get("accuracy", {}).get("tip_err_best_of_K_mm", float("nan"))
    pcc_succ = results["pcc"].get("accuracy", {}).get("success_rate_best_of_K", float("nan"))
    el_err = results["transfer"].get("elastica_tip_err_best_of_K_mm", float("nan"))
    el_succ = results["transfer"].get("elastica_success_rate_best_of_K", float("nan"))
    print(f"pcc_err_mm={pcc_err:.3f} pcc_succ={pcc_succ:.3f} "
          f"elastica_err_mm={el_err:.2f} elastica_succ={el_succ:.3f}")
    print("compare vs unregularized n=10000: pcc_err=136.259 pcc_succ=0.000 "
          "elastica_err=63.47 elastica_succ=0.100")
    print("compare vs no-finetune baseline: pcc_err=0.438 pcc_succ=1.000 "
          "elastica_err=140.58 elastica_succ=0.000")


if __name__ == "__main__":
    main()
