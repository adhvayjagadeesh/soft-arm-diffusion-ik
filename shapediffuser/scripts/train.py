#!/usr/bin/env python3
"""Train ShapeDiffuser or a baseline.

Usage:
    python scripts/train.py --model diffusion
    python scripts/train.py --model mlp
    python scripts/train.py --model mdn
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shapediffuser import BabblingDataset, build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["diffusion", "mlp", "mdn"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    tr_cfg, d_cfg, m_cfg = cfg["train"], cfg["data"], cfg["model"]
    device = tr_cfg["device"] if torch.cuda.is_available() else "cpu"
    if device == "cpu" and tr_cfg["device"] == "cuda":
        print("WARNING: CUDA unavailable, training on CPU (will be slow).")

    train_np = dict(np.load(os.path.join(args.data, "train.npz")))
    val_np = dict(np.load(os.path.join(args.data, "val.npz")))
    train_ds = BabblingDataset(train_np, cond_type=d_cfg["cond_type"])
    val_ds = BabblingDataset(val_np, cond_type=d_cfg["cond_type"],
                             cond_norm=train_ds.cond_norm)
    dl = DataLoader(train_ds, batch_size=tr_cfg["batch_size"], shuffle=True,
                    num_workers=4, pin_memory=(device == "cuda"), drop_last=True)
    vdl = DataLoader(val_ds, batch_size=tr_cfg["batch_size"])

    q_dim = train_ds.q.shape[1]
    cond_dim = train_ds.cond.shape[1]
    model = build_model(args.model, cond_dim, q_dim, m_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr_cfg["lr"],
                            weight_decay=tr_cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tr_cfg["epochs"])

    os.makedirs(tr_cfg["ckpt_dir"], exist_ok=True)
    best_val = float("inf")
    ckpt_path = os.path.join(tr_cfg["ckpt_dir"], f"{args.model}.pt")

    for epoch in range(tr_cfg["epochs"]):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for q, cond in dl:
            q, cond = q.to(device), cond.to(device)
            loss = model.loss(q, cond)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            vtot, vnb = 0.0, 0
            for q, cond in vdl:
                vtot += model.loss(q.to(device), cond.to(device)).item()
                vnb += 1
        vloss = vtot / max(vnb, 1)
        print(f"[{args.model}] epoch {epoch + 1}/{tr_cfg['epochs']} "
              f"train {tot / nb:.5f} val {vloss:.5f} ({time.time() - t0:.1f}s)")
        if vloss < best_val:
            best_val = vloss
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": args.model,
                    "q_dim": q_dim,
                    "cond_dim": cond_dim,
                    "cond_type": d_cfg["cond_type"],
                    "cond_norm": train_ds.cond_norm.state_dict(),
                    "model_cfg": m_cfg,
                },
                ckpt_path,
            )
    print(f"Best val loss {best_val:.5f}; checkpoint at {ckpt_path}")


if __name__ == "__main__":
    main()
