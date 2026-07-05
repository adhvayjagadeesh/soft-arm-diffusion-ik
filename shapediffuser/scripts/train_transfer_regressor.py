#!/usr/bin/env python3
"""Collect Elastica-labeled training data and fit a TransferRegressor.

For each of --n_targets training targets, samples K candidates from the
already-trained (frozen) diffusion checkpoint, evaluates each under Elastica,
and regresses predicted transfer error from (candidate, target). This
regressor is later used as a guidance signal during diffusion sampling
(GaussianDiffusion.sample_with_transfer_guidance in models.py) - steering the
generative process itself toward Elastica-compatible solutions, not just
reranking/correcting already-sampled candidates after the fact.

    python scripts/train_transfer_regressor.py --n_targets 25 --k 16
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shapediffuser import PCCArm, BabblingDataset, TransferRegressor  # noqa: E402
from shapediffuser.elastica_arm import ElasticaArm  # noqa: E402
from shapediffuser.metrics import sample_reachable_targets  # noqa: E402
from evaluate import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--n_targets", type=int, default=25)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=8888,
                    help="distinct from other studies' target seeds to avoid overlap")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--out", default="checkpoints/transfer_regressor.pt")
    ap.add_argument("--data_out", default="transfer_regressor_train_data.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ev = cfg["eval"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pcc = PCCArm(**cfg["arm"])
    elastica = ElasticaArm(
        n_segments=cfg["arm"]["n_segments"],
        seg_length=cfg["arm"]["seg_length"],
        base_radius=cfg["arm"]["rod_radius"],
    )

    model, norm, ck = load_model(os.path.join(args.ckpt_dir, "diffusion.pt"), device)
    targets = sample_reachable_targets(pcc, args.n_targets, seed=args.seed)

    all_q_scaled, all_cond, all_err = [], [], []
    t_start = time.time()
    for ti, t in enumerate(targets):
        cond = norm.encode(t.unsqueeze(0).to(device))
        with torch.no_grad():
            q_scaled = model.sample(cond, n_samples=args.k,
                                    steps=ev["ddim_steps"], guidance=ev["guidance"])[0]
        pressures = BabblingDataset.q_to_pressures(q_scaled).cpu()

        elastica_tip = elastica.forward(pressures)["tip"]
        err_mm = (elastica_tip - t).norm(dim=-1).numpy() * 1000

        all_q_scaled.append(q_scaled.cpu().numpy())
        all_cond.append(cond.cpu().numpy().repeat(args.k, axis=0))
        all_err.append(err_mm)
        print(f"target {ti + 1}/{len(targets)}: mean elastica err "
              f"{err_mm.mean():.1f}mm (min {err_mm.min():.1f}mm)", flush=True)

    q_scaled_arr = np.concatenate(all_q_scaled)
    cond_arr = np.concatenate(all_cond)
    err_arr = np.concatenate(all_err)
    print(f"\nCollected {len(err_arr)} labeled points in {time.time() - t_start:.0f}s")

    with open(args.data_out, "w") as f:
        json.dump({"q_scaled": q_scaled_arr.tolist(), "cond": cond_arr.tolist(),
                  "err_mm": err_arr.tolist(), "n_targets": args.n_targets, "k": args.k}, f)

    q_t = torch.as_tensor(q_scaled_arr, dtype=torch.float32, device=device)
    cond_t = torch.as_tensor(cond_arr, dtype=torch.float32, device=device)
    err_t = torch.as_tensor(err_arr / 100.0, dtype=torch.float32, device=device)  # scale for stability

    regressor = TransferRegressor(q_dim=q_t.shape[1], cond_dim=cond_t.shape[1]).to(device)
    opt = torch.optim.AdamW(regressor.parameters(), lr=1e-3, weight_decay=1e-5)
    for epoch in range(args.epochs):
        opt.zero_grad()
        pred = regressor(q_t, cond_t)
        loss = F.mse_loss(pred, err_t)
        loss.backward()
        opt.step()
        if (epoch + 1) % 50 == 0:
            print(f"regressor epoch {epoch + 1}/{args.epochs} mse {loss.item():.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model": regressor.state_dict(), "q_dim": q_t.shape[1],
               "cond_dim": cond_t.shape[1], "err_scale": 100.0}, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
