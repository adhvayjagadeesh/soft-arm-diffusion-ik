#!/usr/bin/env python3
"""Generate an Elastica-simulated dataset for the real-data-mixing experiment.

Uses the exact same data-generation logic as generate_data.py (uniform-random
actuations pushed through forward kinematics), but with ElasticaArm instead
of PCCArm - the same physics-simulated "high-fidelity" engine used
throughout tonight's transfer studies, not real hardware (see README's
honesty notes: no real-world data anywhere in this project).

Unbatched and ~4.1s/sample, so a large run (e.g. 10,000 samples ~ 11 hours)
is a long unattended job. This is checkpointed and resumable: the full set
of random actuations is pre-generated deterministically from --seed (cheap,
instant) so resuming after an interruption just means reloading the
checkpoint and continuing from the next un-simulated index, not restarting.

    python scripts/generate_elastica_dataset.py --n_samples 10000 --checkpoint_every 250
    # if interrupted, just rerun the same command - it resumes automatically
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shapediffuser.elastica_arm import ElasticaArm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n_samples", type=int, default=10000)
    ap.add_argument("--shape_points", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7777,
                    help="distinct from other studies' target seeds")
    ap.add_argument("--checkpoint_every", type=int, default=250)
    ap.add_argument("--chunk", type=int, default=10,
                    help="samples per Elastica call; doesn't affect speed (unbatched "
                         "internally) but controls checkpoint granularity")
    ap.add_argument("--out", default="elastica_finetune_data.npz")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    arm = ElasticaArm(
        n_segments=cfg["arm"]["n_segments"],
        seg_length=cfg["arm"]["seg_length"],
        base_radius=cfg["arm"]["rod_radius"],
    )

    # Pre-generate the full actuation set deterministically so resuming after
    # an interruption reproduces the exact same q's, just continuing where
    # simulation left off.
    rng = np.random.default_rng(args.seed)
    q_all = rng.uniform(0.0, 1.0, size=(args.n_samples, arm.q_dim)).astype(np.float32)

    check_n = min(len(q_all), 100)
    ckpt_path = args.out + ".ckpt.npz"
    if os.path.exists(ckpt_path):
        ckpt = np.load(ckpt_path)
        start_done = int(ckpt["done"])
        tips = list(ckpt["tips"])
        shapes = list(ckpt["shapes"])
        assert np.allclose(ckpt["q_all_check"], q_all[:check_n]), \
            "checkpoint's actuations don't match this seed's - refusing to resume with mismatched data"
        print(f"Resuming from checkpoint: {start_done}/{args.n_samples} already done", flush=True)
    else:
        start_done, tips, shapes = 0, [], []

    done = start_done
    t_start = time.time()
    for i in range(done, args.n_samples, args.chunk):
        qb = torch.as_tensor(q_all[i : i + args.chunk], dtype=torch.float32)
        out = arm.forward(qb)
        bb = out["backbone"]
        idx = torch.linspace(0, bb.shape[1] - 1, args.shape_points).long()
        tips.extend(out["tip"].numpy())
        shapes.extend(bb[:, idx, :].reshape(qb.shape[0], -1).numpy())
        done = i + qb.shape[0]

        if done % args.checkpoint_every < args.chunk or done >= args.n_samples:
            elapsed = time.time() - t_start
            rate_per_sample = elapsed / max(1, done - start_done)
            eta_s = (args.n_samples - done) * rate_per_sample
            print(f"{done}/{args.n_samples} done ({elapsed:.0f}s elapsed this run, "
                  f"~{eta_s / 60:.0f}min remaining)", flush=True)
            np.savez(ckpt_path, done=done, tips=np.stack(tips), shapes=np.stack(shapes),
                    q_all_check=q_all[:check_n])

    q_used = q_all[:done]
    np.savez_compressed(args.out, q=q_used, tip=np.stack(tips).astype(np.float32),
                        shape=np.stack(shapes).astype(np.float32))
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"\nWrote {args.out} ({done} samples, {time.time() - t_start:.0f}s total)")


if __name__ == "__main__":
    main()
