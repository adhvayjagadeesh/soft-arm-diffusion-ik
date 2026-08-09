#!/usr/bin/env python3
"""Dress rehearsal for the hardware experiments, against MockArm.

Runs the same sequence transfer_study.py runs - sample K candidates from a
trained policy, execute each on the target-domain arm, score best-of-K - but
with MockArm standing in for RealArm. The numbers are meaningless. The point
is that every shape, dtype, unit and retry path is exercised now, so that the
first time real hardware is involved, nothing fails for a reason that had
nothing to do with the hardware.

    python rehearse_pipeline.py --ckpt ../checkpoints_morph2seg --n 8 --k 8
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

HERE = __file__.rsplit("/", 1)[0]
ROOT = __file__.rsplit("/", 2)[0]
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT + "/src")
sys.path.insert(0, ROOT + "/scripts")

from mock_arm import MockArm                                        # noqa: E402
from evaluate import load_model, make_sampler                       # noqa: E402
from shapediffuser.metrics import sample_reachable_targets          # noqa: E402

MAX_RETRIES = 8


def measure_with_retry(arm, q):
    """Detection can fail. Everything downstream must survive that."""
    for _ in range(MAX_RETRIES):
        try:
            return arm.forward(q)
        except RuntimeError:
            continue
    raise RuntimeError("target abandoned after repeated detection failures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=ROOT + "/checkpoints_morph2seg")
    ap.add_argument("--n", type=int, default=8, help="targets")
    ap.add_argument("--k", type=int, default=8, help="candidates per target")
    ap.add_argument("--tol-mm", type=float, default=7.6, help="2% of 15 in arm")
    args = ap.parse_args()

    arm = MockArm()
    model, norm, ck = load_model(args.ckpt + "/diffusion.pt", "cpu")
    print(f"checkpoint q_dim={ck['q_dim']}  arm q_dim={arm.q_dim}")
    if ck["q_dim"] != arm.q_dim:
        sys.exit(
            f"\nDIMENSION MISMATCH: this policy emits {ck['q_dim']}-D actuations but the\n"
            f"arm takes {arm.q_dim}. The headline checkpoints were trained on the\n"
            "4-segment simulated arm (12 tendons); the physical build has 2 sections\n"
            "(6 tendons). Use checkpoints_morph2seg, or retrain on the fitted PCC\n"
            "model of the real arm as the paper's method specifies.")

    sampler = make_sampler(model, norm, "cpu", ck.get("ddim_steps", 20), 1.5)

    # Matching q_dim is necessary but nowhere near sufficient. The conditioning
    # normaliser carries the workspace the policy was trained on; if that
    # workspace does not overlap this arm's, every target is out of
    # distribution and the policy returns confident nonsense - with no error
    # anywhere, which is exactly how it would waste a hardware session.
    trained_r = float(np.linalg.norm(norm.mean.numpy()))
    arm_r = float(arm.source_forward(np.full((1, arm.q_dim), 0.5))["tip"].norm())
    if not (0.5 < trained_r / max(arm_r, 1e-6) < 2.0):
        print(f"\n*** WORKSPACE MISMATCH ***")
        print(f"  policy trained around |target| ~ {trained_r*1000:.0f} mm")
        print(f"  this arm reaches              ~ {arm_r*1000:.0f} mm")
        print("  The policy's inverse map is for a differently sized arm, so every")
        print("  target here is out of distribution. Retrain on the fitted PCC model")
        print("  of the real arm - which is what the paper's method specifies, and")
        print("  why paper-1 checkpoints are a starting point at best.\n")

    # targets must be reachable in the SOURCE model - the same convention the
    # simulation study used, so the policy is never asked for the impossible
    src_arm = arm._arm
    qs_rand = torch.rand(args.n, arm.q_dim) * arm.source_gain
    targets = src_arm.forward(qs_rand)["tip"]
    print(f"{args.n} targets, {args.k} candidates each\n")

    src_err, tgt_err, abandoned = [], [], 0
    t0 = time.time()
    for i, t in enumerate(targets, 1):
        cand = sampler(t.unsqueeze(0), args.k)[0]           # (k, q_dim) in [0,1]

        s = arm.source_forward(cand)["tip"]
        e_src = (s - t).norm(dim=-1).min().item() * 1000

        try:
            m = measure_with_retry(arm, cand)["tip"]
        except RuntimeError:
            abandoned += 1
            print(f"  {i:>2}: source {e_src:7.2f} mm   ABANDONED")
            continue
        e_tgt = (m - t).norm(dim=-1).min().item() * 1000

        src_err.append(e_src); tgt_err.append(e_tgt)
        print(f"  {i:>2}: source {e_src:7.2f} mm   measured {e_tgt:7.2f} mm")

    src_err, tgt_err = np.array(src_err), np.array(tgt_err)
    print(f"\n{'':16}{'source':>10}{'measured':>11}")
    print(f"{'best-of-K err':16}{src_err.mean():>9.2f}mm{tgt_err.mean():>10.2f}mm")
    print(f"{'success @tol':16}{(src_err<args.tol_mm).mean():>10.0%}{(tgt_err<args.tol_mm).mean():>11.0%}")
    print(f"\nabandoned targets: {abandoned}   wall clock: {time.time()-t0:.1f}s")
    print("\nPipeline exercised end to end: sampling, both arms, retry on dropped\n"
          "detections, best-of-K scoring. Numbers are mock; the plumbing is real.")


if __name__ == "__main__":
    main()
