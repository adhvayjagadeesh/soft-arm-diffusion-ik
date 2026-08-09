#!/usr/bin/env python3
"""Fit a PCC model to the physical arm.

This is the step between "the arm moves" and "a policy can be trained for it".
It produces the *source model* the diffusion policy is trained on - and the
pairing that makes the hardware experiment sharp. The source model here is not
some unrelated alternative dynamics, as PCC-vs-Cosserat was in simulation; it
is the best constant-curvature description of this arm that the data supports.
Its residual is reported so a reader can judge how good a source model the
policy was actually given.

Two stages:

  1. Capstan radius, per servo. Command a known rotation, measure the tendon
     length that paid out. This is a *direct* measurement, not a fit.
  2. Curvature gain, per section. Sweep actuations, measure disc positions,
     and least-squares fit the gain that best explains the observed shapes.

    python calibrate_arm.py --measure-spool     # stage 1, interactive
    python calibrate_arm.py --sweep --n 60      # collect stage-2 data
    python calibrate_arm.py --fit               # fit and report residuals
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

try:
    import torch
    from scipy.optimize import least_squares
except ImportError as e:
    sys.exit(f"missing dependency: {e}")

sys.path.insert(0, __file__.rsplit("/", 1)[0])
sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/src")

from real_arm import RealArm, RealArmError, DISC_MARKER_IDS      # noqa: E402
from servo_bus import ServoBus, UNITS_PER_REV                     # noqa: E402
from shapediffuser import PCCArm                                  # noqa: E402

SWEEP_FILE = "calib_sweep.npz"
SPOOL_FILE = "spool_travel.json"
FIT_FILE = "arm_calibration.json"

SEG_LEN_M = 0.1524          # 6 in per section (3 discs at 2 in pitch)
N_SEGMENTS = 2


# --------------------------------------------------------------------------- #
def measure_spool():
    """Stage 1. Direct measurement beats any amount of fitting."""
    print(__doc__.split("Two stages:")[0])
    print("For ONE servo with cable wound in a single layer:\n"
          "  - mark the cable where it leaves the capstan\n"
          "  - the script commands exactly one revolution\n"
          "  - caliper how much cable paid out\n")
    bus = ServoBus().connect()
    try:
        bus.configure()
        bus.torque(True)
        dxl = bus.ids[0]
        start = bus.read_positions()
        input(f"servo {dxl} will turn one revolution. Mark the cable, then Enter...")
        bus.move_and_settle({**start, dxl: start[dxl] + UNITS_PER_REV}, settle=1.0)
        payout = float(input("cable paid out, in mm: "))
        bus.move_and_settle(start, settle=1.0)
    finally:
        bus.torque(False)
        bus.close()

    circumference_m = payout / 1000.0
    radius_m = circumference_m / (2 * np.pi)
    print(f"\n  payout per revolution : {payout:.2f} mm")
    print(f"  effective capstan radius: {radius_m*1000:.3f} mm")

    # how many revolutions to reach a useful bend on the outer tendon circle
    r_outer = 0.550 * 0.0254
    for kappa, label in ((6.0, "gentle, ~50 deg"), (12.0, "strong, ~105 deg")):
        travel = kappa * r_outer * SEG_LEN_M
        print(f"  {label}: needs {travel*1000:.1f} mm = "
              f"{travel/circumference_m:.2f} rev")
    travel_rev = float(input("\ntravel_rev to use (revolutions for q=1): "))
    with open(SPOOL_FILE, "w") as f:
        json.dump({"payout_mm_per_rev": payout,
                   "capstan_radius_mm": radius_m * 1000,
                   "travel_rev": travel_rev}, f, indent=2)
    print(f"wrote {SPOOL_FILE}")
    if travel_rev > 1.0:
        print("NOTE: >1 revolution confirms extended-position mode is required.")


# --------------------------------------------------------------------------- #
def sweep(n: int, seed: int = 0):
    """Stage 2 data: random actuations, measured disc positions."""
    rng = np.random.default_rng(seed)
    Q = rng.uniform(0.0, 0.7, size=(n, 6))
    rows_q, rows_pts, failed = [], [], 0
    with RealArm() as arm:
        arm.home()
        for k, q in enumerate(Q, 1):
            try:
                out = arm.forward(q)
            except RealArmError as e:
                failed += 1
                print(f"  {k}/{n} SKIP ({e})")
                continue
            rows_q.append(q)
            rows_pts.append(out["backbone"][0].numpy())
            tip = out["tip"][0].numpy() * 1000
            print(f"  {k}/{n}  tip [{tip[0]:7.1f} {tip[1]:7.1f} {tip[2]:7.1f}] mm")
        arm.home()
    np.savez(SWEEP_FILE, q=np.stack(rows_q), pts=np.stack(rows_pts))
    print(f"\nwrote {SWEEP_FILE}: {len(rows_q)} usable of {n} ({failed} dropped)")


# --------------------------------------------------------------------------- #
def fit():
    """Least-squares fit of per-section curvature gain."""
    z = np.load(SWEEP_FILE)
    q, pts = z["q"], z["pts"]          # pts: (N, 1+len(DISC_MARKER_IDS), 3)
    n = len(q)
    meas = pts[:, 1:, :]               # drop the base point at the origin
    n_disc = meas.shape[1]
    print(f"fitting on {n} samples, {n_disc} discs each")

    def residual(theta):
        g = np.abs(theta[:N_SEGMENTS])
        pcc = PCCArm(n_segments=N_SEGMENTS, seg_length=SEG_LEN_M,
                     curvature_gain=1.0,
                     points_per_seg=max(1, n_disc // N_SEGMENTS))
        qs = torch.as_tensor(q, dtype=torch.float32).clone()
        for s in range(N_SEGMENTS):
            qs[:, 3*s:3*s+3] *= g[s]
        bb = pcc.forward(qs)["backbone"].numpy()[:, 1:, :]
        m = min(bb.shape[1], n_disc)
        return (bb[:, :m, :] - meas[:, :m, :]).reshape(-1)

    # diff_step is NOT optional. PCCArm computes in float32, and scipy's
    # default finite-difference step (~1e-8 relative) changes the output by
    # less than float32 can represent, so the Jacobian comes back identically
    # zero and the optimiser returns the initial guess after one evaluation -
    # silently, reporting success. Verified on synthetic data with known
    # gains: default step recovers nothing (error 3.0 on both sections),
    # 1e-3 recovers 22.055/28.123 against a true 22.0/28.0 under 2 mm noise.
    theta0 = np.array([25.0] * N_SEGMENTS)
    sol = least_squares(residual, theta0, method="lm",
                        diff_step=1e-3, max_nfev=600)
    gains = np.abs(sol.x[:N_SEGMENTS])
    if np.allclose(gains, theta0, rtol=1e-6):
        print("WARNING: gains identical to the initial guess - the optimiser "
              "almost certainly never moved. Check that the sweep data varies.")

    r = residual(sol.x).reshape(n, -1, 3)
    per_pt = np.linalg.norm(r, axis=-1) * 1000
    print(f"\nfitted curvature gains: " +
          "  ".join(f"section {s+1}: {g:.3f}" for s, g in enumerate(gains)))
    print(f"\nresidual (measured vs fitted PCC), mm:")
    print(f"  RMS      {np.sqrt((per_pt**2).mean()):8.2f}")
    print(f"  mean     {per_pt.mean():8.2f}")
    print(f"  median   {np.median(per_pt):8.2f}")
    print(f"  90th pct {np.percentile(per_pt, 90):8.2f}")
    print(f"  max      {per_pt.max():8.2f}")

    with open(FIT_FILE, "w") as f:
        json.dump({"curvature_gain_per_section": gains.tolist(),
                   "seg_length_m": SEG_LEN_M, "n_samples": int(n),
                   "residual_rms_mm": float(np.sqrt((per_pt**2).mean())),
                   "residual_median_mm": float(np.median(per_pt)),
                   "residual_max_mm": float(per_pt.max())}, f, indent=2)
    print(f"\nwrote {FIT_FILE}")
    print("\nReport this residual in the paper. It is how a reader judges how")
    print("good a source model the policy was given - and it upper-bounds how")
    print("much of the eventual transfer gap is a bad fit rather than a real")
    print("limitation of the learned inverse map. Compare it against the vision")
    print("noise floor: a residual near the floor means PCC describes this arm")
    print("about as well as you can measure it.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--measure-spool", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.measure_spool:
        measure_spool()
    elif args.sweep:
        sweep(args.n, args.seed)
    elif args.fit:
        fit()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
