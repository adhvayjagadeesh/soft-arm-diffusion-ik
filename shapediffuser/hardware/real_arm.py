#!/usr/bin/env python3
"""The physical arm, behind the same interface as PCCArm and ElasticaArm.

    forward(q) -> {"backbone": (B, P, 3), "tip": (B, 3)}

That signature is the whole point. Every script written for the simulation
study - evaluate.py, metrics.py, the diffusion sampler, transfer_study.py -
calls exactly this and nothing else, so swapping ElasticaArm for RealArm turns
the sim-to-sim experiment into sim-to-real with no changes downstream.

Frames
------
Poses come out of the camera in the camera frame, which is arbitrary and moves
if anything is bumped. A reference marker bonded to the base plate defines the
arm's base frame instead, and every disc position is expressed relative to it.
Bump the tripod and the numbers stay valid; the reference marker moves with the
arm, not with the camera.

    python real_arm.py --test          # one round trip, no policy involved
    python real_arm.py --repeatability # actuation repeatability floor
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

try:
    import cv2
    import torch
except ImportError as e:
    sys.exit(f"missing dependency: {e}")

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from servo_bus import ServoBus, ServoBusError, UNITS_PER_REV      # noqa: E402
from vision import Camera, MarkerTracker, load_calib              # noqa: E402

# --------------------------------------------------------------------------- #
REF_MARKER_ID = 9          # bonded flat to the base plate; defines the base frame
DISC_MARKER_IDS = [0, 1, 2, 3, 4, 5]        # V1..V6, bottom to tip
TIP_MARKER_ID = 5

SETTLE_S = 1.5             # fixed, deterministic - see the note in forward()
N_REPEATS = 3              # frames averaged per measurement
MAX_RETRIES = 6            # detection failures before giving up on a sample

TRAVEL_REV_FILE = "spool_travel.json"       # written by --measure-spool


class RealArmError(RuntimeError):
    pass


def _rt_from(pose) -> tuple:
    R, _ = cv2.Rodrigues(pose["rvec"].reshape(3, 1))
    return R, pose["t"].reshape(3)


class RealArm:
    """Physical tendon-driven continuum arm.

    q_dim matches the simulated 2-section arm: 3 tendons x 2 sections.
    """

    def __init__(self, travel_rev: float | None = None, settle: float = SETTLE_S,
                 n_repeats: int = N_REPEATS, marker_m: float | None = None):
        self.bus = ServoBus().connect()
        K, dist = load_calib()
        self.tracker = MarkerTracker(K, dist, **({"marker_m": marker_m}
                                                 if marker_m else {}))
        self.cam = Camera()
        self.settle = settle
        self.n_repeats = n_repeats
        self.q_dim = len(self.bus.ids)
        self.travel_rev = travel_rev if travel_rev is not None else self._load_travel()
        if not self.bus.home:
            raise RealArmError(
                "no servo_home.json - run `python servo_bus.py --set-home` with the "
                "tendons strung just taut and the arm straight.")

    def _load_travel(self) -> float:
        try:
            with open(TRAVEL_REV_FILE) as f:
                v = float(json.load(f)["travel_rev"])
            return v
        except Exception:
            raise RealArmError(
                f"no {TRAVEL_REV_FILE} - measure cable payout per servo revolution "
                "first (see --measure-spool), or pass travel_rev= explicitly. "
                "Guessing this number silently mis-scales every actuation.")

    # ---------------- measurement ---------------- #
    def _measure_once(self) -> dict | None:
        """One frame -> {marker_id: xyz in base frame}, or None if unusable."""
        poses = self.tracker.poses(self.cam.frame())
        if REF_MARKER_ID not in poses:
            return None
        R_ref, t_ref = _rt_from(poses[REF_MARKER_ID])
        out = {}
        for mid, p in poses.items():
            if mid == REF_MARKER_ID:
                continue
            _, t = _rt_from(p)
            out[mid] = R_ref.T @ (t - t_ref)        # camera frame -> base frame
        return out

    def measure(self) -> dict:
        """Average n_repeats frames. Averaging cuts random sensing noise; it
        does nothing for bias, which is why the noise-floor run reports both."""
        acc, used = {}, 0
        for _ in range(MAX_RETRIES * self.n_repeats):
            m = self._measure_once()
            if m is None or not all(i in m for i in DISC_MARKER_IDS):
                continue
            for k, v in m.items():
                acc.setdefault(k, []).append(v)
            used += 1
            if used >= self.n_repeats:
                break
        if used == 0:
            raise RealArmError(
                "no usable frame: need the reference marker and all disc markers "
                "visible at once. Check framing, lighting and occlusion.")
        return {k: np.mean(np.stack(v), axis=0) for k, v in acc.items()}

    # ---------------- actuation ---------------- #
    def _q_to_positions(self, q) -> dict:
        q = np.clip(np.asarray(q, dtype=float).reshape(-1), 0.0, 1.0)
        if q.size != self.q_dim:
            raise ValueError(f"expected q of length {self.q_dim}, got {q.size}")
        return {i: int(self.bus.home[i] + qi * self.travel_rev * UNITS_PER_REV)
                for i, qi in zip(self.bus.ids, q)}

    def forward(self, q) -> dict:
        """q: (B, q_dim) in [0,1] -> {"backbone": (B,P,3), "tip": (B,3)} metres.

        Serial and slow by nature: one physical actuation and settle per row.
        The settle is fixed rather than adaptive so that every sample is taken
        under an identical protocol - the same deterministic-snapshot
        convention used for the simulated target domain.
        """
        arr = q.detach().cpu().numpy() if torch.is_tensor(q) else np.asarray(q)
        if arr.ndim == 1:
            arr = arr[None, :]
        backbones, tips = [], []
        for row in arr:
            self.bus.move_and_settle(self._q_to_positions(row), settle=self.settle)
            m = self.measure()
            pts = np.stack([m[i] for i in DISC_MARKER_IDS])
            backbones.append(np.vstack([np.zeros(3), pts]))   # base + discs
            tips.append(m[TIP_MARKER_ID])
        bb = torch.as_tensor(np.stack(backbones), dtype=torch.float32)
        return {"backbone": bb, "tip": torch.as_tensor(np.stack(tips),
                                                       dtype=torch.float32)}

    def home(self):
        self.bus.configure()
        self.bus.torque(True)
        self.bus.go_home(settle=self.settle)

    def close(self):
        try:
            self.cam.release()
        finally:
            self.bus.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------- #
def repeatability(arm: RealArm, n_poses: int = 5, k: int = 4, seed: int = 0):
    """Actuation repeatability: same command, many times, in randomized order.

    Together with the vision noise floor this bounds how small a transfer gap
    may be interpreted. It folds in servo backlash, cable hysteresis and creep -
    everything the simulator gets exactly right for free.
    """
    rng = np.random.default_rng(seed)
    poses = rng.uniform(0.15, 0.55, size=(n_poses, arm.q_dim))
    order = [(i, r) for r in range(k) for i in range(n_poses)]
    rng.shuffle(order)

    tips = {i: [] for i in range(n_poses)}
    for n, (i, _) in enumerate(order, 1):
        arm.home()
        out = arm.forward(poses[i])
        tips[i].append(out["tip"][0].numpy())
        print(f"  {n}/{len(order)}  pose {i}  tip {tips[i][-1]*1000}")

    print(f"\n{'pose':>5} {'spread mm':>11} {'max dev mm':>12}")
    spreads = []
    for i in range(n_poses):
        t = np.stack(tips[i]) * 1000
        c = t.mean(axis=0)
        dev = np.linalg.norm(t - c, axis=1)
        spreads.append(dev.max())
        print(f"{i:>5} {dev.std():>11.2f} {dev.max():>12.2f}")
    print(f"\nactuation repeatability floor: {max(spreads):.2f} mm (worst-case deviation)")
    with open("actuation_repeatability.json", "w") as f:
        json.dump({"n_poses": n_poses, "k": k,
                   "worst_deviation_mm": float(max(spreads))}, f, indent=2)
    print("wrote actuation_repeatability.json")


def selftest(arm: RealArm):
    print("homing...")
    arm.home()
    m = arm.measure()
    print(f"markers seen in base frame: {sorted(m.keys())}")
    for mid in sorted(m):
        print(f"  id {mid}: {np.round(m[mid]*1000, 1)} mm")

    q = np.full(arm.q_dim, 0.3)
    print(f"\nforward(q={q.round(2)}) ...")
    t0 = time.time()
    out = arm.forward(q)
    print(f"  took {time.time()-t0:.1f} s")
    print(f"  backbone shape {tuple(out['backbone'].shape)}  "
          f"tip {np.round(out['tip'][0].numpy()*1000, 1)} mm")
    print("\nInterface matches PCCArm.forward - downstream scripts need no changes.")
    arm.home()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--repeatability", action="store_true")
    ap.add_argument("--n-poses", type=int, default=5)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--travel-rev", type=float, default=None)
    args = ap.parse_args()

    try:
        with RealArm(travel_rev=args.travel_rev) as arm:
            if args.repeatability:
                repeatability(arm, args.n_poses, args.k)
            else:
                selftest(arm)
    except (RealArmError, ServoBusError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
