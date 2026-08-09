#!/usr/bin/env python3
"""A stand-in for RealArm that needs no hardware.

Exists so the entire downstream pipeline can be exercised and debugged before
the physical arm is assembled. Hardware time is expensive and serial - roughly
3 s per actuation - so discovering a shape mismatch or a units error while the
arm is sitting on the bench is the worst possible time to discover it.

MockArm matches RealArm's contract exactly:
  * same forward(q) -> {"backbone": (B, 7, 3), "tip": (B, 3)} in METRES
  * same 7-point backbone (base + 6 disc markers), NOT PCCArm's dense backbone
  * same q_dim = 6 (two sections x three tendons)
  * same [0,1] normalised actuation

and then adds the error sources a simulator gives you for free:
  * a different curvature gain from the source model  -> model mismatch
  * directional hysteresis                            -> tendon backlash
  * Gaussian position noise                           -> marker sensing noise
  * occasional dropped measurements                   -> detection failures

    python mock_arm.py --demo
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/src")

from shapediffuser import PCCArm                                   # noqa: E402

N_SECTIONS = 2
SEG_LEN_M = 0.1524            # 6 in per section, matching the build
DISCS_PER_SECTION = 3


class MockArm:
    """Behaves like the physical arm. Every default here is a guess about
    hardware that has not been measured yet - the point is exercising shapes
    and code paths, not predicting numbers."""

    def __init__(self, source_gain: float = 25.0, true_gain: float = 21.5,
                 sensing_noise_mm: float = 1.5, backlash_mm: float = 2.0,
                 dropout: float = 0.02, seed: int = 0):
        self.q_dim = 3 * N_SECTIONS
        self.n_discs = DISCS_PER_SECTION * N_SECTIONS
        self.source_gain = source_gain
        self.true_gain = true_gain
        self.sensing_noise_m = sensing_noise_mm / 1000.0
        self.backlash_m = backlash_mm / 1000.0
        self.dropout = dropout
        self.rng = np.random.default_rng(seed)
        self._last_q = None
        self._arm = PCCArm(n_segments=N_SECTIONS, seg_length=SEG_LEN_M,
                           curvature_gain=1.0, points_per_seg=DISCS_PER_SECTION)

    # -- the model the policy was trained on (what the source model predicts) --
    def source_forward(self, q) -> dict:
        return self._shape(q, self.source_gain, noisy=False)

    def _shape(self, q, gain, noisy):
        arr = q.detach().cpu().numpy() if torch.is_tensor(q) else np.asarray(q, float)
        if arr.ndim == 1:
            arr = arr[None, :]
        qs = torch.as_tensor(np.clip(arr, 0.0, 1.0), dtype=torch.float32).clone()
        qs = qs * gain
        bb = self._arm.forward(qs)["backbone"].numpy()      # (B, 1+n_discs, 3)
        if noisy:
            bb = bb + self.rng.normal(0, self.sensing_noise_m, bb.shape)
            bb[:, 0, :] = 0.0                               # base is the origin
        return {"backbone": torch.as_tensor(bb, dtype=torch.float32),
                "tip": torch.as_tensor(bb[:, -1, :], dtype=torch.float32)}

    # ------------------------- the RealArm contract ------------------------- #
    def forward(self, q) -> dict:
        arr = q.detach().cpu().numpy() if torch.is_tensor(q) else np.asarray(q, float)
        if arr.ndim == 1:
            arr = arr[None, :]

        # backlash: a tendon reverses direction and takes up slack first
        adj = np.empty_like(arr, dtype=float)
        for b, row in enumerate(arr):
            if self._last_q is None:
                adj[b] = row
            else:
                d = np.sign(row - self._last_q)
                adj[b] = row - d * (self.backlash_m / (self.true_gain * SEG_LEN_M))
            self._last_q = row.copy()

        out = self._shape(adj, self.true_gain, noisy=True)

        # dropped detections: the caller must cope, exactly as with real markers
        if self.dropout > 0 and self.rng.random() < self.dropout:
            raise RuntimeError("mock: marker detection failed (simulated dropout)")
        return out

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def demo():
    arm = MockArm()
    print(f"q_dim={arm.q_dim}  discs={arm.n_discs}")

    q = np.full((4, arm.q_dim), 0.35)
    q[1, :3] = 0.6
    q[2, 3:] = 0.6
    q[3] = 0.15

    src = arm.source_forward(q)
    print(f"\nsource_forward: backbone {tuple(src['backbone'].shape)}  "
          f"tip {tuple(src['tip'].shape)}")

    got = None
    for _ in range(10):
        try:
            got = arm.forward(q)
            break
        except RuntimeError as e:
            print(f"  retry: {e}")
    print(f"forward       : backbone {tuple(got['backbone'].shape)}  "
          f"tip {tuple(got['tip'].shape)}")

    d = (got["tip"] - src["tip"]).norm(dim=-1) * 1000
    print(f"\nsource-vs-measured tip error (mm): {np.round(d.numpy(), 1)}")
    print(f"mean {d.mean():.1f} mm - this is the mock's stand-in for the")
    print("sim-to-real gap, and is a guess, not a prediction.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.parse_args()
    demo()
