"""Evaluation metrics for ShapeDiffuser and baselines.

Covers the four result pillars of the paper:
  1. Accuracy      — tip error, whole-body chamfer distance.
  2. Multimodality — ground-truth mode enumeration (dense sweep + DBSCAN),
                     mode recall, solution diversity.
  3. Downstream    — obstacle-conditioned solution selection (see evaluate.py).
  4. Efficiency    — wall-clock sampling time (timed in evaluate.py).
"""

from __future__ import annotations

import numpy as np
import torch

from .pcc_arm import PCCArm

try:
    from sklearn.cluster import DBSCAN

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False


# --------------------------------------------------------------------------- #
def tip_error(arm: PCCArm, q: torch.Tensor, target_tip: torch.Tensor) -> torch.Tensor:
    """q: (..., q_dim) pressures in [0,1]; target_tip: (..., 3). Returns (...,) L2."""
    flat = q.reshape(-1, arm.q_dim)
    with torch.no_grad():
        tips = arm.forward(flat)["tip"]
    return (tips.view(*q.shape[:-1], 3) - target_tip).norm(dim=-1)


def chamfer(bb_a: torch.Tensor, bb_b: torch.Tensor) -> torch.Tensor:
    """Symmetric chamfer distance between backbones. (B, P, 3) x (B, P, 3) -> (B,)."""
    d = torch.cdist(bb_a, bb_b)  # (B, P, P)
    return 0.5 * (d.min(dim=-1).values.mean(dim=-1) + d.min(dim=-2).values.mean(dim=-1))


def diversity(q_success: torch.Tensor) -> float:
    """Mean pairwise L2 among successful solutions for one target. (n, q_dim)."""
    n = q_success.shape[0]
    if n < 2:
        return 0.0
    d = torch.cdist(q_success, q_success)
    return (d.sum() / (n * (n - 1))).item()


# --------------------------------------------------------------------------- #
def enumerate_modes(
    arm: PCCArm,
    target_tip: torch.Tensor,
    pool_size: int = 2_000_000,
    tol: float = 0.005,
    dbscan_eps: float = 0.35,
    min_samples: int = 10,
    batch: int = 65536,
    seed: int = 0,
) -> torch.Tensor:
    """Ground-truth IK mode enumeration by dense random sweep + clustering.

    Returns (n_modes, q_dim) cluster medoids of actuations whose tip lands
    within `tol` of the target. Because the arm and tolerance are ours to
    choose in simulation, this gives an honest reference solution set.
    """
    if not _HAS_SKLEARN:
        raise RuntimeError("mode enumeration requires scikit-learn")
    rng = np.random.default_rng(seed)
    kept = []
    tgt = target_tip.view(1, 3)
    for i in range(0, pool_size, batch):
        n = min(batch, pool_size - i)
        q = torch.as_tensor(rng.uniform(0, 1, size=(n, arm.q_dim)), dtype=torch.float32)
        with torch.no_grad():
            tips = arm.forward(q)["tip"]
        m = (tips - tgt).norm(dim=-1) < tol
        if m.any():
            kept.append(q[m])
    if not kept:
        return torch.empty(0, arm.q_dim)
    qs = torch.cat(kept).numpy()
    labels = DBSCAN(eps=dbscan_eps, min_samples=min_samples).fit(qs).labels_
    modes = []
    for lab in sorted(set(labels) - {-1}):
        cluster = qs[labels == lab]
        center = cluster.mean(axis=0)
        medoid = cluster[np.argmin(((cluster - center) ** 2).sum(axis=1))]
        modes.append(medoid)
    return torch.as_tensor(np.stack(modes), dtype=torch.float32) if modes else torch.empty(0, arm.q_dim)


def mode_recall(
    gt_modes: torch.Tensor, q_samples: torch.Tensor, success_mask: torch.Tensor,
    radius: float = 0.35,
) -> float:
    """Fraction of ground-truth modes that have at least one *successful*
    generated sample within `radius` in actuation space."""
    if gt_modes.shape[0] == 0:
        return float("nan")
    q_ok = q_samples[success_mask]
    if q_ok.shape[0] == 0:
        return 0.0
    d = torch.cdist(gt_modes, q_ok)  # (M, n_ok)
    return (d.min(dim=-1).values < radius).float().mean().item()


def sample_reachable_targets(
    arm: PCCArm, n_targets: int, seed: int = 123
) -> torch.Tensor:
    """Draw guaranteed-reachable tip targets by pushing random q through FK."""
    rng = np.random.default_rng(seed)
    q = torch.as_tensor(rng.uniform(0, 1, size=(n_targets, arm.q_dim)), dtype=torch.float32)
    with torch.no_grad():
        return arm.forward(q)["tip"]
