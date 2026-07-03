"""Motor-babbling data generation and dataset utilities.

Two exploration modes:
  * "uniform": i.i.d. uniform pressures — best workspace coverage.
  * "babble":  bounded random walk |q_{i+1} - q_i| < E (matches the classic
               motor-babbling protocol from the lecture material).

Each sample stores actuation q, tip position, and a downsampled backbone so the
model can be conditioned on either the tip target ("tip") or a whole-body shape
target ("shape").
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .pcc_arm import PCCArm


def generate_dataset(
    arm: PCCArm,
    n_samples: int,
    mode: str = "uniform",
    babble_step: float = 0.08,
    shape_points: int = 8,
    batch: int = 8192,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    if mode == "uniform":
        q = rng.uniform(0.0, 1.0, size=(n_samples, arm.q_dim))
    elif mode == "babble":
        q = np.empty((n_samples, arm.q_dim))
        q[0] = rng.uniform(0.0, 1.0, size=arm.q_dim)
        for i in range(1, n_samples):
            step = rng.uniform(-babble_step, babble_step, size=arm.q_dim)
            q[i] = np.clip(q[i - 1] + step, 0.0, 1.0)
    else:
        raise ValueError(f"unknown exploration mode: {mode}")

    tips, shapes = [], []
    for i in range(0, n_samples, batch):
        qb = torch.as_tensor(q[i : i + batch], dtype=torch.float32)
        with torch.no_grad():
            out = arm.forward(qb)
        bb = out["backbone"]  # (b, P, 3)
        idx = torch.linspace(0, bb.shape[1] - 1, shape_points).long()
        tips.append(out["tip"].cpu().numpy())
        shapes.append(bb[:, idx, :].reshape(qb.shape[0], -1).cpu().numpy())

    return {
        "q": q.astype(np.float32),
        "tip": np.concatenate(tips).astype(np.float32),
        "shape": np.concatenate(shapes).astype(np.float32),
    }


class Normalizer:
    """Mean/std normalizer with save/load round-trip."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(np.maximum(std, 1e-8), dtype=torch.float32)

    @classmethod
    def fit(cls, x: np.ndarray) -> "Normalizer":
        return cls(x.mean(axis=0), x.std(axis=0))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def state_dict(self) -> dict:
        return {"mean": self.mean.numpy(), "std": self.std.numpy()}

    @classmethod
    def from_state(cls, d: dict) -> "Normalizer":
        return cls(d["mean"], d["std"])


class BabblingDataset(Dataset):
    """Yields (q_scaled, cond_normalized).

    Actuations are affinely mapped [0,1] -> [-1,1] (natural for diffusion);
    conditions are z-scored with a fitted Normalizer.
    """

    def __init__(self, data: dict, cond_type: str = "tip", cond_norm: Normalizer | None = None):
        assert cond_type in ("tip", "shape")
        self.q = torch.as_tensor(data["q"]) * 2.0 - 1.0
        cond = data[cond_type]
        self.cond_norm = cond_norm or Normalizer.fit(cond)
        self.cond = self.cond_norm.encode(torch.as_tensor(cond))
        self.cond_type = cond_type

    def __len__(self) -> int:
        return self.q.shape[0]

    def __getitem__(self, i: int):
        return self.q[i], self.cond[i]

    @staticmethod
    def q_to_pressures(q_scaled: torch.Tensor) -> torch.Tensor:
        return ((q_scaled + 1.0) * 0.5).clamp(0.0, 1.0)
