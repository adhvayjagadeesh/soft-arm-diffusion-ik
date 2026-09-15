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
    curvature_gain_range: tuple[float, float] | None = None,
    morph_ranges: dict | None = None,
) -> dict:
    """curvature_gain_range: if given, draws a fresh per-sample curvature_gain
    ~ Uniform(*curvature_gain_range) instead of using arm.curvature_gain,
    for domain-randomization training data. The model is not conditioned on
    the sampled gain, so it can't tell which one applies to a given
    sample - the point is to force actuations that work reasonably well
    across the whole range, not to teach the model the range itself.
    None (default) reproduces the exact prior fixed-gain behavior.

    morph_ranges: if given, {"seg_length": (lo, hi), "curvature_gain": (lo, hi)}.
    Every sample gets its OWN morphology - N segment lengths and N gains drawn
    independently - and that morphology is written to the dataset as "morph"
    (B, 2N) so a model can be conditioned on it. This is the amortized-IK
    regime, and it differs from curvature_gain_range in exactly one respect:
    the model is told which arm each sample came from. Paper 1 showed that
    randomizing without telling (DR) recovers nothing under model change; the
    question here is whether telling is what was missing.
    Mutually exclusive with curvature_gain_range."""
    rng = np.random.default_rng(seed)
    if morph_ranges is not None and curvature_gain_range is not None:
        raise ValueError("morph_ranges and curvature_gain_range are mutually exclusive")
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

    gains = None
    if curvature_gain_range is not None:
        gains = rng.uniform(curvature_gain_range[0], curvature_gain_range[1], size=n_samples)

    morph = None
    if morph_ranges is not None:
        N = arm.n_segments
        lo_L, hi_L = morph_ranges["seg_length"]
        lo_G, hi_G = morph_ranges["curvature_gain"]
        Ls = rng.uniform(lo_L, hi_L, size=(n_samples, N))
        Gs = rng.uniform(lo_G, hi_G, size=(n_samples, N))
        morph = np.concatenate([Ls, Gs], axis=1).astype(np.float32)  # same order as PCCArm.morph

    tips, shapes = [], []
    for i in range(0, n_samples, batch):
        qb = torch.as_tensor(q[i : i + batch], dtype=torch.float32)
        with torch.no_grad():
            if morph is not None:
                N = arm.n_segments
                mb = torch.as_tensor(morph[i : i + batch])
                out = arm.forward(qb, gain=mb[:, N:], lengths=mb[:, :N])
            elif gains is not None:
                gb = torch.as_tensor(gains[i : i + batch], dtype=torch.float32)
                out = arm.forward(qb, gain=gb)
            else:
                # don't pass gain= at all when unused, so arms without a
                # gain kwarg (e.g. ElasticaArm) work identically to PCCArm
                out = arm.forward(qb)
        bb = out["backbone"]  # (b, P, 3)
        idx = torch.linspace(0, bb.shape[1] - 1, shape_points).long()
        tips.append(out["tip"].cpu().numpy())
        shapes.append(bb[:, idx, :].reshape(qb.shape[0], -1).cpu().numpy())

    out = {
        "q": q.astype(np.float32),
        "tip": np.concatenate(tips).astype(np.float32),
        "shape": np.concatenate(shapes).astype(np.float32),
    }
    if morph is not None:
        out["morph"] = morph
    return out


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

    COND_TYPES = ("tip", "shape", "tip_morph", "shape_morph")

    def __init__(self, data: dict, cond_type: str = "tip", cond_norm: Normalizer | None = None):
        assert cond_type in self.COND_TYPES, f"cond_type must be one of {self.COND_TYPES}"
        self.q = torch.as_tensor(data["q"]) * 2.0 - 1.0
        cond = self.build_cond(data, cond_type)
        self.cond_norm = cond_norm or Normalizer.fit(cond)
        self.cond = self.cond_norm.encode(torch.as_tensor(cond))
        self.cond_type = cond_type

    @staticmethod
    def build_cond(data: dict, cond_type: str) -> np.ndarray:
        """The conditioning vector for a cond_type, from a dataset dict.

        "*_morph" appends the (2N,) morphology to the target, so an amortized
        model sees [what to reach, on which arm]. Kept here so evaluation code
        builds conditions the same way training did."""
        base, _, morph = cond_type.partition("_")
        cond = data[base]
        if morph == "morph":
            if "morph" not in data:
                raise KeyError(f"cond_type={cond_type} needs a 'morph' array; "
                               "generate the data with morph_ranges set")
            cond = np.concatenate([cond, data["morph"]], axis=1)
        return np.asarray(cond, dtype=np.float32)

    def __len__(self) -> int:
        return self.q.shape[0]

    def __getitem__(self, i: int):
        return self.q[i], self.cond[i]

    @staticmethod
    def q_to_pressures(q_scaled: torch.Tensor) -> torch.Tensor:
        return ((q_scaled + 1.0) * 0.5).clamp(0.0, 1.0)
