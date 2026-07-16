"""High-fidelity Cosserat-rod arm via PyElastica.

This mirrors the PCCArm interface (forward(q) -> {"backbone", "tip"}) but runs a
PyElastica simulation per sample: the arm is one Cosserat rod split into
`n_segments` regions, each receiving a constant internal couple derived from
the 3 chamber pressures, integrated with linear damping from rest for a fixed
`settle_time` horizon. NOTE (robustness check, see paper): the damped rod
still oscillates at (and well beyond) the default 1.5 s horizon, so the
returned state is a fixed, deterministic dynamic snapshot - the same protocol
for every sample - not a static equilibrium.

Notes:
  * ~1000x slower than PCCArm; use it to (a) generate a smaller high-fidelity
    test set, (b) run the "PCC-trained -> Cosserat-evaluated" transfer study,
    which reviewers like because it probes model-mismatch robustness.
  * Verified against pyelastica 1.0.0 (uniform pressures stay straight,
    asymmetric pressures bend as expected); the only fix needed from the
    original scaffold was progress_bar=False on ea.integrate.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    import elastica as ea

    _HAS_ELASTICA = True
except ImportError:  # pragma: no cover
    _HAS_ELASTICA = False


class ElasticaArm:
    def __init__(
        self,
        n_segments: int = 4,
        seg_length: float = 0.06,
        n_elem_per_seg: int = 12,
        base_radius: float = 0.012,
        youngs_modulus: float = 1.0e5,  # Ecoflex-scale, ~10^5 Pa
        density: float = 1000.0,
        torque_gain: float = 5.0e-3,
        settle_time: float = 1.5,
        dt: float = 2.0e-5,
    ):
        if not _HAS_ELASTICA:
            raise ImportError("pip install pyelastica to use ElasticaArm")
        self.n_segments = n_segments
        self.seg_length = seg_length
        self.n_elem = n_elem_per_seg * n_segments
        self.n_elem_per_seg = n_elem_per_seg
        self.base_radius = base_radius
        self.youngs_modulus = youngs_modulus
        self.density = density
        self.torque_gain = torque_gain
        self.settle_time = settle_time
        self.dt = dt
        self.q_dim = 3 * n_segments

    # ------------------------------------------------------------------ #
    def _simulate_one(self, q: np.ndarray) -> np.ndarray:
        class Sim(
            ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping
        ):
            pass

        sim = Sim()
        rod = ea.CosseratRod.straight_rod(
            n_elements=self.n_elem,
            start=np.zeros(3),
            direction=np.array([0.0, 0.0, 1.0]),
            normal=np.array([1.0, 0.0, 0.0]),
            base_length=self.seg_length * self.n_segments,
            base_radius=self.base_radius,
            density=self.density,
            youngs_modulus=self.youngs_modulus,
            shear_modulus=self.youngs_modulus / 3.0,
        )
        sim.append(rod)
        sim.constrain(rod).using(
            ea.OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
        )
        sim.dampen(rod).using(
            ea.AnalyticalLinearDamper, damping_constant=2.0, time_step=self.dt
        )

        # Chamber pressures -> per-segment internal couple (local x/y bending).
        p = q.reshape(self.n_segments, 3)
        for s in range(self.n_segments):
            tx = self.torque_gain * (p[s, 0] - 0.5 * p[s, 1] - 0.5 * p[s, 2])
            ty = self.torque_gain * 0.8660254 * (p[s, 1] - p[s, 2])
            i0, i1 = s * self.n_elem_per_seg, (s + 1) * self.n_elem_per_seg
            sim.add_forcing_to(rod).using(
                _SegmentCouple, torque=np.array([tx, ty, 0.0]), start=i0, end=i1
            )

        sim.finalize()
        ts = ea.PositionVerlet()
        ea.integrate(ts, sim, final_time=self.settle_time,
                     n_steps=int(self.settle_time / self.dt), progress_bar=False)
        return rod.position_collection.T.copy()  # (n_nodes, 3)

    def forward(self, q: torch.Tensor) -> dict:
        """q: (B, 3N) pressures in [0,1]. NOT differentiable; numpy under the hood."""
        qs = q.detach().cpu().numpy()
        backbones = np.stack([self._simulate_one(row) for row in qs])
        bb = torch.as_tensor(backbones, dtype=torch.float32)
        return {"backbone": bb, "tip": bb[:, -1, :]}


if _HAS_ELASTICA:

    class _SegmentCouple(ea.NoForces):
        """Constant couple applied to a contiguous element range of the rod."""

        def __init__(self, torque: np.ndarray, start: int, end: int):
            super().__init__()
            self.torque = torque
            self.start, self.end = start, end

        def apply_torques(self, system, time: float = 0.0):
            # Rotate the couple into each element's local material frame.
            for i in range(self.start, self.end):
                Qi = system.director_collection[..., i]  # (3, 3)
                system.external_torques[..., i] += Qi @ self.torque
