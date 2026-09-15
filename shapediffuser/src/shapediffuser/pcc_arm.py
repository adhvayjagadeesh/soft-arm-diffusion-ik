"""Differentiable piecewise-constant-curvature (PCC) soft manipulator in PyTorch.

Models an N-segment pneumatic continuum arm (STIFF-FLOP-like). Each segment has
3 chambers with normalized pressures in [0, 1]; a linear chamber->curvature map
gives per-segment curvature components (kx, ky). Forward kinematics composes
constant-curvature arcs. Fully batched and differentiable, so it also serves as
the gradient-based-IK baseline and a fast ground-truth engine for mode
enumeration. Swap in `elastica_arm.ElasticaArm` for high-fidelity Cosserat data.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _rz(phi: torch.Tensor) -> torch.Tensor:
    """Batched rotation about z. phi: (B,) -> (B, 3, 3)."""
    c, s = torch.cos(phi), torch.sin(phi)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack(
        [
            torch.stack([c, -s, zero], dim=-1),
            torch.stack([s, c, zero], dim=-1),
            torch.stack([zero, zero, one], dim=-1),
        ],
        dim=-2,
    )


def _ry(theta: torch.Tensor) -> torch.Tensor:
    """Batched rotation about y. theta: (B,) -> (B, 3, 3)."""
    c, s = torch.cos(theta), torch.sin(theta)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack(
        [
            torch.stack([c, zero, s], dim=-1),
            torch.stack([zero, one, zero], dim=-1),
            torch.stack([-s, zero, c], dim=-1),
        ],
        dim=-2,
    )


class PCCArm:
    """N-segment constant-curvature arm with 3 pressure chambers per segment.

    Actuation q in [0, 1]^{3N}. Chamber geometry (120 deg apart) maps pressures
    to bending-plane curvature components:
        kx = g * (p0 - 0.5 p1 - 0.5 p2)
        ky = g * (sqrt(3)/2) * (p1 - p2)
    """

    def __init__(
        self,
        n_segments: int = 4,
        seg_length: float | Sequence[float] = 0.06,
        curvature_gain: float | Sequence[float] = 25.0,
        points_per_seg: int = 10,
        rod_radius: float = 0.012,
        device: str | torch.device = "cpu",
    ):
        self.n_segments = n_segments
        self.points_per_seg = points_per_seg
        self.rod_radius = rod_radius
        self.device = torch.device(device)
        self.q_dim = 3 * n_segments
        # seg_length / curvature_gain may be a scalar (every segment alike -
        # the original arm) or one value per segment (a specific morphology
        # from the amortized family). Stored as (N,) tensors either way; the
        # scalar attributes are kept for code that reads them.
        self._L = self._per_seg(seg_length, "seg_length")
        self._G = self._per_seg(curvature_gain, "curvature_gain")
        self.seg_length = float(self._L.mean())
        self.curvature_gain = float(self._G.mean())
        # Arc-length fractions at which we evaluate backbone points (excl. base).
        self._s = torch.linspace(
            1.0 / points_per_seg, 1.0, points_per_seg, device=self.device
        )

    def _per_seg(self, v, name: str) -> torch.Tensor:
        t = torch.as_tensor(v, dtype=torch.float32, device=self.device).flatten()
        if t.numel() == 1:
            t = t.expand(self.n_segments).clone()
        if t.numel() != self.n_segments:
            raise ValueError(f"{name}: expected scalar or {self.n_segments} values, got {t.numel()}")
        return t

    @property
    def morph(self) -> torch.Tensor:
        """The morphology vector this arm IS: (2N,) = [L_1..L_N, g_1..g_N].

        This is what an amortized model is conditioned on. Ordering matters -
        data.generate_dataset writes it in the same order."""
        return torch.cat([self._L, self._G])

    def to(self, device) -> "PCCArm":
        self.device = torch.device(device)
        self._s = self._s.to(self.device)
        self._L = self._L.to(self.device)
        self._G = self._G.to(self.device)
        return self

    # ------------------------------------------------------------------ #
    def pressures_to_curvature(
        self, q: torch.Tensor, gain: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """q: (B, 3N) in [0,1] -> kx, ky each (B, N).

        gain: optional (B,) per-sample curvature_gain override, for domain
        randomization during data generation (see data.generate_dataset's
        curvature_gain_range). Defaults to self.curvature_gain (scalar,
        identical to prior behavior) when not provided.
        """
        B = q.shape[0]
        p = q.view(B, self.n_segments, 3)
        if gain is None:
            g = self._G.to(q.device).view(1, -1)                      # (1, N)
        else:
            g = gain.to(q.device)
            g = g.view(B, 1) if g.dim() == 1 else g.view(B, self.n_segments)
        kx = g * (p[..., 0] - 0.5 * p[..., 1] - 0.5 * p[..., 2])
        ky = g * (0.8660254037844386 * (p[..., 1] - p[..., 2]))
        return kx, ky

    def forward(
        self,
        q: torch.Tensor,
        gain: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> dict:
        """Batched FK. q: (B, 3N) -> dict with:
        backbone: (B, 1 + N*points_per_seg, 3) points from base to tip,
        tip:      (B, 3).
        gain:    optional per-sample curvature gain, (B,) shared across
                 segments or (B, N) per segment. Defaults to this arm's.
        lengths: optional per-sample segment lengths, (B, N). Defaults to
                 this arm's. Together with gain this lets one arm object
                 evaluate a whole batch of DIFFERENT morphologies, which is
                 what amortized-IK data generation needs.
        """
        q = q.to(self.device)
        B = q.shape[0]
        kx, ky = self.pressures_to_curvature(q, gain=gain)
        kappa = torch.sqrt(kx**2 + ky**2 + 1e-12)  # (B, N)
        phi = torch.atan2(ky, kx)  # (B, N)
        Lm = (self._L.view(1, -1).expand(B, -1) if lengths is None
              else lengths.to(self.device).view(B, self.n_segments))  # (B, N)

        R = torch.eye(3, device=self.device).expand(B, 3, 3).contiguous()
        p = torch.zeros(B, 3, device=self.device)
        pts = [p]

        small = kappa < 1e-4  # Taylor branch for near-straight segments
        for i in range(self.n_segments):
            k_i = kappa[:, i]  # (B,)
            phi_i = phi[:, i]
            L = Lm[:, i].unsqueeze(-1)  # (B, 1)
            ls = L * self._s  # (B, S)
            theta_s = k_i.unsqueeze(-1) * ls  # (B, S)

            # Safe arc integrals: a = (1 - cos th)/k, b = sin th / k
            k_safe = k_i.clamp_min(1e-6).unsqueeze(-1)
            a_exact = (1.0 - torch.cos(theta_s)) / k_safe
            b_exact = torch.sin(theta_s) / k_safe
            a_taylor = 0.5 * (ls**2) * k_i.unsqueeze(-1)
            b_taylor = ls
            a = torch.where(small[:, i : i + 1], a_taylor, a_exact)  # (B, S)
            b = torch.where(small[:, i : i + 1], b_taylor, b_exact)

            c_phi = torch.cos(phi_i).unsqueeze(-1)
            s_phi = torch.sin(phi_i).unsqueeze(-1)
            local = torch.stack([c_phi * a, s_phi * a, b], dim=-1)  # (B, S, 3)

            world = p.unsqueeze(1) + torch.einsum("bij,bsj->bsi", R, local)
            pts.append(world.reshape(B * self.points_per_seg, 3).view(B, -1, 3))

            theta_end = k_i * L.squeeze(-1)
            R_seg = _rz(phi_i) @ _ry(theta_end) @ _rz(-phi_i)  # (B, 3, 3)
            p = world[:, -1, :]
            R = R @ R_seg

        backbone = torch.cat(
            [pts[0].unsqueeze(1)] + [w for w in pts[1:]], dim=1
        )  # (B, 1 + N*S, 3)
        return {"backbone": backbone, "tip": p}

    # ------------------------------------------------------------------ #
    def collides(self, backbone: torch.Tensor, obstacles: torch.Tensor) -> torch.Tensor:
        """backbone: (B, P, 3); obstacles: (M, 4) rows [cx, cy, cz, radius].
        Returns bool (B,) True if any backbone point is inside any obstacle
        (inflated by rod radius)."""
        if obstacles.numel() == 0:
            return torch.zeros(backbone.shape[0], dtype=torch.bool, device=backbone.device)
        centers = obstacles[:, :3].to(backbone.device)  # (M, 3)
        radii = obstacles[:, 3].to(backbone.device) + self.rod_radius  # (M,)
        d = torch.cdist(backbone, centers.unsqueeze(0).expand(backbone.shape[0], -1, -1))
        return (d < radii.view(1, 1, -1)).any(dim=(1, 2))


def grad_ik(
    arm: PCCArm,
    target_tip: torch.Tensor,
    n_restarts: int = 32,
    iters: int = 300,
    lr: float = 0.05,
) -> torch.Tensor:
    """Gradient-descent IK baseline through the differentiable simulator.

    target_tip: (3,). Returns (n_restarts, q_dim) candidate solutions sorted by
    final tip error (best first).

    Restarts are initialized uniformly over PRESSURE space (logit-transformed
    so the sigmoid reparameterization reproduces them exactly). An earlier
    version drew the logits themselves ~ U(0,1), which confines the initial
    pressures to [0.5, 0.73] and materially understates the baseline's mode
    coverage (recall 0.75 -> 0.80, solution diversity 9.3 -> 28 after the fix).
    """
    p0 = torch.rand(n_restarts, arm.q_dim, device=arm.device).clamp(1e-4, 1 - 1e-4)
    q = torch.logit(p0).requires_grad_(True)
    opt = torch.optim.Adam([q], lr=lr)
    tgt = target_tip.to(arm.device).unsqueeze(0)
    for _ in range(iters):
        opt.zero_grad()
        out = arm.forward(torch.sigmoid(q))  # keep pressures in [0,1]
        loss = ((out["tip"] - tgt) ** 2).sum(dim=-1).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        qs = torch.sigmoid(q)
        err = (arm.forward(qs)["tip"] - tgt).norm(dim=-1)
        order = torch.argsort(err)
    return qs[order].detach()
