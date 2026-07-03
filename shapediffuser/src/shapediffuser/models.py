"""ShapeDiffuser model and baselines.

* ConditionalDenoiser: residual MLP with FiLM conditioning on (timestep, target).
* GaussianDiffusion:   DDPM training, DDIM sampling, classifier-free guidance.
* MLPRegressor:        deterministic IK baseline (mode-averaging by construction).
* MDNRegressor:        mixture-density-network baseline (fixed number of modes).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1)
        )
        ang = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class FiLMBlock(nn.Module):
    def __init__(self, hidden: int, emb_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.film = nn.Linear(emb_dim, 2 * hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(emb).chunk(2, dim=-1)
        h = self.norm(x) * (1.0 + scale) + shift
        h = self.fc2(F.silu(self.fc1(F.silu(h))))
        return x + h


class ConditionalDenoiser(nn.Module):
    def __init__(
        self,
        q_dim: int,
        cond_dim: int,
        hidden: int = 512,
        n_blocks: int = 6,
        emb_dim: int = 256,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(emb_dim), nn.Linear(emb_dim, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.null_cond = nn.Parameter(torch.zeros(emb_dim))
        self.in_proj = nn.Linear(q_dim, hidden)
        self.blocks = nn.ModuleList(FiLMBlock(hidden, emb_dim) for _ in range(n_blocks))
        self.out = nn.Linear(hidden, q_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None,
        drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.time_mlp(t)
        if cond is None:
            c = self.null_cond.unsqueeze(0).expand(x.shape[0], -1)
        else:
            c = self.cond_mlp(cond)
            if drop_mask is not None:
                c = torch.where(drop_mask.unsqueeze(-1), self.null_cond.unsqueeze(0), c)
        emb = emb + c
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h, emb)
        return self.out(h)


# --------------------------------------------------------------------------- #
def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    ac = f / f[0]
    betas = 1.0 - ac[1:] / ac[:-1]
    return betas.clamp(1e-8, 0.999).float()


class GaussianDiffusion(nn.Module):
    def __init__(self, denoiser: ConditionalDenoiser, q_dim: int, T: int = 1000,
                 cond_drop: float = 0.1):
        super().__init__()
        self.denoiser = denoiser
        self.q_dim = q_dim
        self.T = T
        self.cond_drop = cond_drop
        betas = cosine_beta_schedule(T)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def loss(self, x0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        a = self.alphas_cumprod[t].unsqueeze(-1)
        xt = a.sqrt() * x0 + (1.0 - a).sqrt() * noise
        drop = torch.rand(B, device=x0.device) < self.cond_drop
        eps = self.denoiser(xt, t, cond, drop_mask=drop)
        return F.mse_loss(eps, noise)

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        n_samples: int = 1,
        steps: int = 50,
        guidance: float = 1.0,
    ) -> torch.Tensor:
        """cond: (B, cond_dim) -> (B, n_samples, q_dim), values in ~[-1, 1]."""
        device = cond.device
        B = cond.shape[0]
        x = torch.randn(B * n_samples, self.q_dim, device=device)
        cond_rep = cond.repeat_interleave(n_samples, dim=0)
        times = torch.linspace(self.T - 1, 0, steps, device=device).round().long()
        for i in range(len(times)):
            t = times[i]
            tb = torch.full((x.shape[0],), int(t), device=device, dtype=torch.long)
            eps = self.denoiser(x, tb, cond_rep)
            if guidance != 1.0:
                eps_u = self.denoiser(x, tb, None)
                eps = eps_u + guidance * (eps - eps_u)
            a_t = self.alphas_cumprod[t]
            x0 = (x - (1.0 - a_t).sqrt() * eps) / a_t.sqrt()
            x0 = x0.clamp(-1.2, 1.2)
            if i + 1 < len(times):
                a_prev = self.alphas_cumprod[times[i + 1]]
                x = a_prev.sqrt() * x0 + (1.0 - a_prev).sqrt() * eps
            else:
                x = x0
        return x.view(B, n_samples, self.q_dim)


# --------------------------------------------------------------------------- #
class MLPRegressor(nn.Module):
    """Deterministic cond -> q regressor (the motor-babbling-CNN analogue)."""

    def __init__(self, cond_dim: int, q_dim: int, hidden: int = 512, n_layers: int = 5):
        super().__init__()
        layers, d = [], cond_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, q_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)

    def loss(self, q: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.forward(cond), q)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, n_samples: int = 1, **_) -> torch.Tensor:
        q = self.forward(cond)
        return q.unsqueeze(1).expand(-1, n_samples, -1).contiguous()


class MDNRegressor(nn.Module):
    """Mixture density network: cond -> GMM over q with K components."""

    def __init__(self, cond_dim: int, q_dim: int, k: int = 8, hidden: int = 512):
        super().__init__()
        self.k, self.q_dim = k, q_dim
        self.trunk = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.pi = nn.Linear(hidden, k)
        self.mu = nn.Linear(hidden, k * q_dim)
        self.log_sigma = nn.Linear(hidden, k * q_dim)

    def _params(self, cond: torch.Tensor):
        h = self.trunk(cond)
        log_pi = F.log_softmax(self.pi(h), dim=-1)
        mu = self.mu(h).view(-1, self.k, self.q_dim)
        log_sigma = self.log_sigma(h).view(-1, self.k, self.q_dim).clamp(-6.0, 2.0)
        return log_pi, mu, log_sigma

    def loss(self, q: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        log_pi, mu, log_sigma = self._params(cond)
        z = (q.unsqueeze(1) - mu) / log_sigma.exp()
        log_prob = -0.5 * (z**2 + 2 * log_sigma + math.log(2 * math.pi)).sum(dim=-1)
        return -torch.logsumexp(log_pi + log_prob, dim=-1).mean()

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, n_samples: int = 1, **_) -> torch.Tensor:
        log_pi, mu, log_sigma = self._params(cond)
        B = cond.shape[0]
        comp = torch.distributions.Categorical(logits=log_pi).sample((n_samples,)).T  # (B, n)
        idx = comp.unsqueeze(-1).expand(-1, -1, self.q_dim)
        mu_s = torch.gather(mu, 1, idx)
        sig_s = torch.gather(log_sigma.exp(), 1, idx)
        return mu_s + sig_s * torch.randn(B, n_samples, self.q_dim, device=cond.device)


def build_model(name: str, cond_dim: int, q_dim: int, cfg: dict) -> nn.Module:
    if name == "diffusion":
        den = ConditionalDenoiser(
            q_dim, cond_dim,
            hidden=cfg.get("hidden", 512),
            n_blocks=cfg.get("n_blocks", 6),
        )
        return GaussianDiffusion(den, q_dim, T=cfg.get("T", 1000),
                                 cond_drop=cfg.get("cond_drop", 0.1))
    if name == "mlp":
        return MLPRegressor(cond_dim, q_dim, hidden=cfg.get("hidden", 512))
    if name == "mdn":
        return MDNRegressor(cond_dim, q_dim, k=cfg.get("mdn_k", 8),
                            hidden=cfg.get("hidden", 512))
    raise ValueError(f"unknown model: {name}")
