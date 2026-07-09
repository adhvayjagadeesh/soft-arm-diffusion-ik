#!/usr/bin/env python3
"""Supplementary video for the RA-L submission (Task #5).

Two scenes, both from existing checkpoints (no new training):
  1. DDIM denoising: eight candidate arms evolve from noise into distinct
     IK solutions for one tip target (the method, visibly).
  2. Obstacle avoidance: the MLP's single answer fails; diffusion's
     candidates include accurate, collision-free alternatives (the payoff).

Anonymity-safe: no names, logos, or URLs anywhere in the video (RA-L
double-anonymous rules apply to supplementary material too).

    python scripts/make_supplementary_video.py --out paper/video/supplementary.mp4
    python scripts/make_supplementary_video.py --preview   # short/fast render
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import imageio_ffmpeg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
from matplotlib.animation import FFMpegWriter  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shapediffuser import PCCArm, BabblingDataset  # noqa: E402
from shapediffuser.metrics import sample_reachable_targets, tip_error  # noqa: E402
from evaluate import load_model  # noqa: E402

BLUE, ORANGE, GRAY, RED = "#2a78d6", "#eb6834", "#9a9a94", "#e34948"


@torch.no_grad()
def sample_with_trajectory(model, cond, n_samples, steps, guidance):
    """DDIM loop mirroring GaussianDiffusion.sample, recording the predicted
    x0 at every step (the visually meaningful quantity)."""
    device = cond.device
    x = torch.randn(n_samples, model.q_dim, device=device)
    cond_rep = cond.repeat(n_samples, 1)
    times = torch.linspace(model.T - 1, 0, steps, device=device).round().long()
    traj = []
    for i in range(len(times)):
        t = times[i]
        tb = torch.full((x.shape[0],), int(t), device=device, dtype=torch.long)
        eps = model.denoiser(x, tb, cond_rep)
        if guidance != 1.0:
            eps_u = model.denoiser(x, tb, None)
            eps = eps_u + guidance * (eps - eps_u)
        a_t = model.alphas_cumprod[t]
        x0 = ((x - (1.0 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1.2, 1.2)
        traj.append(x0.clone())
        if i + 1 < len(times):
            a_prev = model.alphas_cumprod[times[i + 1]]
            x = a_prev.sqrt() * x0 + (1.0 - a_prev).sqrt() * eps
        else:
            x = x0
    return traj  # list of (n_samples, q_dim), scaled space


def backbones(arm, q_scaled):
    press = BabblingDataset.q_to_pressures(q_scaled)
    with torch.no_grad():
        return arm.forward(press)["backbone"].numpy()


def draw_sphere(ax, c, r, color="#c9c8c2", alpha=0.4):
    u = np.linspace(0, 2 * np.pi, 14)
    v = np.linspace(0, np.pi, 10)
    x = c[0] + r * np.outer(np.cos(u), np.sin(v))
    y = c[1] + r * np.outer(np.sin(u), np.sin(v))
    z = c[2] + r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=True)


def setup_ax(ax):
    ax.set_xlim(-0.17, 0.17)
    ax.set_ylim(-0.17, 0.17)
    ax.set_zlim(0.0, 0.26)
    ax.set_box_aspect((1, 1, 0.72))
    ax.set_position([0.02, -0.04, 0.96, 1.04])  # fill the frame; axis is off
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--out", default="paper/video/supplementary.mp4")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--preview", action="store_true",
                    help="tiny/fast render for smoke-testing")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ev = cfg["eval"]
    arm = PCCArm(**cfg["arm"])
    tol = ev["success_tol"]

    diff, dnorm, _ = load_model(os.path.join(args.ckpt_dir, "diffusion.pt"), "cpu")
    mlp, mnorm, _ = load_model(os.path.join(args.ckpt_dir, "mlp.pt"), "cpu")

    fps = args.fps
    frames_per_ddim_step = 3 if args.preview else 12
    hold = fps // (4 if args.preview else 1)

    # ---------- Scene 1 data: denoising trajectory ---------- #
    torch.manual_seed(7)
    target1 = sample_reachable_targets(arm, 1, seed=4141)[0]
    traj = sample_with_trajectory(diff, dnorm.encode(target1.unsqueeze(0)),
                                  n_samples=8, steps=ev["ddim_steps"],
                                  guidance=ev["guidance"])
    # interpolate between consecutive x0 predictions for smooth motion
    s1_backbones = []
    for a, b in zip(traj[:-1], traj[1:]):
        for f in range(frames_per_ddim_step):
            w = f / frames_per_ddim_step
            s1_backbones.append(backbones(arm, (1 - w) * a + w * b))
    s1_backbones += [backbones(arm, traj[-1])] * hold  # hold the final pose

    # ---------- Scene 2 data: obstacle trials ---------- #
    rng = np.random.default_rng(2026)
    reach = sample_reachable_targets(arm, 200, seed=999)
    total_len = cfg["arm"]["n_segments"] * cfg["arm"]["seg_length"]
    K = 16
    trials = []
    torch.manual_seed(11)
    for i in range(200):
        t = reach[i]
        obs = []
        while len(obs) < ev["n_obstacles"]:
            c = rng.uniform(-0.6, 0.6, size=3) * total_len
            c[2] = rng.uniform(0.1, 0.9) * total_len
            if np.linalg.norm(c - t.numpy()) > ev["obstacle_radius"] + 0.03:
                obs.append(np.concatenate([c, [ev["obstacle_radius"]]]))
        if len(trials) >= 2:
            break
        obstacles = torch.as_tensor(np.stack(obs), dtype=torch.float32)
        with torch.no_grad():
            qd = diff.sample(dnorm.encode(t.unsqueeze(0)), n_samples=K,
                             steps=ev["ddim_steps"], guidance=ev["guidance"])[0]
            qm = mlp.sample(mnorm.encode(t.unsqueeze(0)), n_samples=1)[0]
        pd = BabblingDataset.q_to_pressures(qd)
        pm = BabblingDataset.q_to_pressures(qm)
        ed = tip_error(arm, pd, t.unsqueeze(0).expand(K, -1))
        em = tip_error(arm, pm, t.unsqueeze(0)).item()
        with torch.no_grad():
            bbd = arm.forward(pd)["backbone"]
            bbm = arm.forward(pm)["backbone"]
        cold = arm.collides(bbd, obstacles).numpy()
        colm = arm.collides(bbm, obstacles).numpy()[0]
        d_ok = ((ed.numpy() < tol) & ~cold)
        m_fail = (em > tol) or colm
        if d_ok.any() and m_fail:  # the story we want to show, and the common case
            trials.append(dict(t=t.numpy(), obs=np.stack(obs), bbd=bbd.numpy(),
                               bbm=bbm.numpy()[0], ok=d_ok,
                               acc=(ed.numpy() < tol), col=cold))

    # ---------- timeline ---------- #
    title_frames = fps * (1 if args.preview else 2)
    s2_frames_each = fps * (3 if args.preview else 9)
    end_frames = fps * (1 if args.preview else 2)
    total = (title_frames + len(s1_backbones)
             + s2_frames_each * len(trials) + end_frames)
    print(f"scenes: title {title_frames}f, denoise {len(s1_backbones)}f, "
          f"{len(trials)} obstacle trials x {s2_frames_each}f, end {end_frames}f "
          f"= {total} frames @ {fps}fps", flush=True)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    caption = fig.text(0.5, 0.06, "", ha="center", fontsize=15, color="#333333")
    header = fig.text(0.5, 0.95, "", ha="center", fontsize=17, color="#111111")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=3000)
    cmap = plt.get_cmap("viridis")

    with writer.saving(fig, args.out, dpi=100):
        frame = 0
        for k in range(total):
            ax.clear()
            setup_ax(ax)
            azim = -60 + 40 * np.sin(2 * np.pi * k / (fps * 30))
            ax.view_init(elev=18, azim=azim)

            if k < title_frames:
                header.set_text("Multimodal Diffusion Inverse Kinematics"
                                "\nfor Soft Continuum Arms — Supplementary Video")
                caption.set_text("")
            elif k < title_frames + len(s1_backbones):
                i = k - title_frames
                header.set_text("Conditional diffusion sampling")
                caption.set_text("Eight candidates denoise from noise into "
                                 "distinct IK solutions for the same tip target")
                bbs = s1_backbones[i]
                for j in range(bbs.shape[0]):
                    ax.plot(bbs[j, :, 0], bbs[j, :, 1], bbs[j, :, 2],
                            color=cmap(j / 7), lw=2.2, alpha=0.9)
                ax.scatter(*target1.numpy(), color="black", s=110, marker="*")
            elif k < total - end_frames:
                i2 = (k - title_frames - len(s1_backbones)) // s2_frames_each
                f2 = (k - title_frames - len(s1_backbones)) % s2_frames_each
                tr = trials[min(i2, len(trials) - 1)]
                header.set_text(f"Obstacle avoidance — trial {min(i2, len(trials) - 1) + 1}")
                caption.set_text("MLP's single answer (red) fails; diffusion "
                                 "candidates include accurate, collision-free "
                                 "solutions (blue)")
                for o in tr["obs"]:
                    draw_sphere(ax, o[:3], o[3])
                ax.scatter(*tr["t"], color="black", s=110, marker="*")
                n_show = min(tr["bbd"].shape[0],
                             1 + int(tr["bbd"].shape[0] * min(1.0, f2 / (s2_frames_each * 0.5))))
                for j in range(n_show):
                    if tr["ok"][j]:
                        c, lw, al = BLUE, 2.6, 0.95
                    elif tr["acc"][j]:
                        c, lw, al = ORANGE, 1.6, 0.55
                    else:
                        c, lw, al = GRAY, 1.2, 0.35
                    ax.plot(tr["bbd"][j, :, 0], tr["bbd"][j, :, 1],
                            tr["bbd"][j, :, 2], color=c, lw=lw, alpha=al)
                # MLP last so it stays visible over the candidate bundle
                ax.plot(tr["bbm"][:, 0], tr["bbm"][:, 1], tr["bbm"][:, 2],
                        color=RED, lw=3.4, ls="--", zorder=10)
            else:
                header.set_text("All code, data, and results:"
                                "\nsee the repository linked in the paper")
                caption.set_text("")

            writer.grab_frame()
            frame += 1
            if frame % 100 == 0:
                print(f"{frame}/{total} frames", flush=True)

    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
