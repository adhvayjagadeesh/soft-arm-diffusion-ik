#!/usr/bin/env python3
"""Graphical abstract for the MDPI submission.

MDPI spec: a single self-explanatory image, minimum 1100 px wide and
560 px tall, PNG/JPG/TIFF. This is the paper in one glance: the envelope
sweep (one model, flat across a family of arms, cliff outside it) beside
the three-way comparison (told the morphology vs. not).

    python scripts/make_graphical_abstract.py
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, YELLOW, GRAY, AQUA, RED = "#2a78d6", "#eda100", "#6f6e69", "#1baf7a", "#d6452a"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "lines.linewidth": 1.8, "lines.markersize": 5,
})

E = json.load(open("results_envelope_seed0.json"))
R = json.load(open("results_morph_3seed.json"))["by_band"]

fig = plt.figure(figsize=(7.6, 4.0), dpi=300)
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], left=0.07, right=0.98, top=0.70, bottom=0.17, wspace=0.32)

# ---- headline ----
fig.text(0.5, 0.955, "One model, many arms", ha="center", va="top", fontsize=15, weight="bold")
fig.text(0.5, 0.885,
         "A single diffusion model, conditioned on the arm's morphology, solves inverse kinematics for "
         "soft arms it never saw.\nThe same data with the morphology withheld (domain randomization) does not.",
         ha="center", va="top", fontsize=8.4, color="#333")

# ---- left: envelope over segment length ----
ax = fig.add_subplot(gs[0, 0])
rows = E["sweeps"]["seg_length"]
lo, hi = [v * 1000 for v in E["box"]["seg_length"]]
ax.axvspan(lo, hi, color=AQUA, alpha=0.12, lw=0)
ax.text((lo + hi) / 2, 0.19, "training family", ha="center", va="bottom", color=AQUA, fontsize=8, weight="bold")
x = [r["value"] * 1000 for r in rows]
ax.plot(x, [r["amortized"]["best_of_K_mm"] for r in rows], marker="o", color=BLUE, label="told the morphology")
ax.plot(x, [r["blind"]["best_of_K_mm"] for r in rows], marker="o", color=YELLOW, label="morphology withheld")
ax.axhline(5, color=RED, ls="--", lw=0.9)
ax.text(x[0], 5.4, "5 mm success tolerance", color=RED, fontsize=7.5, va="bottom")
ax.set_yscale("log"); ax.set_ylim(0.15, 150)
ax.set_xlabel("segment length of the queried arm (mm)")
ax.set_ylabel("tip error, best of 32 (mm)")
ax.set_title("Accuracy across a continuous family of arms", fontsize=9.5)
ax.legend(loc="upper left", frameon=False)

# ---- right: three-way bars, in-distribution ----
ax2 = fig.add_subplot(gs[0, 1])
labels = ["told the\nmorphology", "morphology\nwithheld", "trained on\none arm"]
keys = ["amortized/diffusion", "blind/diffusion", "nominal/diffusion"]
cols = [BLUE, YELLOW, GRAY]
mu = [R["in_dist"][k]["tip_err_best_of_K_mm"]["mean"] for k in keys]
sd = [R["in_dist"][k]["tip_err_best_of_K_mm"]["std"] for k in keys]
bars = ax2.bar(range(3), mu, 0.62, yerr=sd, color=cols, capsize=3)
for b, m in zip(bars, mu):
    ax2.text(b.get_x() + b.get_width() / 2, m * 1.25, f"{m:.1f} mm" if m >= 1 else f"{m:.2f} mm",
             ha="center", va="bottom", fontsize=8.5, weight="bold")
ax2.set_yscale("log"); ax2.set_ylim(0.15, 150)
ax2.axhline(5, color=RED, ls="--", lw=0.9)
ax2.set_xticks(range(3)); ax2.set_xticklabels(labels)
ax2.set_ylabel("tip error on unseen arms (mm)")
ax2.set_title("Same data, one conditioning vector: 31×", fontsize=9.5)

fig.savefig("submission/graphical_abstract.png", dpi=300, facecolor="white")
from PIL import Image  # noqa: E402
im = Image.open("submission/graphical_abstract.png")
print(f"wrote submission/graphical_abstract.png  {im.size[0]}x{im.size[1]} px  (MDPI min 1100x560)")
