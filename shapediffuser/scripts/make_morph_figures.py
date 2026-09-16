#!/usr/bin/env python3
"""Figures for the amortization results. Same style as make_paper_figures.py.

    python scripts/make_morph_figures.py            # seed-0 JSONs
    python scripts/make_morph_figures.py --three    # results_morph_3seed.json once it exists
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, AQUA, YELLOW, RED = "#2a78d6", "#1baf7a", "#eda100", "#d6452a"
GRAY = "#6f6e69"
OUT = "paper_journal/figures"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.grid": True,
    "axes.grid.axis": "y", "lines.linewidth": 1.4, "lines.markersize": 4.5,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL = {"amortized": BLUE, "blind": YELLOW, "nominal": GRAY}
LBL = {"amortized": "Amortized (told $\\theta$)", "blind": "Blind (same data, $\\theta$ withheld)",
       "nominal": "Nominal (single-arm)"}


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png", dpi=300)
    plt.close(fig)
    print(f"wrote {OUT}/{name}.pdf + .png")


def fig_envelope():
    """Two panels, one per swept axis. Log error, training box shaded."""
    r = json.load(open("results_envelope_seed0.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, (axis, xlabel, scale) in zip(axes, (("seg_length", "segment length (mm)", 1000.0),
                                                ("curvature_gain", "curvature gain", 1.0))):
        rows = r["sweeps"][axis]
        lo, hi = [v * scale for v in r["box"][axis]]
        ax.axvspan(lo, hi, color=AQUA, alpha=0.10, lw=0, label="training box")
        for m in ("amortized", "blind"):
            x = [row["value"] * scale for row in rows]
            y = [row[m]["best_of_K_mm"] for row in rows]
            ax.plot(x, y, marker="o", color=COL[m], label=LBL[m])
        ax.axhline(5.0, color=RED, ls="--", lw=0.8)
        ax.text(ax.get_xlim()[0], 5.0, " 5 mm tolerance", va="bottom", ha="left", color=RED, fontsize=6.5)
        ax.set_yscale("log"); ax.set_ylim(0.15, 150)
        ax.set_xlabel(xlabel); ax.set_ylabel("best-of-32 tip error (mm)")
    axes[0].set_title("(a) all four segment lengths swept, gain = 25")
    axes[1].set_title("(b) all four gains swept, length = 60 mm")
    axes[0].legend(loc="upper center", frameon=False, ncol=1)
    save(fig, "fig_envelope")


def fig_threeway(three=False):
    """Grouped bars: error (log) and mode recall per band, three models."""
    if three and os.path.exists("results_morph_3seed.json"):
        r = json.load(open("results_morph_3seed.json"))["by_band"]
        get = lambda b, m, k: (r[b][f"{m}/diffusion"][k]["mean"], r[b][f"{m}/diffusion"][k]["std"])
        src = "3 seeds, mean ± std"
    else:
        r = json.load(open("results_morph_modes_seed0.json"))["by_band"]
        get = lambda b, m, k: (r[b][f"{m}/diffusion"][k], 0.0)
        src = "seed 0"
    bands = [("nominal", "nominal arm"), ("in_dist", "in-distribution"), ("extrap", "extrapolation")]
    models = ["amortized", "blind", "nominal"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    w = 0.26
    for ax, (key, ylabel, log) in zip(axes, (("tip_err_best_of_K_mm", "best-of-32 tip error (mm)", True),
                                             ("mode_recall", "mode recall", False))):
        for i, m in enumerate(models):
            xs = np.arange(len(bands)) + (i - 1) * w
            mu = [get(b, m, key)[0] for b, _ in bands]
            sd = [get(b, m, key)[1] for b, _ in bands]
            ax.bar(xs, mu, w, yerr=sd if any(sd) else None, color=COL[m], label=LBL[m], capsize=2)
        ax.set_xticks(range(len(bands))); ax.set_xticklabels([n for _, n in bands])
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log"); ax.set_ylim(0.2, 100)
            ax.axhline(5.0, color=RED, ls="--", lw=0.8)
        else:
            ax.set_ylim(0, 1.0)
    axes[0].set_title(f"(a) accuracy ({src})")
    axes[1].set_title(f"(b) ground-truth mode recall ({src})")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.subplots_adjust(bottom=0.22)
    save(fig, "fig_threeway")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--three", action="store_true")
    a = ap.parse_args()
    fig_envelope()
    fig_threeway(a.three)
