#!/usr/bin/env python3
"""Publication figures for the RA-L submission.

Reads the committed result JSONs (never recomputes anything) and renders the
four paper figures to paper/figures/ as PDF (vector, for the manuscript) and
PNG (300dpi, for quick review). Figure design follows the dataviz method:
form first, one axis per unit, categorical colors in fixed validated order
(blue #2a78d6, aqua #1baf7a, yellow #eda100) with marker shapes as secondary
encoding so identity survives grayscale/CVD print, neutral gray for
reference lines, recessive grid, thin marks.

    python scripts/make_paper_figures.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"
GRAY = "#6f6e69"
OUT = "paper/figures"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.grid": True,
    "axes.grid.axis": "y", "lines.linewidth": 1.4, "lines.markersize": 4.5,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png", dpi=300)
    plt.close(fig)
    print(f"wrote {OUT}/{name}.pdf + .png")


def fig_finetune_cliff():
    """Centerpiece: retention vs transfer across PCC mixing ratios, 3 seeds.
    Both measures are tip error in mm -> one shared axis, two series."""
    d = json.load(open("finetune_grid_summary.json"))
    ratios = [0.0, 0.05, 0.1, 0.15, 0.2, 0.5]
    pcc = [d["finetune_grid"][f"pcc_mix_{r}"]["pcc_err_mm"] for r in ratios]
    ela = [d["finetune_grid"][f"pcc_mix_{r}"]["elastica_err_mm"] for r in ratios]
    base_ela = d["baseline_transfer"]["diffusion"]["elastica_tip_err_best_of_K_mm"]["mean"]
    base_pcc = d["baseline_transfer"]["diffusion"]["pcc_tip_err_best_of_K_mm"]["mean"]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = [r * 100 for r in ratios]
    ax.errorbar(x, [v["mean"] for v in ela], yerr=[v["std"] for v in ela],
                color=BLUE, marker="o", capsize=2.5, label="Elastica error (transfer)")
    ax.errorbar(x, [v["mean"] for v in pcc], yerr=[v["std"] for v in pcc],
                color=AQUA, marker="s", capsize=2.5, label="PCC error (retention)")
    ax.axhline(base_ela, color=GRAY, ls="--", lw=0.9)
    ax.text(3.2, base_ela - 9, f"no-fine-tune Elastica baseline ({base_ela:.0f} mm)",
            ha="left", va="top", color=GRAY, fontsize=6.5)
    ax.axhline(base_pcc, color=GRAY, ls=":", lw=0.9)
    ax.text(0, base_pcc + 5, f"no-fine-tune PCC baseline ({base_pcc:.1f} mm)",
            ha="left", color=GRAY, fontsize=6.5)
    ax.set_xlabel("PCC data fraction during fine-tuning (%)")
    ax.set_ylabel("best-of-K tip error (mm)")
    ax.set_xticks(x)
    ax.set_ylim(0, 175)
    ax.legend(frameon=False, loc="center right")
    save(fig, "fig_finetune_cliff")


def fig_query_budget():
    """Best-of-m Elastica error vs query budget, two candidate orderings."""
    pcc_o = json.load(open("query_budget_results_n45.json"))["budgets"]
    div_o = json.load(open("query_budget_results_n45_diversity.json"))["budgets"]
    mlp_ref = json.load(open("transfer_study_results.json"))["mlp"]["elastica_tip_err_best_of_K_mm"]
    ms = sorted(int(m) for m in pcc_o)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for d, color, marker, label in ((pcc_o, BLUE, "o", "PCC-confidence order"),
                                    (div_o, AQUA, "s", "diversity order")):
        ax.errorbar(ms, [d[str(m)]["elastica_tip_err_best_of_m_mm"] for m in ms],
                    yerr=[d[str(m)]["elastica_tip_err_sem_mm"] for m in ms],
                    color=color, marker=marker, capsize=2.5, label=label)
    ax.axhline(mlp_ref, color=GRAY, ls="--", lw=0.9)
    ax.text(1.18, mlp_ref + 2.2, f"MLP (single candidate, {mlp_ref:.0f} mm)",
            ha="left", color=GRAY, fontsize=6.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ms)
    ax.set_xticklabels([str(m) for m in ms])
    ax.set_xlabel("Elastica query budget $m$")
    ax.set_ylabel("mean best-of-$m$ tip error (mm)")
    ax.legend(frameon=False, loc="lower left")
    save(fig, "fig_query_budget")


def fig_redundancy():
    """(a) diversity collapse under shape conditioning; (b) morphology scaling.
    Different measures -> separate panels, each with its own single axis."""
    tip = json.load(open("multiseed_results.json"))["diffusion"]["multimodality.diversity"]
    shape = json.load(open("multiseed_shape_results.json"))["diffusion"]["multimodality.diversity"]
    morph = json.load(open("morphology_sweep_results.json"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.4))

    xs = [0, 1]
    vals = [tip, shape]
    bars = ax1.bar(xs, [v["mean"] for v in vals], yerr=[v["std"] for v in vals],
                   width=0.55, color=BLUE, capsize=3, error_kw={"linewidth": 0.9})
    for x, v in zip(xs, vals):
        ax1.text(x, v["mean"] * 1.35, f'{v["mean"]:.2f}', ha="center", fontsize=7)
    ax1.set_yscale("log")
    ax1.set_ylim(0.1, 90)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(["tip target\n(3-D)", "whole-shape target\n(24-D)"])
    ax1.set_ylabel("sampled diversity (curvature space)")
    ax1.set_title("(a) redundancy collapses under shape conditioning")

    segs = sorted(int(s) for s in morph.keys())
    for model, color, marker, label in (("diffusion", BLUE, "o", "diffusion"),
                                        ("mlp", AQUA, "s", "MLP")):
        ax2.plot(segs, [morph[str(s)][model]["tip_err_best_of_K_mm"] for s in segs],
                 color=color, marker=marker, label=label)
    ax2.set_yscale("log")
    ax2.set_xticks(segs)
    ax2.set_xlabel("arm segments (actuation redundancy)")
    ax2.set_ylabel("best-of-K tip error (mm)")
    ax2.set_title("(b) advantage grows with redundancy")
    ax2.legend(frameon=False, loc="upper left")
    save(fig, "fig_redundancy")


def fig_scaling():
    """Dataset-size scaling: error and mode recall, separate panels."""
    s = json.load(open("scale_ablation_results.json"))
    ns = sorted(int(n) for n in s)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.3))
    ax1.plot(ns, [s[str(n)]["accuracy"]["tip_err_best_of_K_mm"] for n in ns],
             color=BLUE, marker="o")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("training samples")
    ax1.set_ylabel("best-of-K tip error (mm)")
    ax1.set_title("(a) accuracy vs. dataset size")

    ax2.plot(ns, [s[str(n)]["multimodality"]["mode_recall"] for n in ns],
             color=BLUE, marker="o")
    ax2.set_xscale("log")
    ax2.set_ylim(0, 1.0)
    ax2.set_xlabel("training samples")
    ax2.set_ylabel("mode recall")
    ax2.set_title("(b) mode recall vs. dataset size")
    save(fig, "fig_scaling")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_finetune_cliff()
    fig_query_budget()
    fig_redundancy()
    fig_scaling()
    print("done")
