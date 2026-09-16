#!/usr/bin/env python3
"""Print the LaTeX rows for the journal paper's amortization tables, straight
from the result JSONs. Every number in those tables must come from here, so
that a reviewer (or a later adversarial pass) can regenerate them.

    python scripts/print_paper_tables.py
"""

import json
import os


def pm(d, k, fmt="{:.2f}", scale=1.0):
    return f"${fmt.format(d[k]['mean'] * scale)}\\pm{fmt.format(d[k]['std'] * scale)}$"


def main():
    if not os.path.exists("results_morph_3seed.json"):
        raise SystemExit("results_morph_3seed.json not present yet")
    R = json.load(open("results_morph_3seed.json"))
    B = R["by_band"]
    bands = [("nominal", "Nominal arm"), ("in_dist", "In-distribution"), ("extrap", "Extrapolation")]
    models = ["amortized/diffusion", "blind/diffusion", "nominal/diffusion"]

    print("% ---- Table threeway: error / success (3 seeds) ----")
    for key, lab in bands:
        cells = []
        for m in models:
            d = B[key][m]
            cells.append(pm(d, "tip_err_best_of_K_mm"))
            cells.append(pm(d, "success_rate_best_of_K", "{:.0f}", 100).replace("$", "$", 1) + "\\%")
        print(f"{lab:16} & " + " & ".join(cells) + " \\\\")

    print("\n% ---- Table modes: recall / diversity (3 seeds) ----")
    for key, lab in bands:
        cells = []
        for m in models:
            d = B[key][m]
            cells.append(pm(d, "mode_recall"))
            cells.append(pm(d, "diversity", "{:.1f}"))
        print(f"{lab:16} & " + " & ".join(cells) + " \\\\")

    print("\n% ---- per-seed sanity: amortized in-dist ----")
    d = B["in_dist"]["amortized/diffusion"]
    print("  err per seed:", [round(v, 3) for v in d["tip_err_best_of_K_mm"]["per_seed"]])
    print("  recall per seed:", [round(v, 3) for v in d["mode_recall"]["per_seed"]])
    d = B["in_dist"]["blind/diffusion"]
    print("  blind err per seed:", [round(v, 2) for v in d["tip_err_best_of_K_mm"]["per_seed"]])
    ratio = B["in_dist"]["blind/diffusion"]["tip_err_best_of_K_mm"]["mean"] / B["in_dist"]["amortized/diffusion"]["tip_err_best_of_K_mm"]["mean"]
    print(f"  blind/amortized in-dist error ratio: {ratio:.1f}x")

    print("\n% ---- per-morphology extrapolation (3 seeds), amortized ----")
    for name, e in R["by_morphology"].items():
        if e["band"] != "extrap":
            continue
        d = e["models"]["amortized/diffusion"]
        print(f"  {name:18} {pm(d,'tip_err_best_of_K_mm')}  {pm(d,'success_rate_best_of_K','{:.0f}',100)}\\%  recall {pm(d,'mode_recall')}")

    if os.path.exists("results_morph_scale.json"):
        S = json.load(open("results_morph_scale.json"))["by_band"]
        print("\n% ---- amortized scale curve (seed 0): total samples -> in-dist / nominal ----")
        for m, n in (("n100k/diffusion", "1e5"), ("n200k/diffusion", "2e5"), ("n500k/diffusion", "5e5")):
            i = S["in_dist"][m]; o = S["nominal"][m]
            print(f"  {n}: in-dist {i['tip_err_best_of_K_mm']:.2f} mm / {100*i['success_rate_best_of_K']:.0f}%   "
                  f"nominal {o['tip_err_best_of_K_mm']:.2f} mm / {100*o['success_rate_best_of_K']:.0f}%")
        print("  specialist (single arm) for reference: 1e5 -> 0.71 mm, 2e5 -> 0.51 mm, 5e5 -> 0.40 mm")


if __name__ == "__main__":
    main()
