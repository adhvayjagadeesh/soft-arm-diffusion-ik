#!/usr/bin/env python3
"""Printable ArUco sheets, at an exact and verifiable physical scale.

Marker side is not a formatting choice. Monocular pose recovers range from
apparent size, so range scales linearly with the side length declared in
vision.py: a marker printed 5% small, or declared 5% large, reads 5% far. On
this arm that is ~15 mm of pure bias - it survives averaging, it is invisible
in any single frame, and it presents exactly as a sim-to-real gap.

So every page carries a 100 mm scale bar. Print at 100% (NOT "fit to page",
which silently rescales), then caliper the bar before trusting anything. A
previous printed template on this project came out at 96.9% and was only
caught because it had a bar to measure.

    python make_markers.py                 # disc sheet + size-test sheet
    python make_markers.py --side 25       # different marker side, mm

Sheets written:
  markers_discs.pdf   ids 0-5 at 30 mm for the printed tabs, plus id 9
                      (reference, bonded flat to the base plate) at 40 mm
  markers_sizetest.pdf ids 10/11/12 at 30/25/20 mm, for --size-test
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError as e:
    sys.exit(f"missing dependency: {e}  (pip install opencv-contrib-python matplotlib)")

PAGE_W, PAGE_H = 215.9, 279.4        # US Letter, mm
DISC_IDS = [0, 1, 2, 3, 4, 5]
REF_ID = 9                            # base plate reference, defines the base frame
SIZE_TEST = [(10, 30.0), (11, 25.0), (12, 20.0)]
OUT = os.path.dirname(os.path.abspath(__file__))


def _dict():
    d = cv2.aruco.DICT_4X4_50
    return (cv2.aruco.getPredefinedDictionary(d)
            if hasattr(cv2.aruco, "getPredefinedDictionary")
            else cv2.aruco.Dictionary_get(d))


def _marker_img(mid: int, px: int = 600) -> np.ndarray:
    """Marker bitmap. sidePixels includes the black border, which IS the
    measured side - the quiet zone is white space outside it."""
    D = _dict()
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(D, mid, px)
    return cv2.aruco.drawMarker(D, mid, px)            # pragma: no cover


def _place(fig, img, x_mm, y_mm, side_mm, label):
    """Place a marker with its lower-left corner at (x_mm, y_mm) from the
    page's lower-left. Positioning is in figure fractions so the PDF carries
    true physical size."""
    ax = fig.add_axes([x_mm / PAGE_W, y_mm / PAGE_H,
                       side_mm / PAGE_W, side_mm / PAGE_H])
    ax.imshow(img, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    # caption sits below the quiet zone, never inside it
    cap = fig.add_axes([x_mm / PAGE_W, (y_mm - 7.0) / PAGE_H,
                        side_mm / PAGE_W, 6.0 / PAGE_H])
    cap.axis("off")
    cap.text(0.5, 0.5, label, ha="center", va="center", fontsize=7.5)


def _scale_bar(fig, y_mm=18.0, length_mm=100.0):
    """The whole point of this page. Measure it before using the markers."""
    x0 = (PAGE_W - length_mm) / 2.0
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, PAGE_W); ax.set_ylim(0, PAGE_H)
    ax.add_patch(Rectangle((x0, y_mm), length_mm, 2.0, fc="black", ec="none"))
    for x in (x0, x0 + length_mm):                     # end ticks to measure between
        ax.add_patch(Rectangle((x - 0.4, y_mm - 3.5), 0.8, 9.0, fc="black", ec="none"))
    for k in range(1, 10):                             # 10 mm graduations
        ax.add_patch(Rectangle((x0 + k * 10 - 0.25, y_mm - 1.5), 0.5, 3.0,
                               fc="black", ec="none"))
    ax.text(PAGE_W / 2, y_mm - 7.0,
            f"{length_mm:.0f} mm - MEASURE THIS between the tall end ticks before "
            f"cutting any marker.", ha="center", va="top", fontsize=8)
    ax.text(PAGE_W / 2, y_mm - 12.0,
            "If it is not 100.0 mm, reprint at 100% scale ('Actual size', "
            "not 'Fit to page').", ha="center", va="top", fontsize=7.5,
            style="italic")
    return ax


def _sheet(title, items, note):
    fig = plt.figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4))
    bg = _scale_bar(fig)
    bg.text(PAGE_W / 2, PAGE_H - 14, title, ha="center", va="top",
            fontsize=13, weight="bold")
    bg.text(PAGE_W / 2, PAGE_H - 22, note, ha="center", va="top", fontsize=8.5)
    for img, x, y, s, lab in items:
        _place(fig, img, x, y, s, lab)
    return fig


def discs_sheet(side_mm: float) -> str:
    """Six tab markers plus the base-plate reference."""
    items, cols = [], 3
    pitch_x = PAGE_W / cols
    for k, mid in enumerate(DISC_IDS):
        cx = pitch_x * (k % cols) + pitch_x / 2.0
        cy = PAGE_H - 60.0 - (k // cols) * (side_mm + 26.0)
        items.append((_marker_img(mid), cx - side_mm / 2.0, cy, side_mm,
                      f"id {mid}  disc V{mid+1}  {side_mm:.0f} mm"))
    ref = 40.0
    items.append((_marker_img(REF_ID), PAGE_W / 2 - ref / 2.0, 52.0, ref,
                  f"id {REF_ID}  BASE PLATE reference  {ref:.0f} mm"))
    fig = _sheet("Disc markers", items,
                 f"Cut on the white, leaving >= 5 mm of white on every side. "
                 f"Put {side_mm:.0f} mm into vision.py MARKER_M.")
    p = f"{OUT}/markers_discs.pdf"
    fig.savefig(p); plt.close(fig)
    return p


def sizetest_sheet() -> str:
    items = []
    y = PAGE_H - 75.0
    for mid, s in SIZE_TEST:
        items.append((_marker_img(mid), (PAGE_W - s) / 2.0, y, s,
                      f"id {mid}   {s:.0f} mm"))
        y -= s + 26.0
    fig = _sheet("Marker size test", items,
                 "Tape where the ARM will be, then run:  "
                 "python vision.py --size-test")
    p = f"{OUT}/markers_sizetest.pdf"
    fig.savefig(p); plt.close(fig)
    return p


def checkerboard_sheet(square_mm: float = 25.0, inner=(9, 6)) -> str:
    """The calibration target --capture-calib needs.

    Squares must be inner+1 in each direction: OpenCV counts INTERIOR corners,
    so a (9,6) board is 10 x 7 squares. Getting this backwards is the usual
    reason findChessboardCorners never fires.

    The board is its own scale bar - four squares are exactly 100 mm - so a
    misprinted sheet is caught with the same caliper check. Square size must
    match SQUARE_M in vision.py, or every intrinsic comes out scaled.
    """
    nx, ny = inner[0] + 1, inner[1] + 1
    # A 10x7 board at 25 mm is 250x175, which only fits Letter standing up.
    # Orientation is free: findChessboardCorners is given the board rotated
    # every which way during capture anyway, so print it whichever way fits
    # and keep the squares as large as possible - a bigger board calibrates
    # better than a shrunken one.
    if nx * square_mm > PAGE_W - 8:
        nx, ny = ny, nx
    w, h = nx * square_mm, ny * square_mm
    if w > PAGE_W - 8 or h > PAGE_H - 26:
        sys.exit(f"board {w:.0f}x{h:.0f} mm does not fit US Letter - "
                 f"reduce --square")
    fig = plt.figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, PAGE_W); ax.set_ylim(0, PAGE_H)
    x0, y0 = (PAGE_W - w) / 2.0, (PAGE_H - h) / 2.0 - 6.0
    for i in range(nx):
        for j in range(ny):
            if (i + j) % 2 == 0:
                ax.add_patch(Rectangle((x0 + i * square_mm, y0 + j * square_mm),
                                       square_mm, square_mm, fc="black", ec="none"))
    ax.text(PAGE_W / 2, PAGE_H - 8,
            f"Camera calibration board - {inner[0]}x{inner[1]} inner corners, "
            f"{square_mm:.0f} mm squares", ha="center", va="top",
            fontsize=11, weight="bold")
    ax.text(PAGE_W / 2, PAGE_H - 15,
            "Print at 100%. CHECK: any 4 squares in a row = 100.0 mm exactly. "
            "Mount flat on rigid board - a curled sheet corrupts intrinsics.",
            ha="center", va="top", fontsize=8)
    ax.text(PAGE_W / 2, y0 - 6,
            f"vision.py must have CHESSBOARD = {inner} and "
            f"SQUARE_M = {square_mm/1000:.3f}", ha="center", va="top",
            fontsize=8, style="italic")
    p = f"{OUT}/calibration_board.pdf"
    fig.savefig(p); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", type=float, default=30.0,
                    help="disc marker side in mm (default 30, what the tabs take)")
    ap.add_argument("--square", type=float, default=25.0,
                    help="calibration board square size, mm (match vision.SQUARE_M)")
    args = ap.parse_args()

    a = discs_sheet(args.side)
    b = sizetest_sheet()
    c = checkerboard_sheet(args.square)
    print(f"wrote {a}\n      {b}\n      {c}")
    print(f"\n  1. Print BOTH at 100% scale - 'Actual size', not 'Fit to page'.")
    print(f"  2. Caliper the 100 mm bar on each page. It must read 100.0 mm.")
    print(f"  3. vision.py MARKER_M must equal the disc marker side: "
          f"{args.side/1000:.4f} m")
    print(f"\n  ids 0-5 -> disc tabs (bottom to tip), id {REF_ID} -> base plate,")
    print(f"  ids 10-12 -> size test only, never mounted on the arm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
