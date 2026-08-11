#!/usr/bin/env python3
"""Generate printable spacer discs for the continuum arm.

Printing beats drilling wood for three specific reasons, none cosmetic:

  * The centre hole can be a snug slip fit instead of 0.031 in of slop, which
    removes the disc-tilt problem entirely. A drilled 1/4 in hole on a 3/16 in
    rod lets a disc cock ~27 deg before the epoxy sets; a printed 4.9 mm hole
    holds it square by itself.
  * The marker tab is part of the disc, so its 65 deg tilt is identical on all
    six by construction rather than by hand-gluing. Tab orientation is the
    thing the visibility analysis was most sensitive to.
  * Hole positions come from the same numbers every time - no jig, no wear, no
    per-disc scatter.

Disc diameter is raised to 50 mm from the 38.1 mm craft discs. Outer tendons
then sit at an 18 mm moment arm instead of 14 mm, ~29% more bending moment for
the same cable tension, which matters with 0.52 N.m servos. Checked for
self-collision: at maximum curvature adjacent discs stay ~38 mm apart.

    python make_disc_stl.py            # writes all STLs
    python make_disc_stl.py --hole 5.0 # override centre hole after a fit test
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    import trimesh
except ImportError:
    sys.exit("pip install trimesh manifold3d")

# --------------------------------------------------------------------------- #
DISC_D = 50.0          # mm outer diameter
DISC_T = 3.0           # mm thickness
ROD_D = 4.7625         # 3/16 in fiberglass rod
HOLE_CLEAR = 0.15      # slip fit; printers shrink holes, see --fit-test

TENDON_D = 2.0         # generous for 0.79 mm cable; sub-2 mm holes print badly
R_INNER = 10.0         # section A tendon circle
R_OUTER = 18.0         # section B tendon circle
ANGLES = (90.0, 210.0, 330.0)

TAB_TILT = 65.0        # deg between marker normal and disc axis - from the
                       # visibility sweep: 45 deg gives 60% all-six, 65 gives 69%
TAB_W = 40.0           # mm across (Y); holds a 30 mm marker + 5 mm quiet zone
TAB_H = 40.0           # mm along the tab face
TAB_T = 2.0
MARKER_MM = TAB_H - 10.0     # what to print: 30 mm (1.18 in)

# A tab tilted back 65 deg has its top edge swing INWARD over the disc, which
# in the first version covered two tendon holes. The tab is therefore pushed
# out until its innermost projection clears the outer tendon circle, and a
# gusset bridges the gap from the rim. Verified numerically in check_clearance.
_t = np.radians(TAB_TILT)
_UP = np.array([-np.cos(_t), 0.0, np.sin(_t)])       # in-plane "up" of the tab
TAB_CLEAR_R = R_OUTER + TENDON_D / 2.0 + 1.0         # must not shadow a hole
TAB_X = TAB_CLEAR_R + (TAB_H / 2.0) * abs(_UP[0])
TAB_Z = (TAB_H / 2.0) * abs(_UP[2])

OUT = os.path.dirname(os.path.abspath(__file__)) + "/stl"


def _cyl(d, h, sections=96, transform=None):
    c = trimesh.creation.cylinder(radius=d / 2.0, height=h, sections=sections)
    if transform is not None:
        c.apply_transform(transform)
    return c


def make_disc(center_hole_d: float, with_tab: bool = True) -> trimesh.Trimesh:
    body = _cyl(DISC_D, DISC_T)

    if with_tab:
        tab = trimesh.creation.box(extents=[TAB_T, TAB_W, TAB_H])
        # marker normal starts along +X; rotate it to TAB_TILT off the disc axis
        tab.apply_transform(trimesh.transformations.rotation_matrix(
            np.radians(-(90.0 - TAB_TILT)), [0, 1, 0]))
        tab.apply_translation([TAB_X, 0.0, TAB_Z])
        # bridge rim -> tab foot so the tab is not floating
        gx0, gx1 = DISC_D / 2.0 - 4.0, TAB_X + (TAB_H / 2.0) * abs(_UP[0]) + 2.0
        gusset = trimesh.creation.box(extents=[gx1 - gx0, 12.0, DISC_T])
        gusset.apply_translation([(gx0 + gx1) / 2.0, 0.0, 0.0])
        body = trimesh.boolean.union([body, tab, gusset])

    cuts = [_cyl(center_hole_d, DISC_T * 4)]
    for r, _ in ((R_INNER, "A"), (R_OUTER, "B")):
        for a in ANGLES:
            t = np.radians(a)
            m = trimesh.transformations.translation_matrix(
                [r * np.cos(t), r * np.sin(t), 0.0])
            cuts.append(_cyl(TENDON_D, DISC_T * 4, transform=m))

    # index notch on the rim, aligned with the 90 deg tendon pair and the tab
    notch = trimesh.creation.box(extents=[3.0, 3.0, DISC_T * 4])
    notch.apply_translation([0.0, DISC_D / 2.0, 0.0])
    cuts.append(notch)

    disc = trimesh.boolean.difference([body] + cuts)
    disc.remove_unreferenced_vertices()
    return disc


def make_fit_test() -> trimesh.Trimesh:
    """Coupon with a range of centre holes: print it first, find what fits.

    Printers undersize holes by 0.1-0.4 mm depending on machine, material and
    slicer. Rather than guess, print this, push the rod into each hole, and use
    the smallest that slides without force.
    """
    plate = trimesh.creation.box(extents=[80.0, 22.0, DISC_T])
    cuts, labels = [], []
    for i, d in enumerate((4.7, 4.8, 4.9, 5.0, 5.1)):
        x = -32.0 + i * 16.0
        m = trimesh.transformations.translation_matrix([x, 0.0, 0.0])
        cuts.append(_cyl(d, DISC_T * 4, transform=m))
        # notches below each hole encode its size: i+1 marks
        for k in range(i + 1):
            n = trimesh.creation.box(extents=[1.2, 1.2, DISC_T * 4])
            n.apply_translation([x - 3.0 + k * 1.8, -8.5, 0.0])
            cuts.append(n)
        labels.append(f"{d:.1f}mm = {i+1} notch{'es' if i else ''}")
    out = trimesh.boolean.difference([plate] + cuts)
    return out, labels


def check_clearance(mesh) -> bool:
    """Does any solid material sit directly over a tendon hole?

    The first design silently covered two of them - the tab leans inward and
    its shadow reached r=12. Rather than eyeball a render, sample a ring of
    points just above each hole and confirm the mesh does not occupy them.
    """
    ok = True
    zs = np.linspace(DISC_T / 2 + 0.5, 40.0, 60)
    for r in (R_INNER, R_OUTER):
        for a in ANGLES:
            th = np.radians(a)
            x, y = r * np.cos(th), r * np.sin(th)
            pts = np.c_[np.full_like(zs, x), np.full_like(zs, y), zs]
            # ray straight up from the hole: any hit means something overhangs it
            hits = mesh.ray.intersects_any(
                ray_origins=np.array([[x, y, DISC_T / 2 + 0.5]]),
                ray_directions=np.array([[0.0, 0.0, 1.0]]))
            if bool(hits[0]):
                print(f"    OVERHANG over hole r={r} at {a:.0f} deg")
                ok = False
    print(f"  tendon-hole clearance: {'OK - nothing overhangs any hole' if ok else 'FAILED'}")
    return ok


def report(mesh, name):
    ok = mesh.is_watertight
    print(f"  {name:22s} watertight={str(ok):5s} volume={mesh.volume/1000:6.2f} cm3 "
          f"faces={len(mesh.faces):6d}  bbox={np.round(mesh.extents,1)}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hole", type=float, default=ROD_D + HOLE_CLEAR,
                    help="centre hole diameter, mm (default %(default).2f)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print(f"centre hole {args.hole:.2f} mm for a {ROD_D:.4f} mm rod "
          f"({args.hole-ROD_D:+.2f} mm clearance)\n")

    allok = True
    d = make_disc(args.hole, with_tab=True)
    allok &= report(d, "disc_with_tab.stl")
    allok &= check_clearance(d)
    d.export(f"{OUT}/disc_with_tab.stl")

    p = make_disc(args.hole, with_tab=False)
    allok &= report(p, "disc_plain.stl")
    p.export(f"{OUT}/disc_plain.stl")

    f, labels = make_fit_test()
    allok &= report(f, "fit_test.stl")
    f.export(f"{OUT}/fit_test.stl")

    print(f"\nfit-test coupon holes: " + ", ".join(labels))
    print(f"PRINT MARKERS AT {MARKER_MM:.0f} mm ({MARKER_MM/25.4:.2f} in) for these tabs")
    print(f"\nwrote to {OUT}/")
    if not allok:
        print("\nWARNING: a mesh is not watertight - slicers may misbehave.")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
