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

Disc diameter is 56 mm, chosen from a force budget rather than convenience.

An XL330 at half stall pulls ~95 N through an M3-standoff capstan. Bending a
rod to curvature k needs moment EI*k, supplied by tension x tendon radius, so
    r_tendon >= EI * k / T
With r_outer = 22 mm this covers 90 deg per section on a 1/8 in fiberglass
backbone even at the pessimistic E = 40 GPa. Going wider buys little: cable
travel grows with r (more spool wraps, and stacked wraps break the linear
angle-to-length relation), while mass and disc-to-disc collision grow too.

THE BACKBONE MATTERS FAR MORE THAN THE DISC. Stiffness goes as diameter^4, so
a 3/16 in rod is ~5x stiffer than 1/8 in. At r_outer = 22 mm the same servos
reach ~150 deg per section on 1/8 in but only ~20-40 deg on 3/16 in - a nearly
rigid arm with almost no redundancy to resolve. Use 1/8 in. No practical disc
diameter rescues a 3/16 in backbone; it would need r_tendon of 55-110 mm.

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
DISC_D = 56.0          # mm outer diameter, set by the force budget above
DISC_T = 3.0           # mm thickness
ROD_D = 3.175          # 1/8 in fiberglass rod (see --rod for 3/16 in)
HOLE_CLEAR = 0.15      # slip fit; printers shrink holes, see --fit-test

TENDON_D = 2.0         # generous for 0.79 mm cable; sub-2 mm holes print badly
R_INNER = 12.0         # section A tendon circle
R_OUTER = 22.0         # section B tendon circle
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
        # full tab width: a narrow gusset left the tab's lower corners floating
        # 1.5 mm above the bed, which droops without supports
        gusset = trimesh.creation.box(extents=[gx1 - gx0, TAB_W, DISC_T])
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


N_FIT = 6              # holes on the coupon
FIT_START = 0.10       # smallest clearance over the rod
FIT_STEP = 0.15        # so the largest hole is rod + 0.85 mm
FIT_PITCH = 18.0


def make_fit_test(rod_d: float) -> trimesh.Trimesh:
    """Coupon with a range of centre holes: print it first, find what fits.

    Printers undersize holes by 0.1-0.4 mm depending on machine, material and
    slicer. Rather than guess, print this, push the rod into each hole, and use
    the smallest that slides without force.

    Two design rules here were both learned by the coupon failing in the field.

    RULE 1: the largest hole must be loose enough that SOMETHING always fits.
    The first version ran rod+0.05 to rod+0.45, so a printer that undersized by
    more than 0.45 mm reported "nothing fits" - a null result indistinguishable
    from a scaling error, which is exactly how it was misread. The range now
    runs to rod+0.85, past any plausible single-wall shrinkage, so a coupon
    where nothing fits is positive evidence that the ROD is wrong, not the
    print.

    RULE 2: the coupon must say what rod it is for. The printed part carried no
    record of its own design rod, so a coupon cut for 1/8 in was tested against
    a 3/16 in rod - a 1.14 mm mismatch that no print setting could explain, and
    which cost a print cycle to diagnose. The tally notches along the TOP edge
    now encode the rod in 32nds of an inch: 4 notches = 1/8 in, 6 = 3/16 in.
    Size tallies stay on the BOTTOM edge and are narrower.
    """
    sizes = tuple(round(rod_d + FIT_START + FIT_STEP * k, 2) for k in range(N_FIT))
    span = (N_FIT - 1) * FIT_PITCH
    plate = trimesh.creation.box(extents=[span + 22.0, 26.0, DISC_T])

    def tally(n, x0, y, w, pitch):
        out = []
        for k in range(n):
            b = trimesh.creation.box(extents=[w, w, DISC_T * 4])
            b.apply_translation([x0 + k * pitch, y, 0.0])
            out.append(b)
        return out

    cuts, labels = [], []
    for i, d in enumerate(sizes):
        x = -span / 2.0 + i * FIT_PITCH
        m = trimesh.transformations.translation_matrix([x, 0.0, 0.0])
        cuts.append(_cyl(d, DISC_T * 4, transform=m))
        # size tally, bottom edge. 1.3 mm marks on a 2.6 mm pitch: the previous
        # 0.6 mm gap was under two extrusion widths and blurred into one blob.
        w, pitch = 1.3, 2.6
        cuts += tally(i + 1, x - ((i + 1) * pitch - pitch) / 2.0, -9.5, w, pitch)
        labels.append(f"{d:.2f} mm = {i+1} bottom notch{'es' if i else ''}")

    # rod ID, top edge: rod diameter in 32nds of an inch
    n_id = int(round(rod_d / (25.4 / 32.0)))
    cuts += tally(n_id, -span / 2.0 - 6.0, 9.5, 2.0, 3.4)

    out = trimesh.boolean.difference([plate] + cuts)
    return out, labels, n_id


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


def _write(mesh, stem):
    """Write both .3mf and .stl.

    3MF is the better input for Creality Print / OrcaSlicer: it carries units
    and a proper scene graph, and avoids the vertex-indexing quirks of STL.
    STL is kept for any tool that will not read 3MF.
    """
    mesh.export(f"{OUT}/{stem}.3mf")
    mesh.export(f"{OUT}/{stem}.stl")


def report(mesh, name):
    ok = mesh.is_watertight
    print(f"  {name:22s} watertight={str(ok):5s} volume={mesh.volume/1000:6.2f} cm3 "
          f"faces={len(mesh.faces):6d}  bbox={np.round(mesh.extents,1)}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hole", type=float, default=None,
                    help="centre hole diameter, mm (default: rod + clearance)")
    ap.add_argument("--rod", type=float, default=ROD_D,
                    help="backbone diameter, mm. 3.175 = 1/8 in (recommended), "
                         "4.7625 = 3/16 in (too stiff for these servos)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    hole = args.hole if args.hole is not None else args.rod + HOLE_CLEAR

    print(f"centre hole {hole:.2f} mm for a {args.rod:.4f} mm rod "
          f"({hole-args.rod:+.2f} mm clearance)")
    if args.rod > 4.0:
        print("  WARNING: a 3/16 in backbone is ~5x stiffer than 1/8 in. These")
        print("  servos will only bend it ~20-40 deg per section, leaving almost")
        print("  no redundancy for the experiment to study.")
    print(f"  tendon radii {R_INNER} / {R_OUTER} mm, disc {DISC_D} mm\n")

    allok = True
    d = make_disc(hole, with_tab=True)
    allok &= report(d, "disc_with_tab")
    allok &= check_clearance(d)
    _write(d, "disc_with_tab")

    p = make_disc(hole, with_tab=False)
    allok &= report(p, "disc_plain")
    _write(p, "disc_plain")

    f, labels, n_id = make_fit_test(args.rod)
    allok &= report(f, "fit_test")
    _write(f, "fit_test")

    # six discs arranged on one plate, ready to slice in a single job
    plate = []
    for i in range(6):
        c = d.copy()
        c.apply_translation([(i % 3) * 80.0 - 80.0, (i // 3) * 65.0 - 32.5, 0.0])
        plate.append(c)
    six = trimesh.util.concatenate(plate)
    allok &= report(six, "plate_6_discs")
    _write(six, "plate_6_discs")

    print("\nfit-test coupon:")
    for s in labels:
        print(f"    {s}")
    print(f"  TOP edge carries {n_id} notches = {n_id}/32 in = the rod this coupon is cut for.")
    print(f"  Check that against the rod with calipers BEFORE concluding anything")
    print(f"  about the print: {args.rod:.3f} mm is what these holes assume.")
    print(f"PRINT MARKERS AT {MARKER_MM:.0f} mm ({MARKER_MM/25.4:.2f} in) for these tabs")
    print(f"\nwrote to {OUT}/")
    if not allok:
        print("\nWARNING: a mesh is not watertight - slicers may misbehave.")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
