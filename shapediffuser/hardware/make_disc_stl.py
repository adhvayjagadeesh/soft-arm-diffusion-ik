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
#
# This is computed from the tendon diameter ACTUALLY being cut, not from the
# nominal constant: --shrink widens every hole, which pushes the outer tendon
# circle's edge further out, and a tab placed for 2.0 mm holes would start to
# shadow 2.8 mm ones.
_t = np.radians(TAB_TILT)
_UP = np.array([-np.cos(_t), 0.0, np.sin(_t)])       # in-plane "up" of the tab


def _tab_pose(tendon_d: float):
    clear_r = R_OUTER + tendon_d / 2.0 + 1.0         # must not shadow a hole
    return (clear_r + (TAB_H / 2.0) * abs(_UP[0]),   # TAB_X
            (TAB_H / 2.0) * abs(_UP[2]))             # TAB_Z

OUT = os.path.dirname(os.path.abspath(__file__)) + "/stl"


def _cyl(d, h, sections=96, transform=None):
    c = trimesh.creation.cylinder(radius=d / 2.0, height=h, sections=sections)
    if transform is not None:
        c.apply_transform(transform)
    return c


def make_disc(center_hole_d: float, with_tab: bool = True,
              tendon_d: float = TENDON_D) -> trimesh.Trimesh:
    body = _cyl(DISC_D, DISC_T)
    TAB_X, TAB_Z = _tab_pose(tendon_d)

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
            cuts.append(_cyl(tendon_d, DISC_T * 4, transform=m))

    # Index notch on the rim at 90 deg, on the tendon pair. It is 90 deg FROM
    # the tab, which sits at 0 deg - do not read it as pointing at the tab.
    # Its job is only that every disc is keyed identically, so lining the
    # notches up on the rod lines the tabs up too.
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


def make_fit_test(rod_d: float, sizes=None) -> trimesh.Trimesh:
    """Coupon with a range of centre holes: print it first, find what fits.

    Printers undersize holes by 0.1-0.4 mm depending on machine, material and
    slicer. Rather than guess, print this, push the rod into each hole, and use
    the smallest that slides without force.

    Two design rules here were both learned by the coupon failing in the field.

    RULE 1: the largest hole must be loose enough that SOMETHING always fits.
    The first version ran rod+0.05 to rod+0.45 and returned "nothing fits" - a
    null result indistinguishable from a scaling error, which is how it was
    misread. It was widened to rod+0.85 and returned "nothing fits" AGAIN, on
    a machine whose real offset is ~0.78 mm: 3-5x the 0.15-0.25 mm that
    textbooks quote, and enough that a nominally 4.03 mm hole would not pass a
    3.29 mm rod.

    So do not trust a default range. --fit-holes takes explicit sizes, and the
    right move after any "nothing fits" is to re-bracket around the smallest
    offset the failure PROVES, rather than to widen by another guess. Two
    print cycles were spent learning that a coupon which fails tells you only
    a lower bound.

    An offset that large is a symptom, not just a number: holes that small
    alongside parts printing 0.2 mm TALL is the signature of over-extrusion.
    Compensating it in the model works, but a flow calibration fixes the
    cause and makes every future part come out closer to nominal.

    RULE 2: the coupon must say what rod it is for. The printed part carried no
    record of its own design rod, so a coupon cut for 1/8 in was tested against
    a 3/16 in rod - a 1.14 mm mismatch that no print setting could explain, and
    which cost a print cycle to diagnose. The tally notches along the TOP edge
    now encode the rod in 32nds of an inch: 4 notches = 1/8 in, 6 = 3/16 in.
    Size tallies stay on the BOTTOM edge and are narrower.
    """
    if sizes is None:
        sizes = tuple(round(rod_d + FIT_START + FIT_STEP * k, 2)
                      for k in range(N_FIT))
    sizes = tuple(sizes)
    span = (len(sizes) - 1) * FIT_PITCH
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


PLATE_HOLE = 6.35      # 1/4 in, as the base plate was actually drilled
BUSH_SHRINK = 0.15     # printed outer surfaces run large; undersize to drop in


def make_bushing(center_hole_d: float, plate_hole_d: float = PLATE_HOLE,
                 plate_t: float = 6.35) -> trimesh.Trimesh:
    """Adapter so an oversized base-plate hole still holds the rod square.

    The base plate's centre hole is a clearance hole - the rod is epoxied into
    it and the DISCS are what hold it perpendicular. That works, but the plate
    sets the rod's angle at the one place where an error is multiplied along
    the whole arm, and a 1/8 in rod in a 1/4 in hole can lean ~3.2 mm before
    the epoxy grabs.

    This bushing drops into the drilled hole and restores the same slip fit the
    discs use. It costs one 10-minute print and makes the base joint the most
    accurately located one on the arm rather than the least. It also means a
    hole drilled for one rod does not have to be re-drilled for another - the
    bushing absorbs the difference.
    """
    body = _cyl(plate_hole_d - BUSH_SHRINK, plate_t + 0.5)
    flange = _cyl(plate_hole_d + 6.0, 2.0)
    flange.apply_translation([0.0, 0.0, (plate_t + 0.5) / 2.0 + 1.0])
    bore = _cyl(center_hole_d, (plate_t + 8.0) * 2)
    return trimesh.boolean.difference([trimesh.boolean.union([body, flange]), bore])


def check_clearance(mesh, tendon_d: float = TENDON_D) -> bool:
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


# Output names carry their PRINT ORDER, because the order is not optional and
# getting it wrong is expensive: the six-disc plate is 92 g and three hours,
# and it must not be printed until a coupon and a single disc have confirmed
# the bore. A directory listing now states the sequence by itself.
NAMES = {
    "fit_test":      "ARM-1-FIT-TEST",
    "disc_with_tab": "ARM-2-SINGLE-DISC",
    "plate_6_discs": "ARM-3-SIX-DISCS",
    "base_bushing":  "ARM-4-BASE-BUSHING",
    "disc_plain":    "ARM-X-DISC-NO-TAB",
}


def _write(mesh, stem, want_stl=False):
    """Write .3mf, and .stl only on request.

    3MF is the right input for Creality Print / OrcaSlicer: it carries units
    and a proper scene graph. STL carries NO units, which on this project
    already produced one round of "is the model scaled?" - so it is no longer
    written by default. Pass --stl for a tool that cannot read 3MF.
    """
    name = NAMES.get(stem, stem)
    mesh.export(f"{OUT}/{name}.3mf")
    if want_stl:
        mesh.export(f"{OUT}/{name}.stl")
    return name


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
    ap.add_argument("--plate-hole", type=float, default=PLATE_HOLE,
                    help="base plate centre hole as DRILLED, mm (6.35 = 1/4 in)")
    ap.add_argument("--plate-t", type=float, default=6.35,
                    help="base plate thickness, mm")
    ap.add_argument("--shrink", type=float, default=0.0,
                    help="YOUR PRINTER's hole offset, mm. Every hole is cut "
                         "this much oversize so it comes out on size. Measure "
                         "it with fit_test, do not guess.")
    ap.add_argument("--fit-holes", type=str, default=None,
                    help="explicit coupon hole sizes, comma separated, when "
                         "the default range does not bracket your printer")
    ap.add_argument("--stl", action="store_true",
                    help="also write .stl (unitless; only for tools that "
                         "cannot read 3MF)")
    ap.add_argument("--plain", action="store_true",
                    help="also write the no-tab disc (spares, not in the build)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # The bore must clear the rod's FATTEST measured point, not its average -
    # pultruded rod is out of round and the disc has to pass over all of it.
    hole = args.hole if args.hole is not None else args.rod + HOLE_CLEAR
    hole += args.shrink
    tendon = TENDON_D + args.shrink

    print(f"centre hole {hole:.2f} mm nominal for a {args.rod:.3f} mm rod")
    if args.shrink:
        print(f"  = {args.rod:.3f} rod + {HOLE_CLEAR:.2f} slip fit "
              f"+ {args.shrink:.2f} printer offset")
        print(f"  EXPECTED once printed: {hole-args.shrink:.2f} mm "
              f"({hole-args.shrink-args.rod:+.2f} mm on the rod)")
        print(f"  tendon holes {TENDON_D:.1f} -> {tendon:.2f} mm nominal, "
              f"{TENDON_D:.2f} mm once printed")
    else:
        print(f"  ({hole-args.rod:+.2f} mm clearance, NO printer offset applied)")
    if args.rod > 4.0:
        print("  WARNING: a 3/16 in backbone is ~5x stiffer than 1/8 in. These")
        print("  servos will only bend it ~20-40 deg per section, leaving almost")
        print("  no redundancy for the experiment to study.")
    print(f"  tendon radii {R_INNER} / {R_OUTER} mm, disc {DISC_D} mm\n")

    allok = True
    written = []

    fit_sizes = ([float(x) for x in args.fit_holes.split(",")]
                 if args.fit_holes else None)
    f, labels, n_id = make_fit_test(args.rod, fit_sizes)
    allok &= report(f, "fit_test")
    written.append((_write(f, "fit_test", args.stl), "print FIRST, ~15 min"))

    d = make_disc(hole, with_tab=True, tendon_d=tendon)
    allok &= report(d, "disc_with_tab")
    allok &= check_clearance(d, tendon)
    written.append((_write(d, "disc_with_tab", args.stl),
                    "print SECOND, ~30 min - confirms the bore"))

    # six discs arranged on one plate, ready to slice in a single job
    plate = []
    for i in range(6):
        c = d.copy()
        c.apply_translation([(i % 3) * 80.0 - 80.0, (i // 3) * 65.0 - 32.5, 0.0])
        plate.append(c)
    six = trimesh.util.concatenate(plate)
    allok &= report(six, "plate_6_discs")
    written.append((_write(six, "plate_6_discs", args.stl),
                    "print THIRD, ~3 h - only after the bore is confirmed"))

    b = make_bushing(hole, args.plate_hole, args.plate_t)
    allok &= report(b, "base_bushing")
    written.append((_write(b, "base_bushing", args.stl),
                    "print with the discs, <1 min"))

    if args.plain:
        p = make_disc(hole, with_tab=False, tendon_d=tendon)
        allok &= report(p, "disc_plain")
        written.append((_write(p, "disc_plain", args.stl), "spare, not in the build"))

    print("\nfit-test coupon:")
    for s in labels:
        print(f"    {s}")
    print(f"  TOP edge carries {n_id} notches = {n_id}/32 in = the rod this coupon is cut for.")
    print(f"  Check that against the rod with calipers BEFORE concluding anything")
    print(f"  about the print: {args.rod:.3f} mm is what these holes assume.")
    print(f"PRINT MARKERS AT {MARKER_MM:.0f} mm ({MARKER_MM/25.4:.2f} in) for these tabs")

    print(f"\nwrote to {OUT}/  IN PRINT ORDER:")
    for name, when in written:
        print(f"  {name+'.3mf':26} {when}")
    if not args.stl:
        print("\n  (.stl not written - STL carries no units and cost this project a"
              "\n   round of 'is it scaled?'. Use --stl only for a tool that needs it.)")
    if not allok:
        print("\nWARNING: a mesh is not watertight - slicers may misbehave.")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
