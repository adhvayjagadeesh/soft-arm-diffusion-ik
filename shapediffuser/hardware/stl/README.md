# Printed spacer discs

## Read this first: use a 1/8 in backbone, not 3/16 in

Stiffness scales as diameter^4, so a 3/16 in rod is 5.1x stiffer than 1/8 in.
With these servos and a 22 mm tendon radius that is 90-150 deg of bend per
section against 20-30 deg. Measured consequence: the reachable workspace
shrinks 50x, from 67,000 cm3 to 1,400 cm3, and collapses to a 30 mm-deep
shell in which the tip never leaves full extension - the arm pivots instead
of reaching.

Distinct solutions to one target stay resolvable on the stiff rod (11 mm
apart against a 1.5 mm sensing floor), so this is a much weaker arm rather
than a broken experiment. But the redundancy the study exists to characterize
is largely gone. No disc diameter fixes it: a 3/16 in backbone needs a tendon
radius of 67-107 mm, meaning a 145-225 mm disc.

If you must use 3/16 in, regenerate with `--rod 4.7625` and expect a stiff arm.

## Sourcing the backbone

TAP Plastics pultruded fiberglass round rod, 1/8 in, 3 ft length (the 3 ft cut
carries no oversize fee; 3 ft is one arm plus a full spare on a 305 mm free
length). 65-75% glass. Bay Area stores carry it over the counter.

Buy on **diameter tolerance**, not on price. The bore is a slip fit and it is
what holds each disc square, so the rod's actual diameter sets whether the
design works:

| source | tolerance | rod received | vs a 3.32 mm bore |
|---|---|---|---|
| TAP | +/-0.005 in | 3.048 - 3.302 mm | fits even at worst case |
| McMaster FRP rod | +/-0.010 in | 2.921 - 3.429 mm | worst case does NOT fit |

The looser rod can arrive 0.11 mm too fat for the nominal bore before printer
shrinkage is counted at all. It also doubles the stiffness spread, 0.72-1.36x
against 0.85-1.17x, and that spread lands directly on the curvature gain being
fitted.

Higher glass content makes the rod stiffer, not weaker: 65-75% glass suggests
unidirectional roving, E ~ 40-45 GPa rather than the ~20 GPa of fabric-
reinforced stock. That still clears the 90 deg per section the 22 mm tendon
radius was sized for. Anything up to E = 50 GPa works at this diameter.

## MEASURE THE ROD BEFORE GENERATING DISCS

Even at +/-0.005 in the rod spans a quarter of a millimetre, which is larger
than the slip fit being designed. Do not assume 3.175 mm.

    caliper the rod at several points, rotating at each  (pultruded rod is
    slightly out of round and varies along its length; take the largest)

    python ../make_disc_stl.py --rod 3.19      # whatever you measured
    # print fit_test.3mf, find the smallest hole the rod slides into
    python ../make_disc_stl.py --rod 3.19 --hole 3.57

## Printing

Print `fit_test.3mf` FIRST.

**The coupon is cut for a 1/8 in rod.** The current one is generated for a
measured 3.29 mm rod with a 0.78 mm printer offset, so its holes run 4.00 to
4.75 mm NOMINAL and should come out 3.22 to 3.97 mm once printed.

Those nominal sizes look absurdly large next to a 3.29 mm rod. They are not:
this machine cuts holes ~0.78 mm undersize, so the hole you draw and the hole
you get differ by more than the slip fit you are trying to hold.

Confirm the rod before you conclude anything: the **top** edge carries a tally
in 32nds of an inch, so **4 notches = 1/8 in**, 6 would mean 3/16 in. Caliper
the rod and check it matches.

Then push the rod into each hole and use the smallest it slides into without
force. Hole size is the tally on the **bottom** edge:

| bottom notches | nominal | expected once printed |
|---|---|---|
| 1 | 4.00 mm | 3.22 mm |
| 2 | 4.15 mm | 3.37 mm |
| 3 | 4.30 mm | 3.52 mm |
| 4 | 4.45 mm | 3.67 mm |
| 5 | 4.60 mm | 3.82 mm |
| 6 | 4.75 mm | 3.97 mm |

Regenerate the discs with the winning hole BEFORE printing all six:

    python ../make_disc_stl.py --rod 3.19 --hole 3.57

The default 3.32 mm bore assumes ~0.15 mm of shrinkage and that is optimistic
on at least one real printer. Measured on a K1 Max with a 0.4 nozzle: a
3.175 mm gauge pin would not enter a nominally 3.32 mm printed hole, and a
1.587 mm pin would not enter a nominally 2.0 mm one - so shrinkage exceeded
0.145 mm and 0.413 mm respectively. Small holes shrink proportionally more,
which is why the tendon holes lose so much more than the bore.

The tendon holes losing 0.4 mm does NOT matter: the cable is 0.79 mm and a
1.5 mm hole passes it easily. Only the bore has to hold a dimension.

You do not need the rod to check the printer - caliper the printed coupon
holes directly and compare against the table above. The difference is your
machine's hole offset. Expect 0.1-0.3 mm undersize; more than that is normal
on a fast machine and is exactly what the coupon exists to catch.

If you are using 3/16 in anyway, regenerate the coupon too: `--rod 4.7625`.

## Files

| file | what | qty |
|---|---|---|
| `fit_test.3mf` | hole-size coupon, print this first | 1 |
| `base_bushing.3mf` | adapts the 1/4 in base-plate hole to the rod | 1 |
| `plate_6_discs.3mf` | all six discs pre-arranged, one job | 1 print |
| `disc_with_tab.3mf` | a single disc, if printing individually | 6 |
| `disc_plain.3mf` | same disc, no tab (spares / experiments) | as needed |

## STL is not a printable file

A 3D printer cannot read STL from a USB stick. STL describes a shape; the
printer needs G-code, which is toolpaths for one specific machine, material
and nozzle. Slicing software converts one to the other:

    STL  ->  slicer  ->  .gcode  ->  printer

### Creality K1 Max

**Use the .3mf files.** They carry units and a proper scene graph and avoid
STL's vertex-indexing quirks. `.stl` versions are kept only for tools that
cannot read 3MF. All files are well under 200 KB, far below the point where
mesh complexity could stutter at high speed.

Use **Creality Print** (official, ships a K1 Max profile) or **OrcaSlicer**
(better quality, has a K1 Max profile too). Either one:

1. Open the slicer, select **K1 Max** as the printer.
2. If the hotend has been swapped for a Micro Swiss, set the nozzle diameter
   to match the installed one - a profile expecting 0.4 mm on a 0.6 mm nozzle
   under-extrudes badly, and vice versa.
3. Import the STL, arrange, **Slice**, then **Export G-code** to the USB stick.
4. Print from the printer's own file browser.

The K1 Max is also networked - slicing then sending over LAN avoids the USB
stick entirely, and is the easier path once it is set up.

## Print settings

- **0.2 mm layers, 20-25% infill, PLA.** ~5 g and ~30 min per disc.
- **Lay the disc flat** on the bed, tab pointing up.
- **No supports needed.** The tab underside sits at 65 deg from horizontal and
  its base is fully gusseted; verified there is zero surface below 45 deg.
- No brim needed; the 56 mm footprint is stable.
- All six discs fit on one plate - `plate_6_discs.3mf` is pre-arranged at
  231 x 121 mm on the 300 x 300 mm bed.

### Speed, for these parts specifically

The K1 Max will happily run 600 mm/s, but **do not chase top speed here.**
This design leans on dimensional accuracy in two places:

- the 3.32 mm centre hole is a slip fit, and it is what holds each disc
  square to the backbone - the whole reason for printing rather than drilling
- the 2 mm tendon holes must not close up

Ringing and corner bulge from high acceleration land directly on those
features. Keep outer walls slow (the 200-300 mm/s the K1 profiles use is
already fine) and let infill run fast. The parts are small enough that the
time saved by pushing wall speed is a couple of minutes per disc, against a
fit you cannot recover without reprinting.

Run the printer's input-shaping calibration before the batch - the K1 Max
does this from its own menu, and it is what makes fast moves dimensionally
honest.

## Geometry (all verified against the mesh)

| feature | value |
|---|---|
| disc | 56.0 mm dia x 3.0 mm |
| centre hole | 3.32 mm default - SET IT FROM THE FIT TEST, not from this table |
| tendon holes | 2.0 mm, six of them |
| section A circle | r = 12.0 mm, at 90/210/330 deg |
| section B circle | r = 22.0 mm, same headings |
| marker tab | 40 x 40 mm face at 0 deg, tilted **65 deg** from the disc axis |
| index notch | on the rim at 90 deg - on the tendon pair, 90 deg FROM the tab |

## Markers

**Print ArUco markers at 30 mm (1.18 in)** for these tabs - that leaves a 5 mm
quiet zone on the 40 mm face. Regenerate at that size rather than reusing the
1.25 in sheet.

## Two things that matter on assembly

1. **All six tabs must point the same way.** Staggering headings collapses
   simultaneous marker visibility from 69% to under 10% - measured, not
   guessed. Use the rim notch to align them on the rod.
2. **The snug centre hole is doing real work.** It holds each disc square to
   the rod by itself, which a drilled 1/4 in hole could not - that version
   allowed ~27 deg of tilt before the epoxy set. Do not open the hole up to
   make assembly easier.
