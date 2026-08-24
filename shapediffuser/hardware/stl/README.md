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

| source | tolerance | rod received | clearance in a 3.44 mm printed bore |
|---|---|---|---|
| TAP | +/-0.005 in | 3.048 - 3.302 mm | +0.14 to +0.39 mm, always fits |
| McMaster FRP rod | +/-0.010 in | 2.921 - 3.429 mm | +0.01 mm at worst - SEIZES |

The looser rod can arrive with 0.01 mm of clearance, which is not a slip fit
at all. It also doubles the stiffness spread, 0.72-1.36x against 0.85-1.17x,
and that spread lands directly on the curvature gain being fitted.

The rod actually received measured 3.19 / 3.20 / 3.22 / 3.29 mm - 0.10 mm out
of round. The bore is sized from the FATTEST point: the disc has to pass over
all of it, not over the average.

Higher glass content makes the rod stiffer, not weaker: 65-75% glass suggests
unidirectional roving, E ~ 40-45 GPa rather than the ~20 GPa of fabric-
reinforced stock. That still clears the 90 deg per section the 22 mm tendon
radius was sized for. Anything up to E = 50 GPa works at this diameter.

## MEASURE THE ROD BEFORE GENERATING DISCS

Even at +/-0.005 in the rod spans a quarter of a millimetre, which is larger
than the slip fit being designed. Do not assume 3.175 mm.

    caliper the rod at several points, rotating at each  (pultruded rod is
    slightly out of round and varies along its length; take the largest)

    python ../make_disc_stl.py --rod 3.29 --shrink 0.78
    # print ARM-1-FIT-TEST, find the smallest hole the rod slides into,
    # then re-run with the --shrink your printer actually shows

## Printing

Print `ARM-1-FIT-TEST.3mf` FIRST.

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

    python ../make_disc_stl.py --rod 3.29 --shrink 0.78

`--shrink` is YOUR PRINTER's hole offset and it applies to every hole, not
just the bore. Measured on this K1 Max with a 0.4 nozzle, from three gauge
failures: a 1/16 in bit would not enter a 2.00 mm hole, a 1/8 in bit would not
enter a 3.32 mm hole, and a 3.29 mm rod would not enter a 4.03 mm hole. One
constant offset of 0.75-0.80 mm satisfies all three. 0.78 is used.

That is 3-5x the 0.15-0.25 mm usually quoted. Alongside parts measuring 0.2 mm
TALL against nominal, the signature is OVER-EXTRUSION - too much plastic per
mm of path. Compensating in the model works, and is what --shrink does, but a
flow calibration fixes the cause and is worth doing before the six-disc batch.

You do not need the rod to measure the offset - caliper the printed coupon
holes and compare against the table above.

If you are using 3/16 in anyway, regenerate the coupon too: `--rod 4.7625`.

## Files

| file | what | when |
|---|---|---|
| `ARM-1-FIT-TEST.3mf` | hole-size coupon | FIRST, ~15 min |
| `ARM-2-SINGLE-DISC.3mf` | one disc, confirms the bore | SECOND, ~30 min |
| `ARM-3-SIX-DISCS.3mf` | all six, one job | THIRD, ~3 h - only after the bore is confirmed |
| `ARM-4-BASE-BUSHING.3mf` | adapts the 1/4 in plate hole | with the discs, <1 min |

Names carry the print order because the order is not optional: the six-disc
plate is 92 g and three hours, and must not be printed until a coupon and a
single disc have confirmed the bore.

`.stl` is no longer written. STL carries no units, which already cost this
project one round of "is the model scaled?". Pass `--stl` if some tool needs
it; `--plain` adds a no-tab disc for spares.

## STL is not a printable file

A 3D printer cannot read STL from a USB stick. STL describes a shape; the
printer needs G-code, which is toolpaths for one specific machine, material
and nozzle. Slicing software converts one to the other:

    STL  ->  slicer  ->  .gcode  ->  printer

### Creality K1 Max

**Use the .3mf files.** They carry units and a proper scene graph and avoid
STL's vertex-indexing quirks. STL is not written at all any more - it carries
no units, and that ambiguity already cost this project a round of "is the
model scaled?". All files are well under 200 KB, far below the point where
mesh complexity could stutter at high speed.

Use **Creality Print** (official, ships a K1 Max profile) or **OrcaSlicer**
(better quality, has a K1 Max profile too). Either one:

1. Open the slicer, select **K1 Max** as the printer.
2. If the hotend has been swapped for a Micro Swiss, set the nozzle diameter
   to match the installed one - a profile expecting 0.4 mm on a 0.6 mm nozzle
   under-extrudes badly, and vice versa.
3. Import the .3mf, confirm the size readout, **Slice**, then **Export
   G-code** to the USB stick.
4. Print from the printer's own file browser.

The K1 Max is also networked - slicing then sending over LAN avoids the USB
stick entirely, and is the easier path once it is set up.

## Print settings

### Calibrate flow FIRST

A 0.78 mm hole offset alongside parts measuring 0.2 mm TALL is over-extrusion,
and no slicer checkbox fixes it. In OrcaSlicer: **Calibration -> Flow rate ->
Pass 1**, print, pick the smoothest top surface, apply, then **Pass 2**. Half
an hour, and it does more for dimensional accuracy than every setting below
combined.

### CHANGING ANY SETTING INVALIDATES --shrink

The parts are drawn oversize for one specific machine configuration. Turn on
Precise wall, or fix the flow, and holes come out near nominal - which would
leave a 4.22 mm bore on a 3.29 mm rod, 0.93 mm of slop, and discs that flop
instead of sitting square. Either print exactly as generated, or change the
settings and RE-RUN the fit test before regenerating.

### Settings that matter here

| setting | value | why |
|---|---|---|
| Precise wall | **ON** | corrects outer-wall placement; attacks the bore error directly |
| Wall loops | **3** | the bore is defined by walls, not infill |
| Skirt loops | **2** | with 0 the nozzle is not pressure-stable when the part starts |
| Sparse infill | 25% | supports the walls; minor on a part this solid |
| Prime tower | **OFF** | single material |
| xy hole/contour compensation | **0** | keep compensation in the model, not the profile - a fresh profile silently resizes every part |

- **0.2 mm layers, PLA.** Makes the 3 mm disc exactly 15 layers. Do not go
  finer: with over-extrusion, more passes make holes worse.
- **Lay the disc flat** on the bed, tab pointing up.
- **No supports needed.** The tab underside sits at 65 deg from horizontal and
  its base is fully gusseted; verified there is zero surface below 45 deg.
  Supports would weld scars onto the marker face.
- **Print sequence: by layer.** By-object with six discs risks head collisions.
- No brim needed; the 56 mm footprint is stable.
- All six discs fit on one plate - `ARM-3-SIX-DISCS.3mf` is pre-arranged at
  231 x 121 mm on the 300 x 300 mm bed.

### Speed, for these parts specifically

The K1 Max will happily run 600 mm/s, but **do not chase top speed here.**
This design leans on dimensional accuracy in two places:

- the centre bore is a slip fit, and it is what holds each disc square to
  the backbone - the whole reason for printing rather than drilling
- the tendon holes must not close up

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
| disc | 56.0 mm dia x 3.0 mm, + 8 mm collar 4 mm tall |
| bore bearing length | 6.4 mm straight, then a 0.6 mm lead-in chamfer |
| centre bore | 4.22 mm nominal -> 3.44 mm printed, for a 3.29 mm rod |
| tendon holes | 2.78 mm nominal -> 2.0 mm printed, six of them |
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
2. **The snug bore is doing real work, and the collar doubles it.** Permissible
   tilt is atan(clearance / bearing length), so bore LENGTH matters as much as
   fit. On disc thickness alone that is 3 mm and 2.9 deg, which throws the
   outer tendon holes +/-1.1 mm - about the whole vision noise floor, and
   random per disc because each is bonded at its own angle, so no gain fit
   removes it. The collar takes the bearing to 6.4 mm, the tilt to 1.3 deg,
   and the scatter to +/-0.5 mm. Do not open the bore up to make assembly
   easier; that is what the lead-in chamfer is for.
3. **Collars point UP, same side as the tabs.** Printed that way and assembled
   that way.
