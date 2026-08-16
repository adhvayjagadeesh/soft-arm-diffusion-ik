# Printed spacer discs

## Read this first: use a 1/8 in backbone, not 3/16 in

Stiffness scales as diameter^4, so a 3/16 in rod is ~5x stiffer than 1/8 in.
With these servos and a 22 mm tendon radius that is the difference between
~150 deg of bend per section and ~20-40 deg - a nearly rigid arm with almost
no redundancy left for the experiment to study. No disc diameter fixes it:
a 3/16 in backbone would need a tendon radius of 55-110 mm.

McMaster 8543K31 is 1/8 in x 10 ft, about $9.

If you must use 3/16 in, regenerate with `--rod 4.7625` and expect a stiff arm.

## Printing

Print `fit_test.3mf` FIRST.

**The coupon is cut for a 1/8 in (3.175 mm) rod.** Its six holes run 3.27 to
4.03 mm. A 3/16 in rod is 4.7625 mm and will not enter any of them - that is
the coupon telling you the rod is wrong, not the printer.

Confirm the rod before you conclude anything: the **top** edge carries a tally
in 32nds of an inch, so **4 notches = 1/8 in**, 6 would mean 3/16 in. Caliper
the rod and check it matches.

Then push the rod into each hole and use the smallest it slides into without
force. Hole size is the tally on the **bottom** edge:

| bottom notches | hole |
|---|---|
| 1 | 3.27 mm |
| 2 | 3.42 mm |
| 3 | 3.57 mm |
| 4 | 3.72 mm |
| 5 | 3.88 mm |
| 6 | 4.03 mm |

The discs ship with a 3.32 mm hole. If a different hole fits best, regenerate
the discs with it before printing all six:

    python ../make_disc_stl.py --hole 3.57

You do not need the rod to check the printer. Caliper the printed holes
directly and compare against the table - the difference is your machine's hole
offset, and it should be 0.1-0.3 mm undersize. If it is under 0.45 mm the
already-printed 3.32 mm discs are fine.

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
| centre hole | 3.32 mm (slip fit on a 3.175 mm / 1-8 in rod) |
| tendon holes | 2.0 mm, six of them |
| section A circle | r = 12.0 mm, at 90/210/330 deg |
| section B circle | r = 22.0 mm, same headings |
| marker tab | 40 x 40 mm face, tilted **65 deg** from the disc axis |
| index notch | on the rim at 90 deg, aligned with the tab |

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
