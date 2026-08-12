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

Print `fit_test.stl` FIRST. Push the fiberglass rod into each hole and use the
smallest one it slides into without force; hole size is encoded by the notches
below it. The coupon spans 4.7-5.1 mm, sized for a 3/16 in rod - for the
recommended 1/8 in rod the target is 3.32 mm, so use the coupon only to learn
your printer's offset, then apply it:

    python ../make_disc_stl.py --hole 3.4

## Files

| file | what | qty |
|---|---|---|
| `fit_test.stl` | hole-size coupon, print this first | 1 |
| `disc_with_tab.stl` | spacer disc with the marker tab built in | 6 |
| `disc_plain.stl` | same disc, no tab (spares / experiments) | as needed |

## STL is not a printable file

A 3D printer cannot read STL from a USB stick. STL describes a shape; the
printer needs G-code, which is toolpaths for one specific machine, material
and nozzle. Slicing software converts one to the other:

    STL  ->  slicer  ->  .gcode  ->  printer

### Creality K1 Max

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
- All six discs fit on one plate - the K1 Max bed is 300 x 300 mm and each
  part is 71 x 56 mm.

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
