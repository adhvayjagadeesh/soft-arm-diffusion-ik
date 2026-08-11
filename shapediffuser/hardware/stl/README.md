# Printed spacer discs

Print `fit_test.stl` FIRST. Push the fiberglass rod into each hole and use the
smallest one it slides into without force; hole size is encoded by the notches
below it (1 notch = 4.7 mm, 5 notches = 5.1 mm). Then regenerate at that size:

    python ../make_disc_stl.py --hole 4.9

## Files

| file | what | qty |
|---|---|---|
| `fit_test.stl` | hole-size coupon, print this first | 1 |
| `disc_with_tab.stl` | spacer disc with the marker tab built in | 6 |
| `disc_plain.stl` | same disc, no tab (spares / experiments) | as needed |

## Print settings

- **0.2 mm layers, 20-25% infill, PLA.** ~4 g and ~25 min per disc.
- **Lay the disc flat** on the bed, tab pointing up. The tab needs support -
  enable supports, or print `disc_plain.stl` and glue tabs on.
- No brim needed; the 50 mm footprint is stable.

## Geometry (all verified against the mesh)

| feature | value |
|---|---|
| disc | 50.0 mm dia x 3.0 mm |
| centre hole | 4.91 mm (slip fit on a 4.7625 mm rod) |
| tendon holes | 2.0 mm, six of them |
| section A circle | r = 10.0 mm, at 90/210/330 deg |
| section B circle | r = 18.0 mm, same headings |
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
