# Submission package — MDPI *Robotics*

## Where to submit

- Journal: https://www.mdpi.com/journal/robotics
- Submission system (SuSy): https://susy.mdpi.com/ — create an account, then "Submit Manuscript" → journal *Robotics* → article type *Article*
- Author instructions: https://www.mdpi.com/journal/robotics/instructions

## What is in this folder

| file | upload as |
|---|---|
| `manuscript.pdf` | the manuscript (15 pages) |
| `latex_source.zip` | source files — `main.tex`, `references.bib`, `Definitions/` (MDPI class), `figures/`. Verified to compile standalone from a clean copy: 0 errors. |
| `cover_letter.md` | paste into the cover-letter field |

## Before you click submit

- **Email** — filled in (adhvayjagadeesh@gmail.com). No placeholders remain in the manuscript.
- **Affiliation** — "Independent Researcher". If CCIR or a school should be listed, edit `\address{...}` in `main.tex` and rebuild.
- **Suggested reviewers** — optional. MDPI's instructions say authors *may* suggest reviewers, typically from among the authors they cite; it is not a required field. If you leave it blank the editor assigns them.

## Timeline and cost (MDPI Robotics, median values for H1 2026)

- First decision: **~20 days** after submission
- Acceptance to publication: **3.6 days**
- **APC: CHF 1800** (roughly USD 2,000–2,200), charged **only on acceptance**. Some institutions hold MDPI Institutional Open Access Program (IOAP) memberships that discount this — check before paying.
- Impact Factor 3.6 (2025), 5-year 4.0, JCR Q2 in Robotics

## What MDPI will ask for in the form that is already in the manuscript

- Abstract and keywords — in `main.tex`, will be auto-extracted from the PDF or paste them
- Funding: "This research received no external funding."
- Conflicts of interest: "The author declares no conflict of interest."
- Data availability: the GitHub URL, stated in the Acknowledgments block. The 2020 MDPI class used here predates the dedicated `\dataavailability` macro; the editorial office restructures back matter on acceptance and will move it.

## Repository state

Public, at https://github.com/adhvayjagadeesh/soft-arm-diffusion-ik. `main` is branch-protected (no force-push, no deletion). No AI attribution anywhere in the commit history. Every result file the paper cites is committed; `scripts/print_paper_tables.py` regenerates the amortization tables from them.

## Template version

`Definitions/mdpi.cls` is the 2020 MDPI class from a public mirror. The current template is on Overleaf ("MDPI Article Template"); MDPI accepts either, since production reformats. If you prefer the current one, drop `main.tex`, `references.bib`, and `figures/` into a fresh Overleaf copy of that template and replace the preamble macros — the body is portable.
