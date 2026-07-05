# ShapeDiffuser

**Conditional diffusion models for whole-body inverse kinematics of multi-segment
soft continuum manipulators.**

Research codebase targeting a full academic venue (ICRA / IROS / CoRL / RoboSoft /
Soft Robotics journal). Simulation-only, PyTorch-based.

## The idea (paper thesis)

Soft continuum arms (e.g., STIFF-FLOP-style pneumatic manipulators) are highly
redundant: many actuator configurations realize the same tip pose or body shape.
Classic learned IK — a motor-babbling-trained CNN/MLP regressor — is
deterministic, so on multimodal targets it *averages incompatible solutions*
into an invalid one. ShapeDiffuser instead learns the full conditional
distribution p(actuation | target) with a conditional diffusion model, so it can
(1) recover *all* distinct IK modes, and (2) exploit them downstream, e.g.,
choosing a collision-free whole-arm shape among candidates for the same tip
target.

## Claims -> experiments mapping

| # | Claim | Experiment (scripts/evaluate.py) | Headline metric |
|---|-------|----------------------------------|-----------------|
| E1 | Accuracy on par with deterministic learned IK | held-out reachable targets, best-of-K and single sample vs. MLP/MDN/grad-IK | tip error (mm), success @ 5 mm |
| E2 | Recovers the full IK solution set (deterministic baselines cannot) | ground-truth mode enumeration (2M-sample sweep + DBSCAN) per target | mode recall, solution diversity |
| E3 | Multimodality has downstream value | same tip target + random obstacles; success iff any of K samples is accurate AND collision-free | task success rate |
| E4 | Practical inference cost | wall-clock DDIM sampling | ms per target (batch of K) |

E1-E4 are all validated below with 3-seed statistics. The project has since
grown a second thread (sim-to-sim transfer + query-budget candidate selection)
beyond this original scope — see "Results" and "Novelty / related work" below.

## Results

All numbers are mean +/- std over 3 seeds unless noted. Full config:
`configs/default.yaml` (500k train samples, 60 epochs, ddim_steps=20,
guidance=1.5 - both retuned from an ablation, see below).

**E1-E4, core (multiseed_results.json):**

| | diffusion | mlp | mdn |
|---|---|---|---|
| tip err best-of-K (mm) | 0.444 +/- 0.031 | 13.362 +/- 0.027 | 17.313 +/- 0.146 |
| success @ 5mm | **1.00 +/- 0.00** | 0.060 +/- 0.001 | 0.027 +/- 0.003 |
| mode recall | **0.835 +/- 0.032** | 0.000 | 0.000 |
| diversity (curvature space) | 28.16 +/- 0.22 | 0.00 | 0.00 |
| obstacle-task success | **0.978 +/- 0.002** | 0.040 +/- 0.004 | 0.018 +/- 0.002 |
| ms / target (batch of K, CPU) | 38.6 +/- 4.9 | 0.013 | 0.022 |

All three claims (E1 parity-or-better, E2 multimodality, E3 downstream value)
hold with tight error bars and large, stable gaps over both baselines.

**Ablations (`scripts/ablate.py`):** guidance weight follows the expected
inverted-U (0.0 fails to converge; >=3.0 over-guides and collapses accuracy
*and* diversity; 1.0-2.0 is the plateau; default 1.5 sits at the peak, 100%
success / 0.86 recall). Accuracy and recall saturate by ~20 DDIM steps (50-200
steps add cost with no benefit) - default dropped from 50->20 for a ~3x
inference speedup (85ms -> 27.5ms/target), verified with a full rerun.

**Dataset-size scaling (`scripts/scale_ablation.py`, diffusion only):**

| n_train | 5,000 | 20,000 | 50,000 | 100,000 | 200,000 | 500,000 |
|---|---|---|---|---|---|---|
| best-of-K err (mm) | 12.00 | 1.86 | 0.78 | 0.71 | 0.51 | 0.40 |
| success @ 5mm | 0.20 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 |
| mode recall | 0.02 | 0.61 | 0.73 | 0.90 | 0.83 | 0.88 |

Diminishing returns past ~50-100k samples; the full 500k is not far past the
point of saturation.

**Shape-conditioned variant (`configs/shape.yaml`, `scripts/evaluate_shape.py`,
3 seeds via `scripts/multiseed_shape.py`):**

| | diffusion | mlp |
|---|---|---|
| shape err best-of-K (mm) | 0.158 +/- 0.009 | 0.492 +/- 0.004 |
| success @ 5mm (mean per-point) | 1.00 | 1.00 |
| diversity (curvature space) | 0.346 +/- 0.022 | 0.0019 +/- 0.0004 |

This is the project's cleanest mechanistic result: conditioning on the full
24-D whole-body shape instead of the 3-D tip collapses diffusion's sampled
diversity by **~85x** relative to tip-conditioning (28.16 -> 0.35), because
specifying the whole shape removes almost all of the actuation redundancy that
motivates diffusion in the first place. Correspondingly, mlp's disadvantage
shrinks from **~58x worse** than diffusion (tip-conditioned) to only **~3x
worse** (shape-conditioned). This isolates *why* diffusion helps (resolving
redundancy) rather than just showing it wins everywhere.

**PCC->Elastica sim-to-sim transfer (`scripts/transfer_study.py`,
`elastica_arm.py` verified working against pyelastica 1.0.0):** diffusion's
near-perfect PCC self-consistency (0.44mm, 100% success) collapses entirely
when the same actuations are executed on the higher-fidelity Cosserat-rod
simulator (140.6mm, **0% success**); mlp collapses similarly (13.8mm ->
164.5mm). Calibration checks ruled out a simple unit/gain mismatch between the
simulators - see `transfer_study_results.json` and the commit history for the
`torque_gain` sweep. The learned inverse map is entirely simulator-specific
and does not transfer.

**Query-budget candidate selection (`scripts/query_budget_study.py`, n=45
targets, `query_budget_results_n45.json`):** does having K=32 diverse
candidates + a small budget of m expensive Elastica queries (tried in
ascending PCC-confidence order) recover any of that transfer loss? Binary
success stays at exactly 0% for every budget 1-32 (the 5mm tolerance is ~30x
smaller than the gap, so it's the wrong lens). The real, continuous signal:
mean best-of-m error declines monotonically, 161mm (m=1) -> 130mm (m=32),
a mean per-target improvement of ~30mm that is broad-based (44/45 targets
improve) but highly variable in size (range ~0-75mm). **Multimodality gives a
real but bounded benefit** under this scale of structural model mismatch -
useful, but nowhere near sufficient on its own to reach practical accuracy.

**What explains the per-target variance? (`scripts/query_budget_correlation.py`,
properly powered at n=45 after an n=20 pilot suggested r=0.42):** tested two
PCC-side predictors - candidate diversity (r=0.198, p=0.191) and local PCC
Jacobian sensitivity (r=0.024, p=0.878). **Neither holds up.** The n=20
pilot's r=0.42 was very likely small-sample noise. This is reported as a
genuine negative result, not a reason to keep searching until something
sticks - the mechanism behind which targets benefit from candidate selection
remains an open question.

**Does *how* you spend the query budget matter?
(`scripts/query_budget_ordering_paired.py`, n=45, paired design - one Elastica
pass per target, both orderings reindex the same cached per-candidate errors,
so the comparison isn't confounded by between-run sampling noise):** yes,
**confirmed**. Ordering candidates by diversity (greedy farthest-point in
curvature space, seeded from the same PCC-best candidate as the baseline, so
budget=1 is identical between orderings) significantly beats naive
PCC-confidence ordering in the middle of the budget range:

| budget m | 2 | 4 | 8 | 16 |
|---|---|---|---|---|
| mean advantage (mm) | 3.65 | 4.59 | **6.70** | **3.27** |
| paired t-stat | 1.55 | 1.97 (p~0.055) | **3.34 (p~0.002)** | **3.21 (p~0.002)** |
| targets: diversity wins / pcc wins | 15/13 | 18/18 | 24/12 | 19/9 |

(m=1 and m=32 correctly converge to zero difference by construction - both
orderings start from the same best guess and eventually exhaust the same full
candidate set.) An earlier, less rigorous *independent-runs* comparison (each
ordering sampled its own candidates) showed the same direction but wasn't
statistically distinguishable from noise - the paired design is what makes
this a real result rather than a repeat of the diversity-correlation false
lead above. **This is the project's first confirmed improvement over the
naive query-budget baseline**: multimodality isn't just "helpful in the
aggregate" (the earlier finding), it's *actionably* helpful - a
diversity-aware query strategy measurably outperforms the obvious baseline
strategy, especially once you have a moderate (not minimal, not maximal)
number of queries to spend.

**Morphology generalization (`scripts/morphology_sweep.py`, 2/4/6-segment PCC
arms, `morphology_sweep_results.json`, E1/E3/E4 only - E2's dbscan
calibration is 4-segment-specific, out of scope here):**

| n_segments | diffusion err (mm) / success | mlp err (mm) / success | obstacle: diffusion / mlp |
|---|---|---|---|
| 2 | 0.113 / 100% | 1.719 / 75.0% | 0.745 / 0.460 |
| 4 | 0.277 / 100% | 13.461 / 5.3% | 0.975 / 0.045 |
| 6 | 0.917 / 99.8% | 33.872 / 1.2% | **1.000 / 0.010** |

The core thesis is not an artifact of the one 4-segment arm used everywhere
else: as segment count (and actuation redundancy) grows, mlp's error explodes
and its success collapses while diffusion degrades far more gracefully. The
obstacle-task gap at 6 segments (100% vs 1%) is the largest in the whole
project - **the advantage grows, not shrinks, with more redundancy**, the
more scientifically interesting direction to have confirmed.

## Novelty / related work

Checked against the closest published work before committing to the
sim-to-sim pivot (see commit history for search queries/dates; verify these
are still current before relying on this for a submission):

* **IKDiffuser** (arXiv 2506.13087) is a diffusion IK solver for **rigid
  multi-arm** systems - its "structure-agnostic"/"flexible kinematic trees"
  framing means flexible *topology* (varying numbers of end-effectors), not
  physically compliant material. Materially different problem from soft
  continuum manipulators; low direct overlap.
* General diffusion-policy candidate-selection (picking one action from a
  multimodal policy's samples) is an active area, but framed around learned
  scorers, not a *query-budget* question, and not applied to soft-arm
  sim-to-sim transfer specifically.
* No public dataset was found that drops in as real-world validation for this
  specific actuation-to-shape task (SoPrA is a real, structurally similar
  platform used in several papers, but no confirmed public data release;
  PokeFlex is public and real but is passive deformable-*object* poking, not
  actuated continuum-arm IK).

The most defensible novel framing right now: a soft-continuum-specific,
reproducible **sim-to-sim** testbed (PCC vs. Cosserat) quantifying that (a)
multimodal diffusion IK's advantage is redundancy-specific (shown by the
shape-conditioning collapse), and (b) that advantage does not survive a
severe, structural model-mismatch, with passive candidate selection providing
only a bounded recovery whose per-target variance is not yet explained.

## Repo layout

```
configs/
  default.yaml                   tip-conditioned, main config (4-segment arm)
  shape.yaml                     shape-conditioned variant (cond_type: shape)
  smoke.yaml                     fast end-to-end smoke-test config
  morph_2seg.yaml  morph_6seg.yaml   morphology-generalization sweep (2/6-segment arms)
src/shapediffuser/
  pcc_arm.py                     differentiable PCC arm (fast GT engine + grad-IK baseline)
  elastica_arm.py                PyElastica Cosserat arm (verified working, ~4s/sample, unbatched)
  data.py                        motor babbling generation, Dataset, normalization
  models.py                      diffusion (DDPM/DDIM + CFG), MLP, MDN
  metrics.py                     tip/chamfer error, curvature_features, diversity, mode enumeration/recall
scripts/
  generate_data.py  train.py  evaluate.py  visualize.py      core pipeline
  ablate.py                      guidance-weight / DDIM-steps ablation
  scale_ablation.py              dataset-size ablation (diffusion only)
  multiseed.py                   3-seed statistics for the core E1-E4 run
  evaluate_shape.py               evaluation for the shape-conditioned variant
  multiseed_shape.py             3-seed statistics for the shape-conditioned variant
  transfer_study.py              PCC-trained model executed on Elastica
  query_budget_study.py          candidate-selection-under-mismatch, budget sweep (--order pcc|diversity)
  query_budget_correlation.py    tests predictors of per-target selection benefit
  query_budget_ordering_paired.py  paired pcc-order vs diversity-order comparison (confirmed result)
  morphology_sweep.py             confirms E1/E3/E4 generalize across 2/4/6-segment arms
```

## Quickstart

```bash
pip install -r requirements.txt

python scripts/generate_data.py                 # ~500k samples, ~1 min on CPU
python scripts/train.py --model diffusion       # ~30-40 min on CPU (no GPU needed)
python scripts/train.py --model mlp
python scripts/train.py --model mdn
python scripts/evaluate.py                      # writes results.json (E2 is the slow part)
python scripts/visualize.py                     # headline multimodality figure
```

Everything above has been run end-to-end multiple times on CPU-only hardware
(no GPU required, ~30-40 min for diffusion training on the full 500k dataset).
`configs/smoke.yaml` is a fast (~2 min) reduced config for a first sanity pass.

Optional, reproduces the other results in this README:
```bash
python scripts/multiseed.py --seeds 0,1,2                          # core E1-E4, 3 seeds
python scripts/ablate.py --sweep guidance --values 0.0,0.5,1.0,1.5,2.0,3.0,4.0
python scripts/ablate.py --sweep ddim_steps --values 5,10,20,50,100,200
python scripts/scale_ablation.py --sizes 5000,20000,50000,100000,200000,500000
python scripts/train.py --model diffusion --config configs/shape.yaml
python scripts/train.py --model mlp --config configs/shape.yaml
python scripts/evaluate_shape.py --config configs/shape.yaml
python scripts/multiseed_shape.py --seeds 0,1,2

pip install pyelastica                                              # optional, ~4s/sample
python scripts/transfer_study.py --n_targets 20 --k 8
python scripts/query_budget_study.py --n_targets 45 --k 32
python scripts/query_budget_correlation.py --query_budget_results query_budget_results_n45.json
```

## Status and honesty notes

* All of the above has been executed (not just syntax-checked) on CPU-only
  hardware, multiple times, with 3-seed statistics on the core results and the
  shape-conditioned variant. Two real bugs were found and fixed in E2's mode
  enumeration/matching (a DBSCAN eps/min_samples scale mismatch, then a deeper
  curvature-space/gauge-redundancy issue - raw pressure vectors have a
  4-segment-dimensional null space that contaminated distance-based
  clustering/matching; see `metrics.curvature_features`).
* `elastica_arm.py` is no longer an untested scaffold - verified working
  against pyelastica 1.0.0 (only fix needed: `progress_bar=False`). It is
  slow (~4s/sample, unbatched Cosserat integration), so transfer/query-budget
  studies deliberately use small target counts (20-45), not full-scale E1-E4
  numbers.
* Success bar (from prior discussion) is met for the *primary* PCC-only
  results: E1 well within ~1-2% of arm length; E2 recall 0.835 (target was
  >= 0.8-0.9); E3 a large gap. It is **not** met once Elastica model mismatch
  is introduced (0% success under transfer) - this is now a headline finding
  in its own right, not a gap to hide.
* No real hardware validation anywhere in this project - everything is
  simulator-only (PCC and PyElastica). Explicitly flag this as future work in
  any writeup; a real soft-robot dataset for this exact task (pneumatic
  continuum arm actuation -> tip/shape, ideally with repeated trials to
  capture actuator hysteresis) was searched for and not found publicly
  available.

## Open threads / next steps

**Resolved:** ~~Smarter candidate ordering~~ - confirmed: diversity-ordering
significantly beats PCC-confidence ordering at moderate budgets (m=8,16; see
Results above and `query_budget_ordering_paired.json`).

**Resolved:** ~~Morphology generalization~~ - confirmed: the core E1/E3/E4
story holds (and the diffusion-vs-mlp gap grows) across 2/4/6-segment arms;
see Results above and `morphology_sweep_results.json`. E2 was not
regeneralized (would need per-morphology dbscan recalibration).

Remaining, in rough priority order:

1. **Actual model adaptation**, not just selection: use a handful of Elastica
   samples to fit a residual correction or re-bias the diffusion sampler,
   rather than filtering among candidates from an unadapted model. This is a
   bigger design effort but directly tests whether the query-budget study's
   "selection alone isn't enough" finding can be fixed with adaptation - the
   diversity-ordering result suggests *how* you select already matters a lot,
   so adaptation informed by that (e.g. bias sampling toward whatever
   diversity-ordering is implicitly finding) may be the natural next step
   rather than a from-scratch direction.
2. **Real hardware or a suitable public dataset**, if one turns up - would
   upgrade the sim-to-sim transfer story to genuine sim-to-real.
