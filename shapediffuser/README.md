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

**Classical baseline (`results_gradik.json`, restart-init fix applied - see
honesty notes):** grad-IK (32 random restarts, Adam through the
differentiable PCC model, 300 iters) reaches 0.013mm at 100% success,
recall 0.85, diversity 29.0, obstacle success 0.98, at ~430ms/target
(vs. diffusion's 38.6ms batched / ~84ms single-target). With the simulator
in hand at query time, classical optimization matches diffusion on every
task metric; diffusion's structural advantage is answering from
input-output samples alone, which is what the transfer studies stress.

**Ablations (`scripts/ablate.py`):** guidance weight follows the expected
inverted-U (0.0 fails to converge; >=3.0 over-guides and collapses accuracy
*and* diversity; 1.0-2.0 is the plateau; default 1.5 sits at the peak, 100%
success / 0.86 recall). Accuracy and recall saturate by ~20 DDIM steps (50-200
steps add cost with no benefit) - default dropped from 50->20 for a ~3x
inference speedup (85ms -> 27.5ms/target), verified with a full rerun.

**MDN baseline fairness (`scripts/mdn_tuning_sweep.py`,
`mdn_tuning_results.json`):** sweeping the MDN's component count over
{4, 8, 16, 32} improves best-of-K error only from 18.7 to 14.8mm (never
surpassing the plain MLP's 13.4mm) and mode recall stays **exactly zero at
every setting** - the baseline's failure is not an under-tuning artifact.
Also verified against the trained checkpoint: the MDN does *not* suffer
classic component collapse (mixture weights near-uniform, ~7.8 effective
components of 8; component means spread ~13.7 apart in curvature space) -
its components are diverse but none is accurate enough to produce successful
samples, which is what mode recall counts.

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
diversity by **~80x** relative to tip-conditioning (28.16 -> 0.35), because
specifying the whole shape removes almost all of the actuation redundancy that
motivates diffusion in the first place. Correspondingly, mlp's disadvantage
shrinks from **~30x worse** than diffusion (tip-conditioned: 13.36 vs
0.44mm, 3-seed numbers; an earlier single-seed comparison had put this at
~58x) to only **~3x worse** (shape-conditioned). This isolates *why* diffusion helps (resolving
redundancy) rather than just showing it wins everywhere.

**PCC->Elastica sim-to-sim transfer (`scripts/transfer_study.py`,
`elastica_arm.py` verified working against pyelastica 1.0.0; 3 seeds via
`transfer_study_seed{0,1,2}_results.json`, added for the RA-L submission):**
diffusion's near-perfect PCC self-consistency (0.67 +/- 0.03mm, 100% success)
collapses entirely when the same actuations are executed on the
higher-fidelity Cosserat-rod simulator (139.9 +/- 1.8mm, **0% success on
every seed**); mlp collapses similarly (13.8mm -> 164.4 +/- 0.03mm).
Calibration checks ruled out a simple unit/gain mismatch between the
simulators - the `torque_gain` sweep is committed as
`gain_calibration_sweep` in `robustness_checks.json` (no gain over two
orders of magnitude around the default aligns the tips; mean discrepancy
never drops below ~150mm despite similarly sized workspaces). The damping
constant is ruled out too (`damping_calibration_sweep` +
`transfer_damping_sensitivity`, sixth-pass additions): dampings 0.5-10 move
the never-settled snapshot by up to ~190mm yet never bring tip discrepancy
below ~149mm, and the collapse itself persists at every tested damping
(110-145mm best-of-8, 0% success at 0.5/2/8). The learned
inverse map is entirely simulator-specific and does not transfer, and this is
seed-stable, not a training fluke.

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

**Model adaptation via transfer-guided sampling
(`scripts/train_transfer_regressor.py`, `scripts/evaluate_adaptation.py`,
`GaussianDiffusion.sample_with_transfer_guidance` in `models.py`):** rather
than reranking/correcting already-sampled candidates (the query-budget
thread above), this steers the diffusion sampling process itself using a
small learned regressor that predicts Elastica transfer error from
(candidate, target) - the same mechanism as classifier guidance in image
diffusion, applied to shape the generative distribution toward the
mismatch-robust region rather than filtering after the fact. Trained on 25
targets (seed=8888), tested on **15 held-out targets** (seed=9999, never
seen by the regressor - the whole point, since only helping on training
targets would be memorization):

| | baseline (unguided) | guided |
|---|---|---|
| mean best-of-16 Elastica err (mm) | 134.6 | **129.0** |

Paired delta 5.56 +/- 2.45mm, **t=2.27 (p~0.04)**, 11/15 targets favor
guidance. This crosses conventional significance and genuinely generalizes to
unseen targets, so it's real adaptation, not memorization - but report this
calibrated, not oversold: the effect is modest (~4% error reduction) and the
significance margin is noticeably weaker than the diversity-ordering result
(t=3.34 there vs. t=2.27 here). Worth replicating with more held-out targets
before leaning on it hard in a writeup.

**Do diversity-ordering and guided sampling combine?
(`scripts/combined_adaptation_ordering.py`, same 15 held-out targets, full
2x2 factorial from one paired Elastica pass per target):** **no.** Guided
sampling's benefit replicates (126-135mm vs. 139-164mm at matching budgets),
but diversity-ordering adds nothing on top of it - mildly *worse* at m=2
(135.1 vs 132.9mm) and m=4 (129.5 vs 127.9mm), tied at m=8 and m=16.
Mechanistic read: guided sampling *concentrates* candidates toward the
predicted-good region (that's what gradient guidance does); diversity-
ordering explicitly *spreads* selection across maximally different
candidates. Once guidance has already concentrated the distribution, chasing
diversity on top pulls toward outliers rather than reinforcing the good
region - the two mechanisms are in tension, not complementary. Even the best
single technique (guided sampling, ~126mm) recovers under 15% of the way from
baseline (~164mm) toward the 5mm tolerance: **neither passive selection nor a
small guidance signal substitutes for the model actually having seen the
target dynamics** - both operate strictly within the space of candidates a
PCC-only-trained distribution can produce. Closing the gap for real would
need training-time exposure to Elastica-like variation (e.g. domain
randomization spanning both simulators), a materially bigger undertaking than
anything tested here.

**Does training-time domain randomization help?
(`configs/dr.yaml`, `PCCArm`'s per-sample `gain` override,
`results_dr_pcc.json`, `transfer_study_dr_results.json`):** **no.**
Randomizing `curvature_gain` ~ Uniform(15,35) per training sample (vs. the
fixed 25.0), without conditioning the model on which gain applied, costs real
accuracy on the standard PCC task (0.44mm -> 2.18mm, ~5x worse; success
100% -> 95.5%) and produces zero measurable improvement on Elastica transfer
(140.58mm -> 140.62mm - noise, not signal; success 0% -> 0%). This confirms,
empirically, the concern raised when the experiment was scoped: the
PCC/Elastica mismatch is directional/structural, not a magnitude difference,
so randomizing *how much* the arm bends can't touch a mismatch in *which
direction* it bends. This motivates mixing in real Elastica ground-truth
data directly as the next, more principled step - it doesn't require the
mismatch to be expressible as a simple parameter range.

**Does fine-tuning on real Elastica-simulated data help?
(`scripts/generate_elastica_dataset.py`, 10,000 genuinely Elastica-simulated
(actuation, tip) pairs - simulation, not real hardware, see honesty notes;
`scripts/finetune_on_elastica.py`, heavy-oversampling fine-tune, prefixes
500-10,000, `finetune_scaling_results.json`):** **partially - a real Pareto
tradeoff, not a clean win.** Two clear, opposite monotonic trends as the
fine-tuning prefix grows:

| n (Elastica samples) | PCC err (mm) | PCC success | Elastica err (mm) | Elastica success |
|---|---|---|---|---|
| baseline (no fine-tune) | 0.44 | 100% | 140.6 | 0% |
| 500 | 27.0 | 0.6% | 125.5 | 0% |
| 1,000 | 53.1 | 0.4% | 111.3 | 0% |
| 3,000 | 125.4 | 0% | 79.9 | 5% |
| 5,000 | 130.8 | 0% | 71.6 | 5% |
| 10,000 | 136.3 | 0% | **63.5** | **10%** |

PCC accuracy collapses progressively worse with more fine-tune data (more
exclusive-Elastica gradient steps = more forgetting of the original
500k-sample training); Elastica transfer accuracy improves progressively
with more fine-tune data - dropping from 140.6mm to 63.5mm (~55% reduction,
the largest of any method tried) and reaching **10% success, the first
nonzero transfer success anywhere this session** (diversity-ordering, guided
sampling, and DR all stayed at exactly 0%). This confirms real Elastica data
carries genuinely exploitable transfer signal, validating real-data mixing
over synthetic randomization. But the current execution (heavy oversampling,
zero PCC data mixed in during fine-tuning) is over-aggressive: it isn't
adapting the model to handle both domains, it's overwriting it into a
mediocre Elastica specialist that's lost general PCC capability entirely.
The natural fix - mixing in some PCC data as regularization during
fine-tuning - is now backed by strong evidence the underlying signal is
worth preserving properly, rather than a guess.

**Does mixing PCC data into fine-tuning escape the tradeoff?
(`scripts/finetune_on_elastica_regularized.py`, 50/50 PCC:Elastica mix via a
WeightedRandomSampler, n_elastica=10,000, same held-out seed as the
unregularized comparison point):** **half yes, half no - a different point
on the same tradeoff, not an escape from it.**

| | PCC err (mm) | PCC success | Elastica err (mm) | Elastica success |
|---|---|---|---|---|
| no fine-tune (baseline) | 0.44 | 100% | 140.6 | 0% |
| regularized (50/50) | 3.74 | 93.4% | 145.3 | 5% |
| unregularized (pure Elastica) | 136.3 | 0% | **63.5** | **10%** |

Mixing PCC data back in **solved the forgetting problem decisively**: PCC
accuracy recovers to near-baseline (3.74mm/93.4% vs. the unregularized run's
catastrophic 136.3mm/0%), confirming the diagnosed causal mechanism was
correct. But the 50/50 ratio **diluted almost all of the transfer signal**
that made the unregularized result exciting: 145.3mm/5% is barely
distinguishable from (arguably slightly worse than) the no-fine-tune
baseline, a small fraction of the unregularized run's 63.5mm/10%.

**Is there a sweet spot? (full grid at 3 seeds x SIX ratios -
0/5/10/15/20/50% PCC mix, all conditions through the same script on
`checkpoints_seed0/1/2`; `finetune_grid_summary.json`):**
**no - the curve is a cliff, its edge is located below 5% mixing, and this
is seed-stable.**

Mean +/- std over 3 seeds:

| PCC mix | PCC err (mm) | PCC success | Elastica err (mm) | Elastica success |
|---|---|---|---|---|
| 0% (pure Elastica) | 133.4 +/- 1.2 | 0% | **67.1 +/- 0.9** | **10.0 +/- 0.0%** |
| 5% | 30.7 +/- 0.6 | 6.4% | 133.8 +/- 3.1 | 8.3 +/- 2.4% |
| 10% | 21.7 +/- 0.5 | 17.0% | 143.2 +/- 4.9 | 8.3 +/- 2.4% |
| 15% | 16.7 +/- 0.2 | 26.3% | 146.0 +/- 3.4 | 8.3 +/- 2.4% |
| 20% | 11.6 +/- 0.3 | 50.8% | 148.0 +/- 1.6 | 10.0 +/- 0.0% |
| 50% | 3.0 +/- 0.4 | 95.5% | 152.3 +/- 0.4 | 3.3 +/- 2.4% |

PCC retention improves smoothly and monotonically at every step (133.4 ->
30.7 -> 21.7 -> 16.7 -> 11.6 -> 3.0mm). Elastica transfer does **not**
degrade gracefully in step - the fine sweep (added for the RA-L revision,
addressing the external critique that 3 coarse points couldn't distinguish
a cliff from a smooth tradeoff) *locates* the collapse below the first grid
point: at just 5% PCC data, transfer error has already rebounded from
67.1mm to 133.8mm, forfeiting ~80% of the improvement over the
same-target-set no-fine-tune baseline (151.1 +/- 2.5mm on the seed-55555
held-out targets, `baseline_transfer_seed{0,1,2}_heldout55555.json`; the
oft-quoted 139.9mm is the transfer-collapse experiment's different
20-target draw), then drifts only slowly (143 -> 146 -> 148 ->
152mm). Intermediate-ratio values are stable across seeds (std <= 4.9mm),
so this is not an unlucky run. **Conclusion: retention and transfer sit on
a cliff whose edge lies below 5% mixing** - the transfer signal needs
training almost exclusively on Elastica data to manifest at all. A genuine
middle ground, if one exists, would likely need a different regularization
mechanism entirely (e.g. parameter-level constraints like elastic weight
consolidation, or freezing specific layers) rather than further
data-mixing-ratio tuning.

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
  scorers used to *rerank already-generated* candidates, not to *guide
  generation itself* (the transfer-guided-sampling result above) - and not
  applied to soft-arm sim-to-sim transfer specifically.
* No public dataset was found that drops in as real-world validation for this
  specific actuation-to-shape task (SoPrA is a real, structurally similar
  platform used in several papers, but no confirmed public data release;
  PokeFlex is public and real but is passive deformable-*object* poking, not
  actuated continuum-arm IK).

The most defensible novel framing right now: a soft-continuum-specific,
reproducible **sim-to-sim** testbed (PCC vs. Cosserat) quantifying that (a)
multimodal diffusion IK's advantage is redundancy-specific (shown by the
shape-conditioning collapse) and generalizes across morphologies, (b) that
advantage does not survive a severe, structural model-mismatch, and (c) both
passive candidate selection (diversity-ordering) and active generation
guidance (transfer-guided sampling) provide real but bounded, only partially
understood recovery from that mismatch - candidate diversity itself doesn't
explain the recovery's per-target variance (tested and rejected), so the
underlying mechanism remains an open question even where the effect is
confirmed.

## Repo layout

```
configs/
  default.yaml                   tip-conditioned, main config (4-segment arm)
  shape.yaml                     shape-conditioned variant (cond_type: shape)
  smoke.yaml                     fast end-to-end smoke-test config
  morph_2seg.yaml  morph_6seg.yaml   morphology-generalization sweep (2/6-segment arms)
  dr.yaml                        domain randomization (curvature_gain), tested and rejected
src/shapediffuser/
  pcc_arm.py                     differentiable PCC arm (fast GT engine + grad-IK baseline)
  elastica_arm.py                PyElastica Cosserat arm (verified working, ~4s/sample, unbatched)
  data.py                        motor babbling generation, Dataset, normalization
  models.py                      diffusion (DDPM/DDIM + CFG), MLP, MDN, TransferRegressor + guided sampling
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
  train_transfer_regressor.py    fits the Elastica-error regressor used for guided sampling
  evaluate_adaptation.py         guided vs. unguided sampling on held-out targets (confirmed result)
  combined_adaptation_ordering.py  2x2 factorial: do ordering + guidance stack? (no)
  generate_elastica_dataset.py   checkpointed/resumable Elastica-simulated dataset generation
  finetune_on_elastica.py        real-data fine-tuning scaling curve (partial win, Pareto tradeoff)
  finetune_on_elastica_regularized.py  50/50 PCC-mix fine-tune (solves forgetting, dilutes signal)
  evaluate_gradik.py             classical optimization baseline, full E1/E2/E3 protocol
  mdn_tuning_sweep.py            MDN component-count fairness sweep (4-32; changes nothing)
  mdn_pathology_analysis.py      MDN per-component autopsy (healthy training, imprecise components)
  make_paper_figures.py  make_supplementary_video.py   publication assets
  robustness_checks.py           grad-IK compute/RNG, E2 sensitivity, settle convergence, gain+damping sweeps
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
* **The Elastica domain is a dynamic snapshot, not an equilibrium**
  (found by `scripts/robustness_checks.py`, recorded in
  `robustness_checks.json`): the damped rod still oscillates at the 1.5s
  settling horizon and beyond (tip shifts of 50-145mm persist between
  horizons up to 9s). Every Elastica experiment, dataset, and label used the
  identical deterministic 1.5s protocol, so all results remain internally
  valid as transfer to a fixed dynamic-snapshot domain - but earlier
  "quasi-static equilibrium" descriptions were wrong and have been corrected
  in the paper and code. The same robustness pass also covers grad-IK's
  compute/accuracy tradeoff, restart-RNG stability (std <0.005mm), and E2's
  hyperparameter sensitivity (learned baselines are zero at every eps/radius
  setting; diffusion and grad-IK trade the lead - diffusion higher at 3 of 9,
  grad-IK at 5, one tie).
* **grad-IK's restart initialization was buggy and fixed post-hoc
  (2026-07-16, fifth adversarial pass):** the original code drew Adam's
  *logits* ~ U(0,1), confining initial pressures to [0.5, 0.73] and
  artificially limiting which IK modes the restarts could reach. With the
  fix (restarts uniform over pressure space, logit-reparameterized;
  `pcc_arm.grad_ik`), grad-IK improves from 0.23mm/recall 0.73/diversity
  9.5 to **0.013mm at 100% success, recall 0.85, diversity 29.0, obstacle
  0.98** (`results_gradik.json`) - on par with diffusion on every task
  metric. The honest comparison is therefore: diffusion's advantage over
  *classical optimization* is only cost (~5x single-target, ~11x batched;
  grad-IK at a tuned 100-iteration budget is within a small factor) and not
  needing the simulator at query time; its advantage over *learned*
  baselines (which share that amortization property) is unchanged and
  remains the paper's subject. All grad-IK numbers in the paper and this
  README are from the fixed init.
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

**Resolved:** ~~Actual model adaptation~~ - confirmed, modestly: transfer-guided
sampling (steering generation with a learned Elastica-error regressor, see
Results above) beats plain unguided sampling on held-out targets (t=2.27,
p~0.04), genuinely generalizing rather than memorizing. Effect size is
modest (~4% error reduction) and the significance margin is weaker than the
diversity-ordering result - flagged as "confirmed but modest," worth
replicating with more held-out targets before leaning on it hard.

**Resolved:** ~~Combine diversity-ordering and transfer-guided sampling~~ -
tested, they do **not** stack: guided sampling's benefit replicates but
diversity-ordering adds nothing on top (mildly worse at m=2,4, tied at
m=8,16) - see Results above and `combined_adaptation_ordering_results.json`.
Mechanistic read: guidance concentrates candidates toward a good region;
diversity-ordering explicitly spreads selection, working against that
concentration rather than with it.

**Resolved:** ~~Training-time domain randomization (curvature_gain)~~ -
tested and rejected: randomizing `curvature_gain` per-sample during data
generation (`configs/dr.yaml`, `data.curvature_gain_range`,
`PCCArm.forward`'s optional per-sample `gain` override) costs real accuracy
on the standard PCC task (tip err 0.44mm -> 2.18mm, ~5x worse; success
100% -> 95.5%) and produces **zero** measurable improvement on Elastica
transfer (140.58mm -> 140.62mm, noise not signal; success 0% -> 0%
unchanged). Confirms the concern raised when this was scoped: the
PCC/Elastica mismatch is structural/directional (established via the
`torque_gain` calibration sweep earlier), not a magnitude/gain difference, so
randomizing *how much* the arm bends doesn't touch the actual source of the
mismatch. See `results_dr_pcc.json`, `transfer_study_dr_results.json`.

Remaining, and now the clear ceiling on this whole thread: neither passive
selection, generation-time guidance, nor magnitude-only domain randomization
meaningfully closes the sim-to-sim gap - all three operate within (or, for
gain-DR, a simple reparameterization of) what a PCC-only-trained
distribution can produce, none touch the *directional* nature of the actual
mismatch.

**Resolved (partially):** ~~Mix in real Elastica ground-truth data directly~~
- tested: real signal confirmed (140.6mm -> 63.5mm, first nonzero transfer
success at 10,000 samples), but the heavy-oversampling execution catastrophically
forgets PCC-domain accuracy in the process (0.44mm -> 136mm) - a genuine
Pareto tradeoff, not a finished result. See Results above and
`finetune_scaling_results.json`.

**Resolved:** ~~Fine-tune with PCC data mixed in as regularization / find a
mixing-ratio sweet spot~~ - characterized with three points (0%, 20%, 50%
PCC mix): retention improves smoothly with more PCC data, but transfer
collapses almost immediately past 0% rather than degrading gracefully - a
cliff, not a slope. No sweet spot in per-batch data mixing; a genuine middle
ground would need a different regularization mechanism entirely (see
Results above). This closes out the real-data-mixing thread for this
session with a precise, well-characterized answer rather than an open
question.

Remaining:

1. **Parameter-level regularization** (e.g. elastic weight consolidation,
   frozen layers) instead of data-mixing-ratio tuning, if the retention/
   transfer tension is worth pursuing further - a meaningfully bigger
   undertaking than anything in this session.
2. **Real hardware or a suitable public dataset**, if one turns up - would
   upgrade the sim-to-sim transfer story to genuine sim-to-real. This is now
   the only item on this list not already investigated at least once this
   session.
