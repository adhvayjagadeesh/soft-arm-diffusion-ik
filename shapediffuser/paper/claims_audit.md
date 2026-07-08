# Claims-to-evidence audit (RA-L submission)

Every claim the paper could make, mapped to its evidence file and statistical
strength, with a decision: **HEADLINE** (a contribution the abstract claims),
**SUPPORTING** (a section/paragraph with its own figure/table), **BRIEF** (one
or two honest sentences), or **CUT** (repo-only, not in the paper).

Working title framing: *"When does multimodal diffusion IK help soft continuum
arms - and does it survive model mismatch?"* Two arcs: (I) the advantage and
its mechanism; (II) the sim-to-sim transfer study.

## Arc I - the advantage and why it exists

| # | Claim | Evidence | Strength | Decision |
|---|-------|----------|----------|----------|
| 1 | Diffusion IK beats deterministic baselines on accuracy (0.44+/-0.03mm / 100% vs MLP 13.4mm / 6%, MDN 17.3mm / 2.7%) | `multiseed_results.json` | 3 seeds, tight | **HEADLINE** (Table 1) |
| 2 | Diffusion recovers the IK solution set; deterministic baselines structurally cannot (mode recall 0.835+/-0.032 vs exactly 0) | `multiseed_results.json` | 3 seeds; GT modes from 2M-sample sweep | **HEADLINE** (Table 1) |
| 3 | Multimodality has downstream value: obstacle-task success 0.978+/-0.002 vs 0.040 / 0.018 | `multiseed_results.json` | 3 seeds | **HEADLINE** (Table 1) |
| 4 | Practical inference cost: 38.6+/-4.9 ms/target (CPU, K=32, 20 DDIM steps) | `multiseed_results.json` | 3 seeds | SUPPORTING (Table 1 row) |
| 5 | Guidance weight has an inverted-U optimum (1.5); accuracy/recall saturate by ~20 DDIM steps (3x speedup over 50) | `ablation_guidance.json`, `ablation_ddim_steps.json` | single-seed sweeps | SUPPORTING (methods para; justifies hyperparameters) |
| 6 | Accuracy/recall saturate by ~50-100k training samples | `scale_ablation_results.json` | single seed | SUPPORTING (`fig_scaling`, or supplementary if space-bound) |
| 7 | The advantage is redundancy-specific: whole-shape conditioning collapses sampled diversity ~85x (28.16 -> 0.346) and shrinks MLP's disadvantage from ~58x to ~3x | `multiseed_results.json`, `multiseed_shape_results.json` | 3 seeds both | **HEADLINE** (`fig_redundancy` a) - the paper's mechanistic core |
| 8 | The advantage grows with redundancy across morphologies (2/4/6 segments; obstacle gap reaches 100% vs 1%) | `morphology_sweep_results.json` | single seed per morphology; E2 excluded (dbscan calibration is 4-seg-specific) | SUPPORTING (`fig_redundancy` b) - state single-seed plainly |

## Arc II - the transfer study

| # | Claim | Evidence | Strength | Decision |
|---|-------|----------|----------|----------|
| 9 | The learned inverse map does not transfer PCC->Elastica: 139.9+/-1.8mm, 0% success on every seed (MLP likewise) | `transfer_study_seed{0,1,2}_results.json`, `finetune_grid_summary.json` | 3 seeds | **HEADLINE** - motivates Arc II |
| 10 | The mismatch is structural/directional, not a gain-calibration artifact | `torque_gain` sweep (commit history; qualitative) | qualitative check | BRIEF (one sentence + citation to repo) |
| 11 | Best-of-m selection recovers a real but bounded fraction (161 -> 130mm over m=1..32; 44/45 targets improve; success stays 0%) | `query_budget_results_n45.json` | n=45 targets, single model seed | SUPPORTING (`fig_query_budget`) |
| 12 | Diversity-ordering beats confidence-ordering at moderate budgets (paired delta 6.7mm at m=8, t=3.34, p~0.002; t=3.21 at m=16) | `query_budget_ordering_paired.json` | n=45 paired design (the independent-runs curves in `fig_query_budget` show the same direction but are NOT the significant comparison - caption must say the stats come from the paired design) | SUPPORTING |
| 13 | Transfer-guided sampling generalizes to held-out targets (5.6+/-2.5mm, t=2.27, p~0.04, 11/15 targets) | `adaptation_results.json` | n=15 held-out, single seed; weakest confirmed result | SUPPORTING with explicit calibration ("modest") |
| 14 | Ordering and guidance do not stack (guidance concentrates; diversity-ordering spreads - opposing mechanisms) | `combined_adaptation_ordering_results.json` | n=15, 2x2 factorial | BRIEF (mechanistic sentence) |
| 15 | Neither candidate diversity nor local Jacobian sensitivity predicts per-target selection benefit (r=0.20 p=0.19; r=0.02 p=0.88 at n=45; the n=20 pilot's r=0.42 was small-sample noise) | `query_budget_correlation_n45.json` | properly powered null | BRIEF - honest negative, also a useful cautionary methods note |
| 16 | Gain-domain-randomization fails cleanly: costs PCC accuracy (0.44 -> 2.18mm), zero transfer benefit (140.58 -> 140.62mm) | `results_dr_pcc.json`, `transfer_study_dr_results.json` | single seed | SUPPORTING (one paragraph; motivates real-data mixing) |
| 17 | Real Elastica data carries exploitable transfer signal: pure-Elastica fine-tuning reaches 67.1+/-1.0mm and exactly 10% success on all seeds - the only nonzero transfer success of any method | `finetune_grid_summary.json` | 3 seeds x uniform protocol | **HEADLINE** (`fig_finetune_cliff`) |
| 18 | But retention/transfer is a cliff, not a slope: 20% PCC mix already collapses transfer (148.0+/-1.6mm) while retention recovers smoothly (133 -> 11.6 -> 3.0mm); no data-mixing sweet spot exists | `finetune_grid_summary.json` | 3 seeds; note binary-success granularity is 5% at n_targets=20 - mean error is the reliable lens | **HEADLINE** (`fig_finetune_cliff`) - the paper's closing finding |

## Limitations section (must state, in this order of importance)

1. **Simulation-only** - both domains are simulators (idealized PCC vs. Cosserat-rod
   PyElastica); no hardware anywhere. Frame the whole study as controlled
   sim-to-sim; real-data validation is future work.
2. **Elastica actuation model is simplified** - constant per-segment couples with a
   default gain, quasi-static settling; it is a harder *test* domain, not a
   validated physical model of a specific arm.
3. **Single-seed components** - ordering/guidance/adaptation results (claims
   11-14) use one trained model; baseline transfer and the fine-tuning grid
   (claims 9, 17, 18) are 3-seed. Say exactly which is which.
4. **E2 calibration is morphology-specific** - mode enumeration hyperparameters
   were calibrated for the 4-segment arm; morphology results are E1/E3/E4 only.
5. **Success-rate granularity** - transfer-study binary success uses
   n_targets=20 (5% steps); mean error carries the statistical weight.

## Cut from the paper (repo-only)

- MDN analysis beyond its Table 1 row (it never outperforms MLP here).
- The E2 bug archaeology (dbscan scale mismatch, curvature-space/gauge fix) -
  one methods sentence pointing at `metrics.curvature_features`, no more.
- The n=20 correlation pilot and its retraction (subsumed by the n=45 null).
- Smoke-test / engineering / checkpointing details.
- The 15.6h data-generation logistics (a methods sentence: "10,000 samples,
  ~4s each, quasi-static settling").

## Figure/table plan

| Asset | Content | Status |
|---|---|---|
| Table 1 | Core E1-E4, 3 seeds, diffusion/MLP/MDN | numbers final (`multiseed_results.json`) |
| Table 2 | Fine-tuning grid, 3 seeds x 3 ratios | numbers final (`finetune_grid_summary.json`) |
| Fig 1 | `fig_finetune_cliff` - retention vs transfer cliff | **done** (`paper/figures/`) |
| Fig 2 | `fig_redundancy` - diversity collapse + morphology scaling | **done** |
| Fig 3 | `fig_query_budget` - best-of-m curves, both orderings | **done** (caption must cite paired stats) |
| Fig 4 | `fig_scaling` - dataset-size ablation | **done** (supplementary candidate if space-bound) |
