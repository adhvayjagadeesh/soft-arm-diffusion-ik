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

The paper lives or dies on **E2** (structural advantage) and **E3** (why it
matters). E1 only needs parity. Add ablations before submission: DDIM steps vs.
accuracy, guidance weight, dataset-size scaling, tip- vs. shape-conditioning.

## Repo layout

```
configs/default.yaml            all hyperparameters
src/shapediffuser/
  pcc_arm.py                    differentiable PCC arm (fast GT engine + grad-IK baseline)
  elastica_arm.py               EXPERIMENTAL PyElastica Cosserat arm (high-fidelity path)
  data.py                       motor babbling generation, Dataset, normalization
  models.py                     diffusion (DDPM/DDIM + CFG), MLP, MDN
  metrics.py                    tip/chamfer error, diversity, mode enumeration/recall
scripts/
  generate_data.py  train.py  evaluate.py  visualize.py
```

## Quickstart

```bash
pip install -r requirements.txt

python scripts/generate_data.py                 # ~500k samples, minutes on CPU
python scripts/train.py --model diffusion       # ~30-60 min on one modern GPU
python scripts/train.py --model mlp
python scripts/train.py --model mdn
python scripts/evaluate.py                      # writes results.json (E2 is the slow part)
python scripts/visualize.py                     # headline multimodality figure
```

Smoke test first: in `configs/default.yaml` set `n_train: 20000`, `epochs: 3`,
`n_targets_modes: 2`, `mode_pool: 200000`, and run the whole pipeline end-to-end
in a few minutes (CPU is fine) before any full run.

## Status and honesty notes

* Written by Claude in a sandbox **without PyTorch installed**: every file is
  syntax-checked but **not executed**. Expect the possibility of minor
  first-run fixes (tensor shapes, API details), especially in
  `elastica_arm.py`, which is an untested scaffold.
* The PCC arm is the primary engine: fast, batched, differentiable, and honest
  as ground truth since the paper's claims are about *learning the inverse map
  of a given simulator*, not about simulator fidelity. The PyElastica path
  exists for a transfer study (train on PCC, evaluate on Cosserat) that would
  strengthen the paper.
* Success bar (from prior discussion): E1 within ~1-2% of arm length and parity
  with baselines; E2 recall >= 0.8-0.9 where baselines structurally cap out at
  ~1/k modes; E3 a large gap (single-solution methods can't dodge obstacles).
  Multiple seeds + error bars before submission.

## Working on this repo with Claude Code (handoff)

This README is the context handoff — point Claude Code at it. Suggested first
session:

1. `pip install -r requirements.txt` (plus CUDA-matched torch wheel).
2. Run the smoke-test config end-to-end; fix any first-run errors.
3. Restore full config; launch `generate_data.py` then the three training runs.
4. Run `evaluate.py --skip_modes` for quick E1/E3/E4 numbers, then the full E2.
5. Commit results.json + figures; iterate on guidance weight and mdn_k so the
   MDN baseline is tuned fairly (reviewers check this).
6. Next milestones: ablations, shape-conditioned variant (`cond_type: shape`),
   PCC->Elastica transfer study, multi-seed statistics.
