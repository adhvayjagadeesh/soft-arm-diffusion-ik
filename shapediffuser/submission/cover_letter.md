Dear Editors of *Robotics*,

Please consider the enclosed manuscript, "One Model, Many Arms: Amortized Multimodal Inverse Kinematics Across a Continuous Family of Soft Continuum Manipulators," for publication as a research article.

Learned inverse-kinematics models for soft continuum arms are trained for one arm, yet no two fabricated soft arms are the same arm. The manuscript asks whether a single conditional diffusion model can solve inverse kinematics across a continuous family of arm morphologies it never saw, and what makes that possible. Its findings, all in simulation with three-seed statistics on every headline comparison, are:

- A single morphology-conditioned diffusion model reaches 0.41 ± 0.04 mm best-of-32 tip error with 100% success on held-out morphologies, within 0.06 mm of per-arm specialists trained on 5 × 10⁵ samples of their own arm, with zero samples of any specific arm.
- The identical training data with the morphology withheld (domain randomization) reaches 12.7 ± 0.2 mm — a 31× gap attributable to one conditioning vector. A direct test shows the withheld-morphology model learned the marginal over arms: every sample solves the target on some arm in the family, almost none on the queried one.
- The generalization envelope beyond the training family is measured, not asserted, and is asymmetric.
- Deterministic and mixture-density regressors trained on the same data do not amortize; only a model that represents the multimodal solution set can exploit the conditioning.

The manuscript also reports the single-arm advantage, its redundancy mechanism, and the limits of transfer under dynamics-model change, so that the amortization result is read against an honest account of what a learned inverse map can and cannot do. Negative and bounded results are reported with the same prominence as positive ones.

All code, configuration, and per-experiment result files are public at https://github.com/adhvayjagadeesh/soft-arm-diffusion-ik; every number in the manuscript traces to a committed result file named in its table caption.

This manuscript is original, has not been published previously, and is not under consideration elsewhere. The author declares no conflict of interest and received no external funding.

Sincerely,

Adhvay Jagadeesh
