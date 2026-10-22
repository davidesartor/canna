# Uncertainty in a Single Pass: A Closed-Form Identity for One-Step Flow Matching

Lv / Xing et al., 2026, arXiv preprint · [arXiv:2605.00941](https://arxiv.org/abs/2605.00941)

**Evidence: C.** Not peer-reviewed; 3 citations in 0.5 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

Tweedie's second-moment identity adapted to the linear interpolant x_t = t x1 + (1−t) x0 with **x0 ~ N(0,I)**, so x_t|x1 is Gaussian.
- From E[x1|x_t] = x_t + (1−t)v, differentiating gives Cov(x1|x_t) = (1−t)²/t · [I + (1−t)J_v].
- Trace: U = (1−t)²/t·[d + (1−t) div v].
- The diagonal and the trace come from Hutchinson/Rademacher JVP probes on a pretrained model, with no retraining.
- For MeanFlow one-step generators, evaluate the instantaneous velocity ū(x, ε, ε) at ε = 1e−2. This gives the uncertainty of the endpoint that x0 maps to.
- Using the Jacobian of the average-velocity generation map is not equivalent: there is an extra term.

## What they find

- CIFAR-10 uncertainty–error agreement:
  - Pixel Spearman ρ_pix peaks at 0.407 at t = 0.7, vs UA-Flow 0.346 and MC-dropout 0.373.
  - Sample-level ρ_samp runs from 0.679 (t = 0.3) to 0.966 (t = 0.9), vs UA-Flow 0.562–0.833 and a 5-model ensemble 0.33–0.42.
  - UA-Flow (a trained variance head) wins at early t = 0.3–0.4 on the pixel metrics.
- Boundary F1 top-10%: 0.692 vs 0.596 for the ensemble.
- BraTS MRI: roughly tied with MC-dropout.
- Probes: plateau after S = 64.
- Cost: 9.2 s online with no training, vs UA-Flow at 15.17 h offline training.
- Trace stays below the prior baseline (1−t)²/t·d, consistent with div v < 0.

## Implementation details

- 64 Rademacher probes, forward-mode JVPs.
- Clamp U ≥ 0, because J_v is not symmetric for a trained net and the diagonal can go negative at small t.
- Avoid t → 0, where the prefactor diverges.

## Caveats

- Only "uncertainty correlates with reconstruction error" is evaluated. There is no coverage or calibration test.
- For multi-step samplers the identity gives only Cov(x1|x_t), not the final-sample uncertainty.
- Relies on a Gaussian source and a linear interpolant.
- It measures the *data* posterior spread given x_t, not epistemic model error.
