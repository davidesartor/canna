# Artifact-Conditioned Interval Diagnostics for Flow-Matching Neural Posterior Estimation in a Controlled Gravitational-Wave Benchmark

Luo & Jing, 2026, arXiv preprint, astro-ph.IM · [arXiv:2606.12496](https://arxiv.org/abs/2606.12496)

**Evidence: C.** Not peer-reviewed; 0 citations in 0.4 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

- Toy controlled BBH benchmark: 5 parameters (ℳ, q, d_L, φ_c, t_c), a TaylorF2-like nonspinning waveform, two channels with fixed gains/delays, analytic whitening and a mask channel.
- Artifacts are labelled synthetically: glitch, frequency mask, PSD mismatch.
- FMPE uses straight-line CFM from N(0, I). Targets:
  - Bounded parameters in box-logit coordinates.
  - Coalescence phase as (cos φ_c, sin φ_c), decoded by atan2 after radial normalisation. The radius r_φ is logged as an off-circle diagnostic.
- Calibration: marginal 90% interval coverage error (MA90CE) with geodesic intervals on S¹.
- LAIR: per-artifact interval half-width scale factors, mixed by a 1-D CNN artifact classifier's probabilities.

## What they find

- Waveform resolution: mismatch 0.0317 at 256 bins, 8.5×10⁻⁴ at 1024, 5.9×10⁻⁵ at 2048. Coarse grids change the inference problem.
- 1024-bin FMPE raw MA90CE: 0.031–0.046 across modes. Global rescaling worsens every mode in Table 5, and frequency mask goes to 0.115.
- Six training seeds:
  - Frequency mask: raw 0.044 ± 0.013 vs global 0.118 ± 0.027.
  - Soft LAIR fixes the mask failure (0.067) but is worse than raw for PSD mismatch (0.063 vs 0.040) and mixed (0.059 vs 0.030).
- Conditional RealNVP is comparable to FMPE: worse on masks (0.073), better on PSD mismatch (0.030).
- Classifier: glitch recall only 0.21–0.25.
- **The key negative result:**
  - A prior-only baseline gets MA90CE 0.018–0.071, and a diagonal Gaussian 0.022–0.043. These are as good as FMPE.
  - Against a converged Laplace-preconditioned stretch MCMC (R̂ ≤ 1.009, ESS ≥ 3744), FMPE gives sliced W ≈ 0.39–0.47 and mean **90%-interval overlap ratios of 0.0097–0.011**.
  - So the well-"calibrated" FMPE posteriors are essentially disjoint from the likelihood reference.
  - Dynesty and single-chain HMC both failed their diagnostics on this likelihood. Laplace had Hessian condition number ~2×10⁵, and prior-proposal IS had ESS = 1.
- The FMPE is undertrained: 2000 training steps, 48 ODE steps.

## Implementation details

- The (cos, sin) phase embedding with an r_φ off-circle diagnostic. Geodesic credible intervals on S¹ for coverage.
- Rescale intervals in logit coordinates so bounded parameters stay inside support.
- Event-bootstrap CIs on coverage metrics.

## Caveats

- Toy simulator (no sky or inclination, fixed response).
- Tiny training budget and evaluation sets (256 events per mode).
- Only marginal coverage, no joint SBC or TARP. LAIR only widens or narrows intervals and cannot fix bias.
- The paper's main empirical lesson is in effect a cautionary one about its own metric.
