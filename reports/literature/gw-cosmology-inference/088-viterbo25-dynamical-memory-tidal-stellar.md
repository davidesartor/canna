# The dynamical memory of tidal stellar streams: Joint inference of the Galactic potential and the progenitor of GD-1 with flow matching

Viterbo & Buck, 2025, A&A; arXiv v1 Dec 2025 · [arXiv:2512.04600](https://arxiv.org/abs/2512.04600)

**Evidence: B.** Peer-reviewed (Astronomy & Astrophysics); 2 citations in 0.9 yr (Semantic
Scholar, 2026-10-19).

## What they do

- FMPE (Dax+23 recipe), applied to 13 parameters: 4 host-potential and 9 progenitor.
- The input is the phase space of N = 1000 stream stars.
- Training data: 2×10⁵ simulations from the differentiable JAX N-body code Odisseo.
- The probability path is the Gaussian OT path with σ_min, from an N(0, I) base.

## What they find

- Test set: 10³ simulations, each with 10³ posterior samples.
  - Host parameters are well calibrated on per-marginal P–P plots.
  - The two Plummer parameters (M, a) show S-shaped P–P curves; M_Plummer is over-estimated.
  - v_y^c is miscalibrated.
  - The joint TARP curve is close to the diagonal.
- The mock GD-1 posterior recovers the truth within the 16–84% interval for all parameters and reproduces Alvey+24 on the parameters they share.
- The posterior predictive check (forward-simulating 1000 posterior samples) matches the observed stream morphology.
- No ablations are reported. The architecture was "selected after extensive experimentation", but those experiments are not shown.

## Implementation details

- **Set Transformer velocity network:**
  - star tokens are embedded by an MLP (128, SiLU);
  - 3 self-attention blocks (SAB) with FiLM time-modulation;
  - then a cross-attention block (CAB) in which **the embedded θ_t is the query and the star tokens are keys and values**, followed by FiLM and an MLP head.
  - This is "parameters query the data", one-way: parameter tokens attend to data tokens.
- Time embedding: sinusoidal.
- Optimiser: AdamW, lr 1e-3, ReduceLROnPlateau (patience 5), batch 500.
- Early stopping at patience 30 gave 123 epochs.
- A 50k validation set (a large share of the data).
- Costs: ~7.5 H200-h to simulate the data, ~8 A100-h to train.
- Validation follows the triad that is becoming standard: marginal P–P + joint TARP + posterior predictive check.

## Caveats

- Uniform-in-ratio priors within a factor of 2.
- Observational errors are simplified.
- The failures (Plummer parameters, v_y) are attributed to weak information content and too narrow a prior; this is not tested.
- No comparison against a likelihood-based reference posterior.
