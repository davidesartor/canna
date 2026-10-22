# Peregrine: Sequential simulation-based inference for gravitational wave signals

Bhardwaj et al., 2023, PRD 2024 · [arXiv:2304.02035](https://arxiv.org/abs/2304.02035)

**Evidence: B.** Peer-reviewed; 11 citations in 3.5 yr (Semantic Scholar, 2026-10-19).

## What they do

truncated marginal neural ratio estimation (swyft).
- One classifier per 1-D parameter (and per 2-D pair) learns the marginal ratio p(ϑ|x0)/p(ϑ). The other parameters are varied in every simulation, so they are marginalised implicitly.
- Over 7+ rounds the box prior is truncated where p̂/max p̂ < ε = 10⁻⁵ (≈4.8σ), then fresh simulations are drawn in the truncated region.
- Inference targets a single observation; it is not amortised.
- Network: two parallel 1-D U-Nets, one over the time-domain and one over the frequency-domain strain (H1, L1, V1 aligned), whose outputs are concatenated and fed to an MLP ratio head.

## What they find

- SEOBNRv4PHM, 15-D precessing BBH.
- Two case studies: C1 (SNR ~20, 900 Mpc) and C2 (SNR ~100, 200 Mpc).
- 1-D and 2-D marginals agree with dynesty and are within the sampler-to-sampler JSD spread: (3–9)×10⁻³ nat at low SNR, (10–30)×10⁻³ nat at high SNR.
- Cost for C1: 720k waveforms and 12 h wall time vs 44M and 43 h for dynesty. The 2-D marginals need another 500k simulations.
- Simulation schedule: 30k, 60k, 90k, 120k, 120k, 150k, 150k.
- Expected-coverage tests over 2000 truncated-prior simulations pass for all 15 parameters in both cases.
- Largest disagreements with the sampler are on a₂ and ϕ_c. The sampler uses analytic phase marginalisation, which "is crucial for stable results" in dynesty.

## Implementation details

- **Noise shuffling:** re-pair fresh noise with the same signals every epoch. This stops the network memorising noise and "can reduce the simulation budget by potentially an order of magnitude", for any additive noise model.
- Badly constrained parameters (ψ, a₁) are poorly estimated in early rounds and only converge once the well-measured ones (M_c, θ_jn) have been truncated. Constraining the sharp parameters first makes the weak ones learnable.
- Training: early stopping with patience 7, LR 5×10⁻⁴, batch 256, best-validation checkpoint restored.
- Feed both the time and frequency representations: t_c is a shift in one domain and a phase ramp in the other.

## Caveats

- Truncation is a rectangular box, so it is unsuited to multimodal posteriors (acknowledged).
- Per-event retraining for hours.
- The comparison sampler phase-marginalises while TMNRE does not; disagreement on ϕ_c is waved away.
- Coverage is checked only within the truncated region.
