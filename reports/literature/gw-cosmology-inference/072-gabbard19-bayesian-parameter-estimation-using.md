# Bayesian parameter estimation using conditional variational autoencoders for gravitational-wave astronomy

Gabbard et al., 2019, Nature Physics 18, 112–117 (2022) · [arXiv:1909.06296](https://arxiv.org/abs/1909.06296)

**Evidence: A.** Peer-reviewed (Nature Physics); 204 citations in 7.1 yr (Semantic Scholar,
2026-10-19).

## What they do

VItamin, a CVAE trained on the cross-entropy / ELBO bound.
- The encoder r₁(z|y) outputs a 32-component Gaussian mixture in a 15-D latent space.
- The decoder r₂(x|y,z) outputs physically tailored per-parameter distributions.
- Applied to 3-detector BBH inference over 14–15 parameters.

## What they find

- 250 test injections; 8000 samples in about 0.1 s, roughly 6 orders of magnitude faster than bilby samplers (~2·10⁴–5·10⁵ s).
- Worst-case 1-D P-P p-value is 0.047 for VItamin, versus 0.004–0.007 for the classical samplers.
- The JS divergence to dynesty is comparable to ptemcee-vs-dynesty, except for φ₁₂ and ψ.
- Training takes about 7 days on a V100.

## Implementation details

- **Phase/polarisation reparametrisation** (after Jones 2015): X = ((ψ + φ₀)/π) mod 1 and ψ′ = ψ mod π/2. This removes the exact ψ/φ₀ degeneracy modes; it is valid only without higher modes.
- **Cyclic parameters**: each is output as a 2-D point (angle = value), so there is no boundary at the wrap. The sky uses a von Mises–Fisher distribution, other cyclic parameters a von Mises, and bounded parameters a Gaussian truncated at the prior edges.
- **Annealing**: the KL term is annealed logarithmically 0→1 between epochs 100 and 300 to avoid posterior collapse (KL stuck near 0). The truncation bounds are annealed over the same window, from [−10, 11] to [0, 1].
- **Data**: 10⁷ whitened training waveforms at 1024 Hz over 1 s. Every reuse of a waveform gets a fresh noise realisation, and new waveform sets are augmented in amplitude, phase and arrival time.
- **Optimiser**: batch 1500, lr 1e-4, ~3·10⁴ "epochs" of 2·10⁴ samples.
- **Network**: a 1-D CNN shared across the three networks, with the x input appended after the convolutions.
- **Phase handling**: φ₀ is excluded from the comparisons because the benchmarks marginalise it analytically.

## Caveats

- Time series is only 1 s at 1024 Hz.
- The m₁ > m₂ ordering is imposed by the prior.
- The mixture-of-Gaussians latent space can underfit sharp, non-Gaussian structure.
- A CVAE gives no tractable density, so there is no importance sampling (IS) correction.
