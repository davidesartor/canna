# Neural posterior estimation of Galactic Binary signals for the LISA mission

Delmond, Korsakova, Oberlin, Marsat, Basset, Dobigeon, 2026, preprint ·
[arXiv:2606.29039](https://arxiv.org/abs/2606.29039)

**Evidence: C.** Not peer-reviewed; 0 citations in 0.3 yr (Semantic Scholar, 2026-10-19).
The claims are the authors' alone.

## What they do

NPE for single LISA galactic binaries, with a two-source overlap as a proof of concept. It
is the closest published problem to ours.

- **Data:** complex frequency-domain TDI Ã and Ẽ in a band: 64 bins per channel at low
  frequency (f_c = 1 mHz), up to 512 at high frequency.
- **Network input:** real and imaginary parts concatenated. There is no embedding network
  ("we do not employ any strategy of data embedding").
- **Flow:** a conditional neural spline flow with 12 steps (18 for two sources), each a
  10-block ResNet. Width 512 at low frequency (1.2 × 10⁸ parameters) and 1024 at high
  frequency (3.9 × 10⁸).
- **Parameters:** 8 per source. φ0 and ψ are inferred directly with uniform priors on
  [0, 2π] and [0, π], with no special treatment.
- **Alternative parameterisation tested:** the four JKS amplitudes 𝒜^μ, which absorb
  (A, ι, φ0, ψ) linearly, in place of the physical amplitude parameters.

## What they find

- **Low frequency:** close to MCMC. Median JS divergence < 5e-2, and the p-p test passes
  (1000 injections).
- **High frequency:** JS divergence 5e-2 to 1e-1. The f0, ḟ0, λ and β posteriors are
  "systematically broader" than MCMC, which they attribute to the phase parameters
  producing "more complex posterior structures that are harder to explore".
- **Wider f0 prior** (multi-band): f0 becomes harder to resolve.
- **Two sources:** mixed results. The unfavourable case is "much broader" and some
  parameters are "significantly biased".
- **JKS parameterisation is worse.** The 𝒜^μ posteriors are unimodal and near-Gaussian,
  but the flow learns them less well: "overly broad", with degraded sky localisation. The
  physical (A, ι, φ0, ψ) posteriors, bimodal included, are "very close" to MCMC.
- **Speed:** 10⁴ samples in under 1 s on an A100, against about 1 h for MCMC. Training takes
  about 1 week on an A100 (batch 2048, lr 2e-4).

## Implementation details

- Noise is drawn independently for the real and imaginary parts from the PSD. Training
  samples are kept only if the SNR is in [10, 100].
- The input is a flat real vector of 2 × channels × bins. The networks are entirely
  real-valued.

## Caveats

- An unreviewed, very recent preprint.
- There is no complex-valued or phase-aware architecture to compare against, so it says
  nothing about whether complex layers help. It is a baseline for a real-valued flat input.
- Calibration is reported only for the low-frequency single-source case.
- Read via the arXiv HTML. We did not find an explicit statement on how well φ0 alone is
  recovered, beyond the claim that the physical-parameter posteriors match MCMC.
