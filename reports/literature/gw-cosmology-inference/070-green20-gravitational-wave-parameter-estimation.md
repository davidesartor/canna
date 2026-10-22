# Gravitational-wave parameter estimation with autoregressive neural network flows

Green, Simpson & Gair, 2020, PRD 102, 104057 · [arXiv:2002.07656](https://arxiv.org/abs/2002.07656)

**Evidence: A.** Peer-reviewed (Physical Review D); 111 citations in 6.7 yr (Semantic Scholar,
2026-10-19).

## What they do

- First normalizing flows for GW posteriors.
- It compares three models:
  - a MAF conditioned on strain;
  - a CVAE (as in Gabbard);
  - **CVAE+**, i.e. a CVAE with IAF on the encoder, MAF on the prior and MAF on the decoder.
- Only CVAE+ captures the multimodal φ₀.

## What they find

- **Data:**
  - Single-detector "+" polarisation of IMRPhenomPv2.
  - 1 s at 1024 Hz; 256 Hz caused Gibbs ringing.
  - Whitened, then divided by √(2Δt) so noise is N(0,1) per sample.
- **Parameters:** 5-D (m₁, m₂, φ₀, t_c, d_L).
- **Training set:** 10⁶ fixed (x, h) pairs, noise added fresh at train time. 10⁵ pairs gave "slightly reduced but acceptable" results. Median SNR 25.8.
- **Model sizes and train times:**
  - MAF: 5 MADE × 3×512, 4.0M params, 6.5 h.
  - CVAE: 31.6M, 5.6 h.
  - CVAE+: 18.8M, 14.8 h.
- **Validation loss:**
  - MAF −4.42 vs CVAE −4.31 (an upper bound).
  - CVAE+ −6.95, a big gap that comes from capturing φ₀.
  - p-p KS p-values > 0.35 for CVAE+.
  - KL to the emcee posterior ≈ 0.1 nats.
- **8-D model** (aligned spins and θ_JN): CVAE+ with latent 16, 10 MADEs, 25.2M params; KS > 0.2. The **φ₀ posterior equals the prior** because of the χ₁–χ₂–φ₀ degeneracy, which MCMC confirms.

## Implementation details

- Standardise both x and y.
- **Batch norm** between MADE blocks is essential for stability; reset the running statistics to full-training-set values before validation.
- **Cyclic KL annealing** (10⁻⁵, 1/3, 2/3, 1, 1, 1 per epoch, ×4 cycles), and a free-bits threshold that ignores the KL below 0.2.
- The IAF encoder is conditioned on x and h only, not y: "high-dimensional y overpowered x".
- Different lrs for the Gaussian parts (1e-4) and the flows (4e-4), halved every 80 epochs.
- Batch 512, 250 epochs.
- **Failure mode:** affine MAF layers cannot reliably resolve the φ₀ multimodality. Deeper MAFs sometimes did, but not reproducibly.

## Caveats

- Fixed PSD, single detector, time domain, narrow prior.
- A CNN embedding was suggested but not used.
