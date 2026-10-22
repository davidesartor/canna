# Stochastic Interpolants with Data-Dependent Couplings

Albergo, Goldstein, Boffi, Ranganath, Vanden-Eijnden, 2023; ICML 2024 · [arXiv:2310.03725](https://arxiv.org/abs/2310.03725)

**Evidence: A.** Peer-reviewed (International Conference on Machine Learning); 111 citations in
3.0 yr (Semantic Scholar, 2026-10-19).

## What they do

I_t = α_t x0 + β_t x1 + γ_t z with base drawn conditionally, x0 = m(x1) + σζ where m(x1) is the corrupted observation (masked/low-res); velocity b_t learned by square loss; with conditioning ξ the flow maps ρ0(·|ξ) to ρ1(·|ξ) (Cor. A.1). Transport-cost bound E|İ|² drops from 2E|x1|²+dσ² (independent) to dσ² when m(x1)=x1.

## What they find

ImageNet-256 in-painting FID 1.35 (Gaussian base, same model) → 1.13 (coupled base); super-res 64→256 FID 2.05 (vs I²SB 2.70, ADM 3.10) — but no matched independent-coupling SR baseline. No ablation on σ or γ.

## Implementation details

in-painting γ=0, masked pixels filled with N(0,1); conditioning also appended as input channels (both coupling *and* conditioning); Adam 2e-4, dopri5.

## Caveats

σ, steps, batch not reported; only images; conditional correctness argued qualitatively.
