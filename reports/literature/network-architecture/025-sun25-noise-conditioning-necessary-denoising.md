# Is Noise Conditioning Necessary for Denoising Generative Models?

Sun et al., 2025 · [arXiv:2502.13129](https://arxiv.org/abs/2502.13129)

**Evidence: A.** Peer-reviewed (International Conference on Machine Learning); 55 citations in
1.7 yr (Semantic Scholar, 2026-10-19).

## What they do

Remove the t input and measure the damage.

## What they find

| Model | FID with t | FID without t |
|---|---|---|
| EDM | 1.99 | 3.36 |
| DDIM ODE | 3.99 | 40.90 |
| FM Euler | 3.01 | 2.61 |
| uEDM | 2.04 | 2.23 |
Error bounds for the noise-blind model: DDIM 3e6, EDM 1e3, FM 1e2.

## Implementation details

—

## Caveats

The argument relies on x_t revealing the noise level, as it does for images noised toward a Gaussian. When the base is a draw from a structured distribution (e.g. a prior), x_t need not reveal t.
