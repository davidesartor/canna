# Translation and Rotation Equivariant Normalizing Flow (TRENF) for Optimal Cosmological Analysis

Dai & Seljak, 2022, MNRAS 516, 2363 · [arXiv:2202.05282](https://arxiv.org/abs/2202.05282)

**Evidence: B.** Peer-reviewed (Monthly notices of the Royal Astronomical Society); 48
citations in 4.7 yr (Semantic Scholar, 2026-10-19).

## What they do

Conditional likelihood flow p(x|y) on 128² projected density fields. Each layer = isotropic Fourier-space convolution T̃(|k|) (cubic spline) followed by a pixelwise monotone RQ-spline Ψ; Jacobian is ∏Ψ′ · ∏_k T̃(k), inverse is trivial. Conditioning: a hypernetwork (MLP 2×512) outputs all O(100) spline params from y=(Ωm,σ8). Posterior via p(x|y)p(y) on a grid/sampler.

## What they find

GRF: learned T̃ and Ψ match analytic optimum (T̃ = 1/(a√P), Ψ = ax) to ~1e-3; likelihood matches analytic. FastPM N-body (512 h⁻¹Mpc, 128³ particles, 10 steps): figure of merit 995 (TRENF) vs 176 (power spectrum) on 100 test maps. Calibration: two-stage training gives 65/100 inside 68% and 95/100 inside 95%; generative-only (NLL) training is overconfident (>5% outside 95%). With 50% masking: FoM 507 (≈ halved, as expected for area), 93/100 within 95%. Ablation: non-isotropic kernels (D_N, N∈{8,16,32}, M∈{1,2,3}) give no gain and converge to a worse loss.

## Implementation details

(1) Two-stage training: NLL (generative) warm start, then discriminative −log p(y|x) with evidence p(x) estimated by importance sampling from a Gaussian at the MAP (ADAM-found per example) with covariance fitted from stage-1 posteriors, M=20 IS samples. Skipping stage 1 does not converge (bad proposal). (2) Latent-space whiteness test: map data → latent at true y; if the latent is white noise (1-pt PDF, P(k), bispectrum) the likelihood is optimal; at wrong y, structure (voids) shows up. (3) Masks: fill missing pixels with noise, then an affine coupling layer "inpaints" conditioned on neighbours; position-dependent nonlinearity Ψ_r for symmetry-breaking effects. (4) inverse-softplus preprocessing for positive fields.

## Caveats

2 params, toy-cheap simulations, small masks, only 100 test maps for coverage; NLE not NPE.
