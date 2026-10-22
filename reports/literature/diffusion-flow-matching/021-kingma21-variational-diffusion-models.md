# Variational Diffusion Models

Kingma, Salimans, Poole, Ho, 2021, NeurIPS 2021 · [arXiv:2107.00630](https://arxiv.org/abs/2107.00630)

**Evidence: A.** Peer-reviewed; 1625 citations in 5.3 yr (Semantic Scholar, 2026-10-19).

## What they do

z_t = α_t x + σ_t ε with SNR(t) = α²/σ², σ² = sigmoid(γ_η(t)), and γ_η a learned monotone network.
- Discrete-time VLB: (T/2)E[(SNR(s) − SNR(t))‖x − x̂‖²]. More steps T always tighten it.
- Continuous limit: −½E[SNR′(t)‖x − x̂‖²] = ½∫_{SNRmin}^{SNRmax}‖x − x̃(z_v)‖²dv. The VLB depends only on the endpoint SNRs; the shape of the schedule in between does not matter (invariance).
- A weighted version w(v) (Eq. 19) contains the DDPM and NCSN objectives as special cases (App. K); L_simple corresponds to w = 1/γ′.

## What they find

- Bits per dim: CIFAR-10 2.65 without augmentation, 2.49 with extensive augmentation (flips, rotations, channel swaps). ImageNet32 3.72, ImageNet64 3.40.
  - Previous best: Sparse Transformer 2.80, ScoreFlow 2.83. VDM passes 2.80 after 2.5 h of training.
- CIFAR-10 FID: 7.41 for the likelihood-tuned model; 4.0 with the Ho+20 weighting.
- Discrete vs continuous (Table 3): T = 10 gives 4.31, T = 1000 gives 2.67, continuous training evaluated at T = ∞ gives 2.65.
- Variance of the VLB estimate (Fig. 4b): learned schedule 0.53, log-SNR-linear 6.35, β-linear 31.6, α-cosine 31.1.
- Fourier features: a large likelihood gain, but only together with learned SNR endpoints.
  - With a fixed schedule, max log-SNR is ~8 and BPD stays above 4; learned endpoints reach log-SNR 13.3.
  - The same features give no gain in PixelCNN++.

## Implementation details

- Fourier features sin/cos(2ⁿπz) concatenated to the network input, best with n = 7–8.
- The network is conditioned on γ_t (rescaled), not on t.
- Low-discrepancy t: t_i = mod(u0 + i/k, 1) across the batch, a free variance reduction.
- The schedule interior is trained to minimise the variance of the loss estimator, using the gradient of the squared loss at no extra cost. The endpoints are trained on the VLB. The learned schedule spends more time at high SNR.
- Numerical stability: expm1 and softplus parametrisations make fp32 sufficient (DDPM needed fp64).
- Optimisation:
  - Adam lr 2e-4, β2 = 0.99 (higher was unstable).
  - EMA 0.9999, weight decay 0.01, dropout 0.1, 10M steps.
  - Attention blocks removed except in the middle (overfitting).

## Caveats

Images only. The Fourier-feature gain is tied to 8-bit discrete data at very high SNR, where fine bits matter. The headline numbers are likelihood, not sample quality.
