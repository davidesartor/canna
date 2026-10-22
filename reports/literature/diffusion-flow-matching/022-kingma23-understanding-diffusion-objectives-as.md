# Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation

Kingma & Gao, 2023, NeurIPS 2023 · [arXiv:2303.00848](https://arxiv.org/abs/2303.00848)

**Evidence: A.** Peer-reviewed (Neural Information Processing Systems); 369 citations in 3.6 yr
(Semantic Scholar, 2026-10-19).

## What they do

Every common diffusion objective is L_w = ½∫w(λ)E‖ε̂ − ε‖²dλ over λ = log-SNR.
- The optimum is invariant to the noise schedule except for its endpoints. The schedule only enters as an importance-sampling density p(λ), with estimator E_{λ~p}[w/p·‖ε̂ − ε‖²] (Eq. 134). It changes estimator variance, not the target.
- Thm 1: if w(λ) is monotone (non-increasing in λ, i.e. more weight at more noise), L_w = E_{p_w(t)}[KL(q(z_{t..1}|x)‖p(z_{t..1}))], which is the ELBO of data augmented with Gaussian noise of random level.
- Thm 2: dKL/dλ = ½σ²·(Fisher divergence), assuming a conservative score.

## What they find

- Implied weightings (Table 1):
  | Objective | Implied w(λ) |
  |---|---|
  | ELBO | 1 |
  | IDDPM | sech(λ/2) |
  | EDM | N(λ; 2.4, 2.4²)·(e^{−λ} + 0.25) |
  | v-prediction with cosine schedule | e^{−λ/2} (monotone) |
  | FM with OT path | e^{−λ/2} (monotone) |
  | InDI | e^{−λ}sech²(λ/4) |
  | P2 | sech(λ/2)/(1 + e^λ)^γ |
  | Min-SNR | sech(λ/2)·min(1, γe^{−λ}) |
- ImageNet64 FID, ε-prediction, DDPM sampler, sigmoid(−λ + k) weighting:
  | Weighting | FID |
  |---|---|
  | sech baseline | 1.85 |
  | k = 1 | 1.75 |
  | k = 2 | 1.68 |
  | k = 3 | 1.73 |
  | k = 4 | 1.80 |
  | k = 5 | 1.94 |
- With the EDM sampler:
  - ε-prediction: baseline 1.55, sigmoid 1.46; with the adaptive schedule, 1.44 for both sigmoid and EDM-monotonic.
  - EDM reproduction: 1.45 (paper reports 1.36); adaptive 1.43; adaptive plus EDM-monotonic 1.43.
  - v-prediction: baseline e^{−λ/2} 1.62, sigmoid 1.51, EDM-monotonic 1.45.
- Higher resolution (U-ViT), train/eval FID without guidance:
  - ImageNet128: Simple Diffusion 1.91/3.23 vs EDM-monotonic-shifted 1.75/2.88.
  - 256 px: 2.40/3.36. 512 px: 2.99/4.09. Both beat Simple Diffusion's 2.77 and 3.54.

## Implementation details

- Adaptive noise schedule (App. F): 100 bins over [λmin, λmax]. Each bin keeps an EMA (decay 0.999, initialised at 1) of w‖ε̂ − ε‖², and p(λ) is set proportional to it via a piecewise-linear inverse CDF, so the loss is spread evenly over λ.
  - Final FID unchanged; it converged faster in half the runs and equally fast in the rest. Its main value is that w can be changed without re-tuning the schedule.
- Sigmoid weighting with k = 2, or "EDM-monotonic" (the EDM weight held at its maximum below the argmax).
- Shift λ by log(64/d) at resolution d.
- Low-discrepancy t, Adam defaults, global gradient clipping at 1.

## Caveats

- Gains are about 0.1–0.2 FID, from single runs.
- Images only. App. J: weighting and other hyperparameters need re-tuning per dataset.
- The ELBO interpretation is for the noise-augmented data, not the clean data.
