# Simpler Diffusion (SiD2): 1.5 FID on ImageNet512 with pixel-space diffusion

Hoogeboom et al., 2024, CVPR 2025 · [arXiv:2410.19324](https://arxiv.org/abs/2410.19324)

**Evidence: A.** Peer-reviewed; 83 citations in 2.0 yr (Semantic Scholar, 2026-10-19).

## What they do

Decouple time sampling (noise schedule λ_t) from loss weighting w(λ): L=E_{t~U}[−dλ/dt · w(λ_t)‖x−x̂‖²], invariant to the schedule except for Monte-Carlo variance. Use the sigmoid weighting w̃(λ)=σ(b−λ) on ε-mse, which in x-mse is e^b σ(λ−b) (Eq 4) — monotonically *increasing* in λ, i.e. cleaner inputs (easier problems) never get less weight. Bias b shifts by ≈ −2 log 2 ≈ −1.4 (they use ~−1.5) per doubling of resolution. Plus flop-heavy scaling (smaller patches rather than more channels) and Residual U-ViTs (one skip per level, identity at init).

## What they find

Table 1a (FID vs bias, flop-heavy): 128²: b=1 1.26, 0 1.30, −1 1.35, −2 1.49. 256²: b=−2 1.42 best (−3 1.47, 0 1.46, 1 1.56). 512²: −4 1.63, −3 1.48, −2 1.55, 0 1.94. 1024²: −5 1.82, −4 1.75, −3 2.32. So optimal b moves ≈ −1.5 to −2 per resolution doubling, slightly more aggressive than theory. Power (multiscale Haar) loss gives no gain over a tuned sigmoid bias (best 2.07 vs 2.19 for b=−3 at 0 levels; lowering bias ≈ adding downsample levels). Sigmoid b=−3 beats ε-mse, multiscale ε-mse, EDM and EDM-monotonic weightings over all of training (Fig 5). Training noise schedule (cosine vs shifted vs interpolated) has little effect with sigmoid weighting (Fig 6); EDM's *narrow* log-SNR training distribution degraded to ~9 FID — "a wide logsnr range is crucial". Video (Kinetics600 FVD): sigmoid b=0 2.3, b=+2 3.2, b=−2 2.5, ε-mse 5.7. Final ImageNet512 1.48 (16-step distilled 1.50); 2× channels overfits (1.9; 1.6 with augmentation) vs flop-heavy 1.54.

## Implementation details

Code given: weight = −0.5·dλ/dt·e^b·sigmoid(λ−b) on x-mse, dλ/dt via jax.jvp. Cosine-interpolated logsnr code with logsnr ∈ [−10,10] before shifting. Sampler variance σ_{t→s}^γ σ_{t|s}^{1−γ}, γ=0.3. Removing block skips costs 0.2 FID for small models, neutral/beneficial for large; decoder-heavy beats encoder-heavy.

## Caveats

Images/video only; the weighting–schedule decoupling relies on Gaussian diffusion where all schedules are reparametrisations of each other; bias tuned per resolution by grid search.
