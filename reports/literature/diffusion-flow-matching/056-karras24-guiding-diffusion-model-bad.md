# Guiding a Diffusion Model with a Bad Version of Itself

Karras, Aittala, Kynkäänniemi, Lehtinen, Aila, Laine, 2024; NeurIPS 2024, not on arXiv page · [arXiv:2406.02507](https://arxiv.org/abs/2406.02507)

**Evidence: A.** Peer-reviewed (Neural Information Processing Systems); 353 citations in 2.4 yr
(Semantic Scholar, 2026-10-19).

## What they do

"Autoguidance" uses D_w = w·D₁ + (1−w)·D₀, where D₀ is a *worse* model with the same conditioning (smaller and/or trained for less time). The weak model makes the same errors as the strong one, only more strongly, and spreads its density more widely. So D₁−D₀ points toward the data manifold and isolates the strong model's errors, without the class-overemphasis of CFG. This is shown on a 2-D fractal GMM with 2,032 components: CFG drops whole branches, naive score scaling contracts isotropically, and autoguidance concentrates on the manifold without dropping branches.

## What they find

ImageNet-512, EDM2-S: FID 2.56 → CFG 2.23 → guidance interval 1.68 → **autoguidance (XS guide, T/16 training) 1.34**. FD_DINOv2 68.6 → 36.7. EDM2-XXL: 1.91 → CFG 1.81 → interval 1.40 → **AG (M, T/3.5) 1.25**. ImageNet-64: 1.58 → 1.01. Unconditional ImageNet-512: 11.67 → 3.86.
- Ablations (EDM2-S FID): full 1.34; same EMA for both models 1.53; **reduced training only 1.51; reduced capacity only 2.13**. Most of the gain comes from the less-trained guide.
- Synthetic degradations do nothing: dropout 5%→10% gives 2.55, input noise 10%→20% gives 2.56, and mismatched degradation pairs are best at w=1 (no guidance).

## Implementation details

Optimal w ≈ 2–2.5, and results are fairly insensitive to w. Tune the EMA lengths of the main and guide models separately (0.070 / 0.125). Best guide capacity is one size below the main model, trained for 1/16–1/3.5 of the iterations. Extra training cost is about 3.6% (S/XS) or 11% (XXL/M). Things that did not work: less training data for the guide, stacking with a guidance interval (it reduced the benefit, suggesting autoguidance helps at all noise levels), synthetic corruption of the main model, and a lower-precision quantised guide.

## Caveats

Purely FID/FD-based, found by about 30k metric evaluations of grid search. There is no theory of the sampled distribution, so it is still extrapolation and not a calibrated posterior.
