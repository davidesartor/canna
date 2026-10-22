# On the Closed-Form of Flow Matching: Generalization Does Not Arise from Target Stochasticity

Bertrand et al., 2025, arXiv v2 Dec 2025; venue not stated in the text · [arXiv:2506.03719](https://arxiv.org/abs/2506.03719)

**Evidence: A.** Peer-reviewed (Neural Information Processing Systems); 51 citations in 1.4 yr
(Semantic Scholar, 2026-10-19).

## What they do

- With an empirical data distribution, the optimal FM velocity is closed form: û*(x,t) = Σ_i softmax_i(−‖x − t x⁽ⁱ⁾‖²/2(1−t)²)·(x⁽ⁱ⁾ − x)/(1−t). Following it exactly can only reproduce training points.
- The paper tests "generalisation through variance" (the noisy CFM target as the source of generalisation) and rejects it. Generalisation instead comes from the network failing to fit û*, mainly at early t.

## What they find

- Target stochasticity: the cosine between û* and u_cond = x₁ − x₀ approaches 1 very early in high dimension (before t ≈ 0.2 on CIFAR10). On Imagenette the fraction of samples with cosine > 0.9 jumps to ~100% as resolution grows. The stochastic regime is a short window at small t.
- Approximation error (CIFAR10 subsets from 10 to 10⁴ images, fixed 34M-parameter U-Net):
  - ‖u_θ − û*‖ is small with 10 samples (test FID > 10³, pure memorisation) and grows with n.
  - The error peaks at two times: t ≈ 0.15, where û* and u_cond start to correlate, and t → 1, where û* is non-Lipschitz.
  - n = 1000 has low test FID yet still memorises.
- Hybrid sampling (follow û* on [0, τ], then u_θ): LPIPS to the training set stays at u_θ's level for τ ≤ 0.2, and samples equal the memorised ones once τ ≥ 0.4. The image is fixed by t ≈ 0.4.
- Empirical FM (EFM), a self-normalised, Rao-Blackwellised target with the generating x₁ always in the M-sample minibatch (Prop. 2: unbiased for û*, lower conditional variance than u_cond):
  - Modest but steady gains in train and test FID (Inception and DINOv2) on CIFAR10 and CelebA64 as M grows (128 → 256 → 1000). M ≥ 1000 shows no further change.
  - M = |B| = 128 costs only a modest time overhead.

## Implementation details

- Rao-Blackwellise the regression target over a minibatch of candidate endpoints, always including the true x₁ to avoid self-normalised importance-sampling bias.
- Diagnose where in t the network departs from the optimal field.

## Caveats

- Empirical and image-only, with Gaussian base and linear paths.
- Generalisation here means not memorising a finite training set; with a simulator and fresh draws every step that regime does not arise.
