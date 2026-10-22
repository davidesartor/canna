# Stable Target Field for Reduced Variance Score Estimation in Diffusion Models

Xu, Tong, Jaakkola; MIT; arXiv v2 Feb 2023, ICLR 2023 · [arXiv:2302.00670](https://arxiv.org/abs/2302.00670)

**Evidence: B.** Peer-reviewed (International Conference on Learning Representations); 50
citations in 3.7 yr (Semantic Scholar, 2026-10-19).

## What they do

- DSM target ∇log p_{t|0}(x_t|x) is a 1-sample MC estimate of E_{p_{0|t}}[·] = ∇log p_t. Its variance, V_DSM(t) = E_{x_t}[Tr Cov_{p_{0|t}}(∇log p_{t|0}(x_t|x))] (Eq 4), peaks where several modes influence x_t comparably.
- **STF target (Eq 5/6):** self-normalised importance sampling with proposal p_0 over a reference batch {x_i}_{i=1}^n ~ p_0 (x_t is made from x_1):
  v(x_t) = Σ_k [p_{t|0}(x_t|x_k) / Σ_j p_{t|0}(x_t|x_j)] ∇log p_{t|0}(x_t|x_k).
  For Gaussian kernels N(x, σ_t²I): (1/σ_t²) Σ_k softmax_k(−‖x_t − x_k‖²/2σ_t²)(x_k − x_t).
- n = 1 recovers DSM.
- **Thm 2:** V_STF(t) ≤ (bound with an f-divergence term D_f(p_{0|t}‖p_0)); when n ≫ d and p_{0|t} ≈ p_0, V_STF ⪅ V_DSM/(n−1). The D_f term dominates near data, so the reduction is real only in Phases 2–3.
- Plugs into any t-density q_t and weight λ(t) unchanged (VE, VP, EDM).

## What they find

- **CIFAR-10 (FID / IS / NFE):**
  - EDM NCSN++: 1.98 → 1.90 (35 NFE). EDM DDPM++: 1.97 → 1.92.
  - VE RK45: 8.90 → 5.51; VE PC: 2.75 → 2.66.
  - VP DDIM: 5.16 → 5.06; VP RK45: 2.90 → **2.99 (worse)**.
- **CelebA 64² (VE):** RK45 7.56 → 5.34; PC 9.13 → 8.28.
- **Training speed:** comparable FID with **3.6× less wall-clock** on CIFAR-10 VE (25 h vs 90 h). Smaller on CelebA; the authors attribute it to fewer well-separated modes.
- **Seed stability:** FID std across seeds 4.41 → 0.06 (VE RK45).
- **Reference batch:** n = 4096 (CIFAR) / 1024 (CelebA). FID-1k improves monotonically with n (Fig 5).
- **Overhead:** memory 36.05 → 40.64 GB/GPU; small wall-clock overhead (no extra network passes).
- **Variance:** measured V_STF ≈ V_DSM/(n−1) on the two-Gaussian toy; large but sub-1/n reduction on CIFAR-10-4096.

## Implementation details

- Draw a large reference batch, subsample the training batch from it, and compute softmax-weighted targets over the whole reference batch. Pure arithmetic, no network calls.
- The true pair x_1 is always in the reference batch, so the weights never vanish entirely.

## Caveats

- Unconditional image models only. The proposal is the data marginal p_0, which is useless when p_{0|t} is far narrower than p_0 (ESS → 1).
- Bias at finite n. VP-RK45 got slightly worse.
- Gains are largest for VE with poor samplers. On the strong EDM baseline the gain is 0.05–0.08 FID.
