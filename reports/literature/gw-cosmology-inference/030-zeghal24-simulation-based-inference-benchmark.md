# Simulation-based inference benchmark for weak lensing cosmology

Zeghal et al., 2024, A&A 699 A327 · [arXiv:2409.17975](https://arxiv.org/abs/2409.17975)

**Evidence: B.** Peer-reviewed (Astronomy & Astrophysics); 16 citations in 2.1 yr (Semantic
Scholar, 2026-10-19).

## What they do

- Benchmarks explicit and implicit full-field inference on the same differentiable sbi_lens model: log-normal LSST-Y10-like convergence maps, 5 tomographic bins × 256², 6 wCDM parameters.
- Explicit: NUTS over θ plus all latent field modes z.
- Implicit: a ResNet-18 VMIM compressor to 6-D, then NLE (RealNVP), then NUTS.
- Also tests whether simulator gradients (SCANDAL/∂NLE, i.e. joint score matching) reduce the simulation budget.
- Metric: C2ST against 160k converged HMC samples, fitted with a KDE.

## What they find

- HMC needs 630k sims (400 effective samples; 126 sims per sample at max tree depth 6). Marginals alone converge at 63k sims (24 ESS), where C2ST is 0.76.
- Implicit: 100k sims for the compressor (not tuned, an upper bound) plus 1k for NLE.
- C2ST never goes below 0.6 even when the marginals visibly match, so the threshold is set at 0.6.
- Gradients: joint-score ∂NLE gives no gain over NLE because the joint score is dominated by latent noise. With *learned marginal* scores (an MLP regressing the joint score, whose minimiser is the marginal score), ∂NLE converges at ~400 sims, about half as many.
- MUSE would need ≥10 sims per marginal gradient, so it is not worth it here.
- App. B (sbi defaults: MAF 5×[50,50] tanh; 5 rounds for sequential methods):
  - NPE ≈ NLE > NRE.
  - Sequential methods beat amortised ones.
  - SNPE/SNLE reach C2ST 0.6 at 1000 sims.
- Too few sims gives an *overconfident* NLE posterior (Fig. 10).

## Implementation details

- The two-step split (lossless compression, then low-dimensional density estimation) is recommended as "more robust and safe" than pixel-level NPE.
- An ensemble of 7 flows for epistemic error.
- A ∂NPE with smooth flows was too simulation-hungry.
- Continuous normalizing flows trained by NLL "took a very long time to train", so they were not used.
- APT needs M = 10 atoms; M = 100 costs real training time. APT leaks mass outside the prior (cites TSNPE).
- C2ST is very sensitive to higher-order structure in 6-D. Check marginals and moments alongside it.

## Caveats

- Log-normal toy model, no systematics, a single observation x₀.
- NUTS only (MCLMC might be cheaper).
- The compressor budget was not explored.
- Near-Gaussian, unimodal posterior.
