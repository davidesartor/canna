# Fast and Credible Likelihood-Free Cosmology with Truncated Marginal Neural Ratio Estimation

Cole et al., 2021, JCAP 09 (2022) 004 · [arXiv:2111.08030](https://arxiv.org/abs/2111.08030)

**Evidence: B.** Peer-reviewed (Journal of Cosmology and Astroparticle Physics); 54 citations
in 4.9 yr (Semantic Scholar, 2026-10-19).

## What they do

Rewrite CMB likelihoods as stochastic simulators (CLASS spectra → Wishart sampling of Ĉ_ℓ at low ℓ, Gaussian above ℓ=52, f_sky via effective degrees of freedom) and run TMNRE. A linear compression C(x) is shared, and one 3×256 MLP classifier per 1-D or 2-D marginal is trained jointly. The BCE loss makes the summary maximise the data-averaged JSD between posterior and prior. Truncation rounds keep the region where each estimated marginal is above ε times its max, and stop when the kept prior mass ratio between rounds is above β=0.8.

## What they find

Planck-like forecast (ℓmax=2500, 6 ΛCDM params, prior ±5σ Fisher): ≤5000 sims, all marginals trained in ~10 min on a GPU, match an MCMC of 10⁵ sim calls; at 3000 sims, 1-D centres and 1σ widths are within 0.05σ. HiLLiPoP TT (10816 bins, 6 cosmo + 13 nuisance): 3000 sims with linear 10816→10 compression, against an MCMC needing ~5×10⁶ calls, so the cost does not grow with nuisance count. Σmν upper bound within 3% of MCMC at 5000 sims. Coverage on 1000 test sims: 68.27% HPD region contains the truth 70.5% of the time, 95.45% → 96.0% (slightly conservative). Undertrained networks are conservative. Wide prior (±25σ, 5000× volume): 9 rounds and 20k sims total (4× the narrow-prior cost) shrink the prior volume 5000× by round 6. Starting at 10k sims instead converges in 3 rounds at roughly the same 20k total. Simulation reuse: CMB+BAO posteriors retrained on the same 10⁴-sim bank at zero new simulator cost.

## Implementation details

Train the marginals you care about directly instead of the joint (cost nearly independent of nuisance count). Truncate 2-D jointly for parameter pairs known to correlate (Fisher), and 1-D otherwise. Grow the per-round sim count when the volume stops shrinking (factor <5). Coverage plotted as empirical ẑ vs nominal z with Jeffreys binomial bands (Beta(n−k+½, k+½)). Store noise-free simulations and add noise at training time, so a new noise model costs nothing. A linear compression is enough when the spectra are near-linear in θ inside the prior. Explicit sky masks are trivial in SBI: zero the masked pixels and recompute the ML spectrum.

## Caveats

Gaussian or Wishart summaries where the likelihood is known; truth is MCMC on the same simulator. Fiducial is Asimov data in the forecast case. Marginal-only: no joint structure beyond 2-D. Coverage is necessary, not sufficient.
