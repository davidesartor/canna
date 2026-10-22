# Lossless, Scalable Implicit Likelihood Inference for Cosmological Fields

Makinen et al., 2021, JCAP 2021(11):049 · [arXiv:2107.07405](https://arxiv.org/abs/2107.07405)

**Evidence: B.** Peer-reviewed (Journal of Cosmology and Astroparticle Physics); 36 citations
in 5.3 yr (Semantic Scholar, 2026-10-19).

## What they do

IMNN redone in JAX with a differentiable simulator: summary derivatives come from autodiff, ∂μ_f/∂θ = (1/n_s) Σ (∂x/∂d)(∂d/∂θ), instead of finite-difference sim pairs, and fresh simulations are generated every epoch ("on-the-fly"), so no validation set and no Δθ to tune. Loss −|det F| + r(Λ_C)·Λ_C, where Λ_C = ½(‖C_f−1‖² + ‖C_f⁻¹−1‖²) pins the summary covariance to the identity, with an adaptive weight r = λΛ_C/(Λ_C + exp(−αΛ_C)). Iterative retraining: if the IMNN estimate on the target is more than 1σ from θ_fid, retrain at θ_fid ← θ_MLE.

## What they find

128² Gaussian field P(k)=Ak^(−B): analytic det F = 1.147×10⁶ (½ ln det F = 6.98 nats). The inception CNN reaches 90% of the Fisher information in 550 epochs and 98% by 1500, with 200 fresh sims per epoch. ABC with these summaries (20,000 sims, 1279 accepted, ε=0.05, about 20 min on one GPU) matches the field-level likelihood posterior. Log-normal Eisenstein–Hu fields (Ω_c, σ8): 100% of the field information at a deliberately bad fiducial (0.6, 0.6). One retraining at the score estimate (0.229, 0.745) gives 98% there and visibly tighter posteriors. DELFI (4 MAFs × 5 MADEs × 2×50, 1000 prior sims + 3 sequential rounds = 4000 sims per target) agrees with ABC (50,000 sims per target), i.e. "≳100× fewer simulations".

## Implementation details

Inception blocks (5², 3², 1² convolutions + maxpool in parallel, stride-4, 55 filters) trained most efficiently among the architectures tried. λ=10, α=0.95. Stop when det F has not improved for 500 epochs. Fresh sims each epoch remove overfitting of the Fisher estimate, which was the 2018 paper's failure mode. A Gaussian approximation from the IMNN Fisher gives instant error bars and a proposal for ABC/SNPE.

## Caveats

2 parameters only, 2D fields, Gaussian or log-normal (information known analytically). The retraining loop is target-specific, so it is not amortised. No coverage tests. Optimality is local to θ_fid; the fact that a poor fiducial still worked is shown for one example only.
