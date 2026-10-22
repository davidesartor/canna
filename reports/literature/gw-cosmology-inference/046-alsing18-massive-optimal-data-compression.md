# Massive optimal data compression and density estimation for scalable, likelihood-free inference in cosmology

Alsing, Wandelt, Feeney, 2018, MNRAS 477, 2874 · [arXiv:1801.01497](https://arxiv.org/abs/1801.01497)

**Evidence: A.** Peer-reviewed (Monthly Notices of the Royal Astronomical Society); 161
citations in 8.8 yr (Semantic Scholar, 2026-10-19).

## What they do

Two-step compression: (I) data D → hand-chosen summaries d (power spectra etc.); (II) d → t = ∇_θ ln L_*(d), the score of an *approximate* (e.g. Gaussian) likelihood at a fiducial θ_*, giving one number per parameter. For a Gaussian with parameter-dependent mean only, t = ∇μ_*ᵀ C_*⁻¹ (d − μ_*) (= MOPED); with prior, add C_P⁻¹(μ_P − θ_*). Then DELFI: fit a Gaussian mixture to the *joint* p(θ, t) (2n dims) and slice at t_o; evidence comes free from the marginal mixture.

## What they find

JLA SNe (740 magnitudes → 6 numbers, 6-parameter wCDM + calibration). DELFI with K=12 GMM on 20k sims matches long-run emcee MCMC (10⁶ samples) with "no perceptible increase in width" ⇒ score compression effectively lossless here (Gaussian likelihood is true by construction). Posterior means converge to ≲0.05σ and stds to a few % after ≈8000 sims. PMC-ABC needed >2×10⁶ sims for a visibly broader posterior. Evidence ln P(t_o)=7.38 vs MultiNest 7.4(1).

## Implementation details

fiducial θ_* found by Fisher scoring θ_{k+1}=θ_k+F_k⁻¹t_k (converges in a few iterations). Derivatives by seed-matched finite differences. Approximate mean/cov/derivatives in the compressor can only cost optimality, never bias (inference after compression is fit to sims). Covariance floor 10⁻⁶ on GMM diagonals against over-fitting; choose K by convergence of the sample log-likelihood. Prior with correlations shaped like F⁻¹ to improve posterior/prior volume ratio (they say it helps the LFI, negligible effect on the posterior).

## Caveats

Validation is a case where the assumed Gaussian likelihood is exact, so "lossless" is by construction; score compression is only Fisher-optimal (local, unimodal); multimodal/non-Gaussian likelihoods may need more summaries.
