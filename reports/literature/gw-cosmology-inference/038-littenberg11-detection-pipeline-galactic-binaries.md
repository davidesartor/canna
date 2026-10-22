# A detection pipeline for galactic binaries in LISA data

Littenberg, 2011, PRD 84 063009 · [arXiv:1106.6355](https://arxiv.org/abs/1106.6355)

**Evidence: B.** Peer-reviewed (Physical Review D); 50 citations in 15.3 yr (Semantic Scholar,
2026-10-19).

## What they do

BAM-style windowed RJMCMC over (f0, ḟ, θ, φ, A, ι, ψ, φ0) per source, with AE channels only (T dropped below f*), per-narrowband-segment noise scale η_i ~ N(1, 4σ_η²), galaxy-density sky/distance prior, uniform ψ, φ0 on [0, 2π], cos ι on [−1, 1]. Phases: search (F-stat, simulated annealing, parallel tempering, "lazy" delayed rejection), characterisation (RJMCMC with proposals from binned search chains), evaluation (Bayes factor threshold 12:1).

## What they find

MLDC-4 training data: ~9000 sources recovered (search halted early, f < 10 mHz); >90% have correlation > 0.9 with an injection; parameter-error histograms peak at zero bias, but "orientation parameters of the binary are typically not well measured, hence the larger tails." D_L measured for ~1000 binaries (SNR > 15, ḟT² ≥ 1).

## Implementation details

(1) annealing schedule β = (1/T_max)·T_max^{i/τ}, with T_max ≈ SNR² estimated as (s|s) − 4N from excess power; effective SNR under tempering ≈ SNR/√T. (2) Secondary modes at integer multiples of the LISA orbital modulation f_m = 1/yr (peaks every 2 Fourier bins at f_m T), with overlap ≳ 70% with the injection; fix: propose f → f + n f_m, n ~ U[−6, 6], let the chain relax sky for a few steps, then accept/reject (biased, search only). (3) Windows with acceptance region + "wings" so templates can absorb leakage from neighbours without double counting. (4) Hold fixed sources far (in bins) from the newly added one; re-fit only overlapping neighbours. (5) RJ proposals from search chains binned on a Fisher-scaled 8-D grid.

## Caveats

MCMC pipeline, no NPE; no posterior calibration study; galaxy-only training data.
