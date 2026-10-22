# Neural density estimation for Galactic Binaries in LISA data analysis

Korsakova, Babak, Katz, Karnesis, Khukhlaev, Gair, 2024, arXiv v1 21 Feb 2024, gr-qc · [arXiv:2402.13701](https://arxiv.org/abs/2402.13701)

**Evidence: B.** Peer-reviewed (Physical Review D); 4 citations in 2.7 yr (Semantic Scholar,
2026-10-19).

## What they do

Unconditional (not amortised) normalising-flow density fits to samples, for three uses: (1) a Galaxy prior/proposal over (A, β, λ) from the Sangria population; (2) a proposal from an existing posterior (time-iterative analysis with 2–4 month data cadence, or from hot parallel-tempering chains, or for RJMCMC/product-space birth); (3) a compact catalogue format (ship flow weights instead of samples).
GB set {A, f0, ḟ, φ0, ι, ψ, β, λ}; Galaxy fit uses (A, β, λ) only. A "generic" broadened Galaxy is built by replacing per-bin counts N ≥ 2 in a 3-D histogram with ⌊log N⌋.

## What they find

- Galaxy fit: > 3×10⁷ Sangria GBs; training 10 min to several hours depending on tolerated outliers; corner plots "almost identical" (no metric).
- Posterior fits for 4 verification binaries (AM CVn, ES Cet, V803 Cen, ZTF J1539; 1.25–4.8 mHz), trained on ptemcee-style MCMC samples: corner plots overlap; log-likelihood distributions match. 1000 epochs ≈ 2.7 h on one V100 (~10 s/epoch); truncating at 100 epochs (~17 min) gives "slight drop" in accuracy.
- Model size ~10 MB per source.

## Implementation details

`lisaflow` package; 4 repeated blocks of LU linear + RQ-spline coupling (NSF) + RealNVP affine coupling; ResNet conditioners with 256 hidden, ReLU, dropout 0.2, batch norm; 4 ResNet layers in NSF, 2 in RealNVP.

## Caveats

Training cost (hours per source) too high for on-the-fly proposal refits across ~10⁴ sources; early-data posteriors favour nearby sources, so amplitude/sky proposals built from 2 months of data cover 2-year sources poorly; a joint posterior across overlapping sources would be needed for catalogues and would not scale linearly — named but not built. No quantitative divergence numbers anywhere; single-source only; no label-switching treatment.
