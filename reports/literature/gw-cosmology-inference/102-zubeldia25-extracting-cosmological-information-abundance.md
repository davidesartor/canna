# Extracting cosmological information from the abundance of galaxy clusters with simulation-based inference

Zubeldia, Bolliet, Challinor, Handley et al., 2025, arXiv; journal not confirmed from HTML · [arXiv:2504.10230](https://arxiv.org/abs/2504.10230)

**Evidence: B.** Peer-reviewed (Physical Review D); 5 citations in 1.5 yr (Semantic Scholar,
2026-10-19).

## What they do

- NPE (sbi NPE-C, MAF, amortized) on SO-like tSZ cluster catalogues, about 16k clusters per catalogue, 10 parameters.
- Catalogues come from the cosmocnc generator: HMF sampling, two-layer log-normal scatter, and a q > 5 cut.
- Each catalogue is compressed to a 21-D vector: 4×5 counts in (q, z) cells plus a stacked CMB-lensing S/N. This is shown to be effectively lossless in prior work.
- Validated against the exact unbinned+stacked likelihood run with Cobaya MCMC, then applied to real Planck MMF3 data (439 clusters, 9 parameters, 5k sims).

## What they find

- SO-like setup, SBI vs likelihood: |Δμ| < 0.2σ for all 10 parameters; σ8 0.081σ, ω_c −0.163σ.
- Amortized bias over 100 test catalogues is under 0.1σ for every parameter: σ8 −0.107 ± 0.086σ, ω_c 0.031 ± 0.063σ.
- Learning rate and N_sim sweep (η 5e-3 to 5e-6; N_sim 500 to 10k):
  - Converged (under 0.1σ change on doubling N_sim) at N_sim = 10k for η of 5e-3 and 5e-4.
  - Widths agree with the likelihood within 5% (σ8) and 10% (ω_c) for N_sim > 2k.
  - η = 5e-6 is clearly worse.
  - Results are not monotone in N_sim, because of training stochasticity.
- Planck MMF3:
  - σ8 = 0.756 ± 0.036 (SBI) vs 0.756 ± 0.034 (likelihood); Ω_m 0.342 vs 0.341.
  - Every parameter agrees within 0.07σ, except σ_lnY at 0.20σ (table).
- Cost: likelihood 12 s per evaluation, 180 h wall / 13,680 CPU-h. SBI 441 s per catalogue, 16.1 h wall / 9,807 CPU-h, so 11.2× faster in wall time.
- Adding correlated scatter or a third observable does not increase generation time.

## Implementation details

- Validate amortized NPE against an exact likelihood on one reference realization (Δμ/σ and σ ratio), plus a bias average over 100 realizations at a fixed truth.
- Sweep learning rate × N_sim, and define convergence as under 0.1σ change on doubling N_sim.
- Simulation-only goodness of fit: regenerate 100 catalogues at the posterior mean and compare a Cash statistic and the stacked observable to their simulated distributions.
- Patience of 50 epochs.

## Caveats

- A single fixed truth for the bias test, so this is not SBC over the prior.
- The data are low-dimensional (21-D) and the posteriors near-Gaussian, an easy regime.
- No coverage or TARP.
