# Sequential simulation-based inference for extreme mass ratio inspirals

P. S. Cole, J. Alvey, L. Speri, C. Weniger, U. Bhardwaj, D. Gerosa, G. Bertone, 2026, PRD 113, 063030 · [arXiv:2505.16795](https://arxiv.org/abs/2505.16795)

**Evidence: B.** Peer-reviewed (Physical Review D); 8 citations in 1.4 yr (Semantic Scholar,
2026-10-19).

## What they do

TMNRE (swyft / peregrine fork) for one Schwarzschild EMRI from broad priors, used as a search: learn all 1-D marginal likelihood-to-evidence ratios by binary classification, truncate each marginal prior where the ratio < 1e-5, resimulate, repeat.
11 parameters, all uniform (Table 1): m1/1e5 U[4, 9], m2 U[20, 60], p0/m1 U[8.5, 14], e0 U[0.05, 0.6], cos theta_K, phi_K, d_L U[87.5, 500] Mpc, Phi_phi0 and Phi_r0 U[-pi, pi], cos theta_S, phi_S. FEW time-domain adiabatic waveform (581 modes), fastlisaresponse TDI-1 A/E on ESA orbits (T dropped), 0.1 yr at dt = 10 s (3.15e5 samples per channel), stationary Gaussian noise with 1-yr galactic confusion fit. The p0 prior is chosen so the segment captures the last part before plunge, and the close d_L range keeps the SNR high (fiducial SNR ~640).

## What they find

After 6 rounds the 2-sigma relative half-widths are m1 0.3%, m2 2%, p0 0.3%, e0 2%, versus 1e-3–1e-2 % from MCMC started near the truth, i.e. 10–1000x too wide. Initial phases Phi_phi0 and Phi_r0 come back exactly the uniform prior. Comparisons at an equal budget of 1.5e5 likelihood calls: differential evolution never finds the true region; an eryn MCMC (32 walkers x 4 temps x 4687 steps, 49 GPU-h) on the round-5 truncated prior does not converge. p-p over 500 round-6 injections is roughly diagonal, slightly overconfident for m1, p0, d_L and theta_K. Earlier-inspiral injections (SNR ~70) shrink the volume by only 3–55x.

## Implementation details

Input is 3 channels (time array, A, E) -> U-Net -> linear compression to 16 features per parameter per channel -> swyft residual log-ratio heads, one per 1-D marginal (2-D marginals optional). AdamW lr 1e-4, batch 128, 90/10 split, early stopping patience 7. "Noise shuffling": noise realisations are permuted within each batch every epoch. 6 rounds x 1.5e5 simulations; ~14 h per round on one A100 (~3 h simulation). Needs a 494 GB node; longer than 0.1 yr does not fit.
noise shuffling within a batch as cheap noise augmentation; truncated-prior rounds as a search stage; restoring the best-validation weights; an FFT input gives comparable results.

## Caveats

- The phases stay uninformative; attributed to the network resolving harmonic frequency evolution but not relative phases (the phases enter as m Phi_phi + n Phi_r).
- Truncation can only clip box edges, so with multimodal marginals it saturates early.
- Precision saturates because raw high-dimensional data need either far more examples or better compression.
- Independently trained 2-D marginals come out much broader than the 1-D ones for multimodal parameters.
- Summed losses across marginals overfit the converged heads.
- **Label switching / multiple sources:** single EMRI; no source confusion (the global fit is future work). Marginal ratios avoid any joint-ordering issue by construction.
- Single injection in the main text; the MCMC baseline is deliberately budget-starved; "proposal distributions" are not claimed to be posteriors.
