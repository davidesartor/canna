# Extracting overlapping gravitational-wave signals of Galactic compact binaries: a mini review

Niu, Zhao, 2024, arXiv v2 20 Nov 2024, gr-qc; Fundamental Research 6, 1360 (2026) · [arXiv:2406.12789](https://arxiv.org/abs/2406.12789)

**Evidence: B.** Peer-reviewed (Fundamental Research); 3 citations in 2.3 yr (Semantic Scholar,
2026-10-19).

## What they do

Review of GCB extraction in two families plus hybrids: (i) iterative subtraction with the ℱ-statistic + particle swarm optimisation (GBSIEVER, O(10⁴) GCBs from LDC1-4; LMPSO-CV for false candidates; coarse template + fine PSO); (ii) global fitting with RJMCMC (GBMCMC/ldasoft: PT, blocked Gibbs, ℱ-statistic proposals on (f0, λ, β), sideband/multimodal proposals, posterior-from-previous-epoch proposals; Eryn: ensemble stretch/group moves, delayed rejection, multiple-try, GPU); (iii) Strub's hybrid (DE/SLSP ML search, global re-optimisation, per-source MCMC, GPR); (iv) ML — only Korsakova 2024 (flows as priors, proposals, and posterior-sharing format). Also covers foreground separation (analytic Cornish–Robson fit, 17 Fourier-coefficient modulated model) and Galactic-structure science.
- **Key facts with numbers:** GCB = 8 parameters (A, f0, ḟ, ι, λ, β, ψ, φ0); ~6000–10000 detected WD+WD of ~10⁸ (Table 1, from the LISA Red Book); GBMCMC default prior N ∈ [0, 30] per bin, i.e. up to 240-D; resolvable cut SNR ≥ 8 at 1 yr in their Fig. 3. Littenberg 2020 is cited for "high correlations among overlapping signals are relatively rare", which is why per-source proposals work.

## Implementation details

Frequency-band segmentation with edge handling; ℱ-statistic maximisation over the 4 extrinsic amplitude parameters; proposals built from earlier-epoch posteriors (incremental data); flow-encoded posteriors for catalogue release.

## Caveats

A review; no new results. The ML section is thin (one paper) and predates SlotFlow, Delmond and others. Future-work list: non-stationary noise, glitches, gaps, mixed source types.
