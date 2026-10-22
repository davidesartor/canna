# Neural Posterior Estimation with guaranteed exact coverage: the ringdown of GW150914

Crisostomi et al., 2023, PRD 108 044029; journal ref not visible in the arXiv HTML · [arXiv:2305.18528](https://arxiv.org/abs/2305.18528)

**Evidence: B.** Peer-reviewed (Physical Review D); 29 citations in 3.4 yr (Semantic Scholar,
2026-10-19).

## What they do

- Truncated-prior (TMNRE-style) locally amortised NPE with a MAF over 6 ringdown parameters: M_f, χ_f, A220, φ220, A221, φ221.
- Calibrate afterwards. For a grid of fixed θ₀, simulate 10³ noise realisations and compute the HPD credibility γ of θ₀ under q(θ|x). This gives a per-θ₀ map γ̂(θ₀) of the credibility needed for the target frequentist coverage.
- The confidence region for the real data is {θ₀ : γ(θ₀, x) < γ̂(θ₀)} (a Neyman construction).

## What they find

- Uncalibrated, A221 = 0 is excluded at 95%. Calibrated, it is inside the 95% region, so the overtone evidence is only about 2σ.
- The network under-covers at 95% for large A221, so calibration enlarges that side.
- Calibration cost: 5×10⁶ network inferences (< 1 s each) vs about 5×10¹¹ pyRing likelihood calls.
- Grids: 2500 points per 2-D map, 1000 simulations per point.
- Truncation: keep the 4σ HPD box; stop when no range shrinks by more than 2×. Roughly one extra round per doubling of SNR. LHO needed 2 rounds (107 and 125 epochs); LLO needed none.
- At 5× GW150914's SNR, calibration was "unnecessary": maps match exact coverage to a few %.
- Data: 0.1 s at 4096 Hz, 100–650 Hz band-pass, Tukey(1/70), rfft, whitening, then re/im concatenated into 112 bins.

## Implementation details

- Each training set has 10⁵ signals × 50 noise realisations. "Providing multiple noise realizations for each element sensibly increased the learning capacity"; more distinct signals did not help.
- 5×10⁶ simulations per stage, batch 100, Adam 5e-4, early stopping with patience 25.
- MAF: 5 layers, 2×120 hidden units, alternating order.
- One network per detector; posteriors combined by rejection sampling, or by KDE-weighted importance sampling (faster, checked equivalent).
- The window leakage is harmless because it is applied identically to simulations and data.
- Phases have uniform priors on [0, 2π] and are treated as plain intervals in the MAF; there is no circular handling.

## Caveats

low-SNR problem with a 6-D toy-ish model. Coverage is checked only on 2-D conditional slices, with the other parameters fixed at the MAP. The construction is frequentist confidence, not a Bayesian posterior. Truncation makes the network valid only locally.
