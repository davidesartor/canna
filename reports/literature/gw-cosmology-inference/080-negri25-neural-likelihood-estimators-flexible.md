# Neural likelihood estimators for flexible Gravitational wave data analysis

Negri & Samajdar, 2025, arXiv preprint · [arXiv:2509.17606](https://arxiv.org/abs/2509.17606)

**Evidence: B.** Peer-reviewed (Monthly notices of the Royal Astronomical Society); 2 citations
in 1.1 yr (Semantic Scholar, 2026-10-19).

## What they do

- Training points: annealed-KDE design. Weighted KDEs with w ∝ L^{1/T} on a T ladder from the prior to T = 1 supply ~1e5 points.
- Fit a 4×64 GELU ResNet (~15k parameters) to log L.
- Run eryn parallel-tempered ensemble MCMC on the surrogate.
- Check the IS ESS of true L / NN L over the posterior samples. Below threshold, add 2e4 new true-likelihood points (posterior plus tempered-chain samples) and retrain; up to 6 cycles.

## What they find

- 99 HLV injections, SNR 12–30, IMRPhenomD aligned spin: overall P–P p = 0.21; only ψ falls below p = 0.05 individually.
- Convergence: 2/99 converge after cycle 1 (median ESS 5.7%); half by cycle 2; 98/99 within 6 cycles. That is ≤2e5 true evaluations, ESS/N ≥ 50%.
- Median ESS gain per cycle: G2 = 6.9, later ~2×.
- Cycle 1 often hallucinates a spurious mode (q ≈ 0.1); one retrain on samples from that mode kills it.
- Speed: NLE 2 µs vs 2 ms true per evaluation.
- GW150914 (PhenomD): 1.8e5 true evaluations vs 1.3e7 for pocomc; 40 min vs 656 min CPU; 1-D JSD < 1.5 mnat for all parameters.
- log Z differs by ~1 (285.2 vs 286.1), attributed to tails. The secondary sky mode is visibly imperfect.
- Four waveforms are consistent; NRSur takes 109 min.

## Implementation details

- **Phase: φc analytically marginalised** (22-mode h ∝ e^{2iφc}); D_L marginalised from a lookup table. That leaves 9 dimensions.
- **Angles enter as (sin, cos); angles with π-periodicity also as (sin 2θ, cos 2θ)**, e.g. ψ, so the surrogate is exactly periodic.
- Sky as detector-frame zenith and azimuth (κ, ε) plus single-detector arrival time.
- Loss: MSE on exp(log L), i.e. linear likelihood space, so error concentrates at high-L points. Per-point weights v_i = KDE(θ_i)^{1/d} up-weight sparse regions. L1 weight penalty, λ = 1e-6.
- log L normalised by its median.
- Optimiser: Adamax, cosine warm restarts, adaptive gradient clipping.
- Overfitting shows up as spurious posterior modes. The acceptance test is the IS ESS against the true likelihood.
- Retraining adds tempered-chain samples to cover secondary modes.

## Caveats

- Only aligned spins with 22-only waveforms, so phase marginalisation is exact.
- Validated only up to SNR ~40 (degradation expected at high SNR).
- Multimodal extrinsics (sky, ψ) are the weak spot.
- Not amortised; per-event cost ~30–110 min CPU.
