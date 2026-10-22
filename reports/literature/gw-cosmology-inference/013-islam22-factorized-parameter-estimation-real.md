# Factorized Parameter Estimation for Real-Time Gravitational Wave Inference

Islam et al., 2022, arXiv; cogwheel · [arXiv:2210.16278](https://arxiv.org/abs/2210.16278)

**Evidence: C.** Not peer-reviewed; 14 citations in 4.0 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

- For aligned-spin (2,±2) CBC, the likelihood depends on the 7 extrinsics through a few complex overlaps: Z (data vs template timeseries per detector) and z0 (template norm in each detector at φc = 0 and D0).
- Distance and phase are integrated analytically. With Y = (D0/D) e^{−2iφc}, the likelihood is Gaussian in Y.
- Sky, ψ and cos ι are integrated by Monte Carlo, with importance sampling over per-detector arrival times τ_k ∝ exp(½|Z_k(τ_k)|²).
- Nested sampling then runs only over 4 intrinsics (m1, m2, χ1, χ2). For each intrinsic sample, the extrinsics are reconstructed in post-processing: pick one weighted (α, δ, ψ, ι) MC tuple, then sample D by inverse CDF, then φc by inverse CDF.

## What they find

- Event-1 (GW150914-like, SNR 18.3 with 2 detectors): JS divergence vs full 11-D cogwheel is < 0.1 nat on every parameter (vs dynesty: 0.017–0.051 nat). Runtime 185 s (36 s max-likelihood + 149 s sampling) vs 4226 s for dynesty, ~20× faster.
- Per-call cost: 3.04 ms marginalised vs 0.49 ms full. It is still faster overall because the sampled space is 4-D.
- ~30 posterior samples/s; sampling time grows with SNR.
- BNS: 250 s, or 650 s with tidal parameters.
- GW170817 (SNR 33): 1124 s vs 9.7 h (~50× faster); JS < 0.075 nat, with φ at 0.032.
- MC settings: n_MC = 10 000 (ψ, cos ι) points, 2000 × 2000 sky grid, 4096 Hz arrival-time grid, PyMultiNest nlive = 512 (vs 2048 for the 11-D run).

## Implementation details

- **Phase conditional (Eq. 13–16):** P(φc) ∝ exp(Re(y z0·Z* e^{−2iφc})). Peak at φ_peak = arg(z0·Z)/2, width σ = (|z0·Z| y)^{−1/2}. Sample Δφ on a 1000-point grid within ±min(7σ, π/2), then add γπ with γ ∈ {0, 1} for the (2,2)-mode π-degeneracy.
- **Distance:** a grid of 1000 points over y ∈ [max(1e−5, y_peak − 7σ_y), y_peak + 7σ_y], with y_peak = |z0·Z|/‖z0‖² and σ_y = 1/‖z0‖², then inverse-CDF sampling.
- Relative binning is modified to give matched-filter time series over a ±100 ms window, by computing summary data for many arrival times.
- Maximum likelihood (relative-binning reference) is found first (36 s); search-pipeline summary data could be reused instead.

## Caveats

- Aligned spin, (2,2) mode only, ground-based detectors, SNR ≤ 33.
- Only 4 BBH injections plus GW170817; no P–P test.
