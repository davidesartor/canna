# Real-time gravitational-wave inference for binary neutron stars using machine learning

Dax, Green et al., 2024, arXiv; published Nature 2025 · [arXiv:2407.09602](https://arxiv.org/abs/2407.09602)

**Evidence: B.** Peer-reviewed (Nature); 60 citations in 2.3 yr (Semantic Scholar, 2026-10-19).

## What they do

BNS signals are too long (up to ~10⁷ input dims) for NPE. They split the problem into narrow regions that are easy to compress.
- **Prior conditioning.** Train with hierarchical draws θ ~ p_ρ(θ), ρ ~ p̂(ρ), and condition the network on ρ. Here ρ = M̃, the centre of a narrow chirp-mass prior U(M̃ ± ΔM).
- Within each local prior, the data are heterodyned by e^{iφ_0PN(f, M̃)} and then multibanded. The network predicts M − M̃.
- At inference, choose M̃ (from a search pipeline or the self-sweep) and sample. Then importance-sample with the exact likelihood.

## What they find

- Speed: 1 s for 10⁴ effective samples on an H100. For 5×10⁴ samples, network sampling takes 0.37 s and the JAX likelihood 0.19 s; the JIT takes 18 s, done ahead of time.
- GW170817: ε = 10.8% on the full signal and 78.9% 10 s before merger. GW190425: ε = 51.3%. LVK took 0.5–2 h with ε of 0.1%.
- Sky localisation: the 90% area is ~30% smaller than Bayestar's (median).
- Exact native-domain likelihood vs the multibanded one: for GW170817, ε = 11.0% for 5×10⁴ samples in 13 s, with JSD < 5×10⁻⁴ nat for every parameter. Multibanding is effectively lossless.
- Multibanded vs decimated signals agree to mismatch ≤ ~10⁻⁷.
- **SNR trend:** efficiency falls with SNR. CE pre-merger runs reach SNR ~10³ fifteen minutes before merger, and ε drops sharply there. Restricting the distance prior to keep SNR down restores ε. Low-SNR and early pre-merger runs (broad posteriors) score highest.
- PSD conditioning on an O2 PSD distribution lowers the mean ε from 57% to 25%.
- EOS evidence: a marginal-network histogram proposal plus a conditional network with IS gives ~50% efficiency, σ_log p ~0.01, in 1–3 s.

## Implementation details

- **Local prior via a learned offset.** The network predicts δM = M − M̃ given (d_M̃, M̃). The output is offset-centred, so the flow always sees a narrow, standardised target.
- **Self-sweep for M̃.** Run every local prior in parallel at ~10 samples each, a few thousand samples in total (<1 s). Take M̃ from the max-likelihood sample. The result does not depend on M̃ as long as the posterior fits inside [M̃ ± ΔM]. A joint sweep over (M̃, t_c) can even act as a trigger.
- **Building the multiband grid.**
  - Band i has Δf_i = 2^i Δf₀ and averages 2^i native bins.
  - Pick the band edges empirically: simulate 10³ heterodyned signals and require ≥32 bins per signal period.
  - Simulate noise directly in the compressed domain: σ_i = σ/√(2^i). Whiten with the averaged 1/√S.
  - Real data are whitened in the native domain and then decimated, to match the training pipeline.
- **Frequency masking.**
  - f_min = f_0PN(−T, M̃) + buffer, with a 1 Hz buffer for LVK and 0.5 Hz for XG.
  - f_max(t, M) comes from a cached lookup table, chosen so that truncation mismatch stays ≤10⁻³. During training f_max is drawn at random (uniform over MFD bins).
  - Bins outside [f_min, f_max] are zeroed. One network covers every pre-merger time.
- **Two networks.** Separate pre-merger (f ≤ 200 Hz) and full-signal networks perform better than a single one.
- **Parameter conditioning.** Condition on a parameter subset (e.g. sky position from an EM counterpart) to get p(θ∖{α, δ}|d, α, δ). Conditioning on (m, λ) gives conditional evidences for EOS work.
- **Training set.** 3×10⁷ samples for 200 epochs (LVK); 6×10⁷ for 100 epochs (CE). Architecture as DINGO with a slightly larger embedding. All priors are uniform except chirp mass, which is uniform in component masses. The distance prior is reweighted to uniform-in-comoving-volume afterwards.
- Training the CE network on a narrower chirp-mass prior allows coarser multibanding.

## Caveats

- The heterodyne assumes only the (2,2) mode, which is fine for BNS. Higher modes would need multiple heterodynes.
- The ±0.005 M⊙ width only works because BNS chirp masses are measured extremely well.
- SNR ~10³ is still unsolved: their wording is that "improving Dingo-BNS for such high SNR events will likely require improved density estimators".
- Assumes stationary Gaussian noise. No calibration marginalisation.
