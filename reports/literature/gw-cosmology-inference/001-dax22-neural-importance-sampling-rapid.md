# Neural Importance Sampling for Rapid and Reliable Gravitational-Wave Inference

Dax, Green et al., 2022, PRL 130, 171403 (2023) · [arXiv:2210.05686](https://arxiv.org/abs/2210.05686)

**Evidence: A.** Peer-reviewed (Physical Review Letters); 102 citations in 4.0 yr (Semantic
Scholar, 2026-10-19).

## What they do

Use the NPE flow q(θ|d) (DINGO, trained with forward KL, hence mass-covering) as an importance-sampling proposal: w_i = p(d|θ_i)p(θ_i)/q(θ_i|d), n_eff = (Σw)²/Σw², efficiency ε = n_eff/n, evidence p(d) = mean(w). σ_{log p(d)} = √((1−ε)/(nε)); the log-evidence bias is −(1−ε)/(2nε), negligible. Because DINGO cannot learn φc, the proposal is factorised q(θ|d) = p(φc|d, θ̃) q(θ̃|d): the flow models the 14 other parameters and φc is drawn from its *exact* likelihood conditional evaluated on a grid. This conditional is part of the proposal density, so IS stays exact.

## What they find

- GW150914 (XPHM): mean/max 1-D JSD vs LALInference 2.2/7.2 ×10⁻³ nat (DINGO) → 0.5/1.4 (DINGO-IS), against Bilby's 1.8/4.0. GW151012: 9.0/53.4 (max on Mc) → 0.7/2.2. JSD ≤ 2×10⁻³ nat counts as indistinguishable. ε = 28.8% and 12.5% for these two events.
- Evidence: DINGO-IS log p(d) ±0.01 vs Bilby-dynesty ±0.10 (10× more precise). Bilby is ~1σ higher on both events. Bilby-IS on 10⁶ Bilby samples gives ε = 8.3% and an evidence consistent with DINGO-IS.
- 42 GWTC-2/3 BBH: per-event ε ranges from 0.0% to 32.6%. Simpler IMRPhenomPv2 with a smaller prior gives median ε = 36.8% on 37 events.
- Cost: 10⁵ proposal samples per event, 1 A100 + 64 CPUs, <1 h (XPHM) or ~10 h (SEOBNRv4PHM), dominated by waveform generation. Plain DINGO samples take ~20 s.
- OOD flagging: a blip glitch in pure noise gives ε ≈ 0.001%; an adversarial input gives ε ≈ 0.006%.
- Plain NPE without GNPE on GW150914 gives ε = 1.48%, versus 28.8% with GNPE. The symmetry trick is worth ~20× in efficiency.

## Implementation details

- **Synthetic phase.** φc is drawn from p(φc|d, θ̃) on a 5001-point grid. It is cheap because, in the L0 frame, φc enters only through the spin-weighted harmonics (e^{−imφc}). The per-m combinations h̃_{+,m}, h̃_{×,m} are cached, so a whole grid costs about one waveform evaluation. Unlike analytic phase marginalisation, this is exact with higher modes. A subtlety: φc also enters the spin-frame conversion. It is fixed there, and θ_JN, φ_JL are re-defined and corrected in post-processing.
- **Density recovery for GNPE.** GNPE (Gibbs over blurred time proxies, ~30 iterations, up to 500; 500 iterations for 5×10⁴ samples take ~6 min on an A100) has no tractable density. They therefore sample 4×10⁵ GNPE draws and fit an unconditional NSF q(t̂_I) to the 2-D proxies (5 flow steps, hidden 256, 4 blocks, 8 bins, batch 4096, 20 epochs, lr 2e-3, 7 min on an A100). IS then runs in the joint (θ, t̂) space with target p(d|θ)p(θ)Πκ(t̂_I − t_I). Fitting a flow to the converged θ samples instead was "less sample efficient and more costly".
- **Training changes vs DINGO-20.** Separate networks for distance priors [0.1, 3], [0.1, 6] and [0.1, 12] Gpc improved O3 performance. Data: [20, 1024] Hz, Δf = 0.125 Hz. The first layer is seeded with PCA (SVD) components of clean waveforms. Mass prior [10, 120] M⊙, spins [0, 0.99].
- **Diagnostics.** Plot log q vs log(L·π): points should lie on the line offset by log p(d), and forward-KL errors sit *below* the line (proposal too broad, which is benign). Plot log p(d) vs n; jumps mean light tails and the result is untrustworthy. If not mass covering, ε → 0 as n grows. Batched self-normalised IS (normalise in batches of k < n) caps the maximum weight but introduces bias.
- Robustness: noisy training data and the PCA-seeded first layer made DINGO hard to attack adversarially.

## Caveats

BBH only (short signals, SNR ~10–25). The synthetic-phase trick needs the waveform's φ-dependence to factor analytically. IS efficiency measures the joint 15-D proposal: low ε does not mean bad marginals. Not a flow-matching model; the density comes from a discrete flow (log q is cheap).
