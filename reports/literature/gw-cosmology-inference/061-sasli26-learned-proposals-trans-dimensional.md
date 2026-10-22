# Learned proposals in trans-dimensional inference are optimal at equilibrium, not during assembly

Sasli, Karnesis, Karamanis, Katz, Kourtesis, Coughlin, Mandic, Stergioulas, 2026, arXiv v1 04 Aug 2026, physics.data-an · [arXiv:2608.12392](https://arxiv.org/abs/2608.12392)

**Evidence: C.** Not peer-reviewed; 0 citations in 0.2 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

In Eryn parallel-tempered RJMCMC (HyperWave package), an adaptive NSF trained online on the cold chain's pooled per-component samples is used as the **birth** proposal (with exact MH/Green correction, 0.15 defensive prior mixture). Structural argument: the optimal birth g*(u|θ) ∝ L(θ∪u)/L(θ) p(u) is residual-dependent while the fit is assembling (data-informed matched-filter births win), but at equilibrium it degenerates to the posterior single-component marginal, which is exactly what the state-independent flow learns.
- **Model:** Morlet–Gabor wavelets, per-component u = (t0, f0, Q, ρ, φ0) with ρ = per-wavelet SNR, prior p(ρ) ∝ ρ/[1+(ρ/2ρ*)²]; shared extrinsic Ω = (α, δ, ψ, ε). Flow = pocoMC "nsf3" preset on zuko: 3 MAF transforms, RQ splines with 8 bins, width-32 residual conditioners; phase on the circle; retrained every 100 iterations, warm-started, needs ≥ 256 samples. N_temp = 10, N_walk = 50.

## What they find

- Benchmark (IMRPhenomPv2 BBH ~36/29 M⊙ in H1/L1 Gaussian noise, network SNR 45.8, 10 seeds per arm, 72k-step cap, V100): pre-specified stopping rule (N > 50τ̂, ESS > 3000, two-half TV of p(D) < 0.02) met 6/10 with flow births (12.0–49.2k steps, median 29.7k) vs 1/10 for Fisher/data-informed births (61.4k); one-sided Fisher exact p = 0.029. τ̂ at cap 51.3–55.3 vs 61.9–78.2. RJ acceptance unchanged (~0.09–0.11). Overlap 0.954 at stop; 0.930 vs 0.931 at a matched 24k.
- Assembly phase: data-informed births reach D ≈ 4–5 at step 400 vs 2–3 for flow births. Crowded variant (K = 30 overlapping wavelets): data-informed births lead throughout.
- Ablations: in-model flow independence move alone lowers acceptance to ~0.10 and helps nothing (0/13); Q-marginal-only refit 2/10 (p = 0.5 vs baseline); multiple-try flow births (4 candidates) double RJ acceptance to 0.18–0.20; SNR-weighted deaths no help.
- Model order recovery: K = 5 wavelets at network SNR 13.4 → p(5) ≈ 0.33–0.37, p(6) ≈ 0.31, p(7) ≈ 0.18 (mild +1 tendency, attributed to amplitude prior mass at low ρ). Richardson–Green galaxy data: mode K = 6.
- Generality: EEG (SNR 25, match 0.883, D = 9, 110 min A100); Sangria MBHB (SNR 351, match 0.999, D = 17, 79 min); GW150914 (SNR 24.9, match 0.97, D ~ 18, 32 min, 90% area 142 deg²). Fixed-dim 11-param Sangria MBHB in Eryn: flow raises in-model acceptance 0.085 → 0.124 but no ESS gain (τ already a few steps). Heterodyned BBH vs bilby–dynesty: max JS 0.006, 11 vs 73 min (with full likelihood their sampler is slower, 169 min).

## Caveats

Stopping threshold below its own sample-size noise floor (TV ≈ 0.06 at 24k steps) made early comparisons a lottery; unseeded move-kernel RNG gave ±30% τ̂ scatter; a cluster kernel/driver update slowed mixing ~3× at bit-identical likelihoods. Proposed but untested: a switched or residual-conditioned flow birth.
no GB problem; benchmark is one injection; gains measured on a stopping rule, not on posterior accuracy.
