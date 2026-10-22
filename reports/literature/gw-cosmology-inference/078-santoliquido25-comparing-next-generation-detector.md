# Comparing next-generation detector configurations for high-redshift gravitational wave sources with neural posterior estimation

Santoliquido et al., 2025, arXiv v2 Apr 2026 · [arXiv:2512.20699](https://arxiv.org/abs/2512.20699)

**Evidence: B.** Peer-reviewed (Astronomy & Astrophysics); 4 citations in 0.8 yr (Semantic
Scholar, 2026-10-19).

## What they do

Dingo-IS trained separately for 7 XG networks (ET Δ, 2L aligned/misaligned, plus LIGO A#, plus CE). Sources are massive high-z BBH (M_d > 100 M☉). Networks are ranked by information gain (prior→posterior KL), sky area, number of sky modes, and comoving volume.

## What they find

- Sample efficiency above 1% (≥10³ effective samples out of 10⁵) for 85–96% of the 1000 prior injections, at median SNR 40–70.
- **Ablation:** enlarging the embedding from 128 to 256 features (plus extra hidden layers, 4×10⁸ parameters) raised Δ's fraction from 85% (2504.21087) to 96%. The authors call the embedding network "crucial".
- Efficiency "drops sharply" around SNR ~1000. Events with M_d ≲ 150 and D_L ≲ 6 Gpc (high SNR) are almost all excluded, because "their posteriors are much narrower than the prior".
- Dingo-IS takes minutes vs Bilby 22 h (Δ) and 50 h (2L MisA). All JSDs are below threshold.
- Sky modes: 80% of Δ events have 8 modes, vs ~20% for 2L MisA. D_L is multimodal about 20% of the time with 2L vs about 2% with Δ.
- Raw-NPE P–P per-parameter KS p-values (Table 4) are mostly fine; ϕ_c has p = 0.04 for 2L MisA + CE. ϕ_c is again handled by Dingo's analytic phase step.

## Implementation details

- Information gain via Monte Carlo KL as a single scalar figure of merit.
- Mode counting via HEALPix flood fill (ligo-skymap).
- Proposed future fix for the high-SNR efficiency drop: neural population priors (Wouters+25), i.e. train on a prior closer to the observed population.

## Caveats

results are restricted to short, massive BBH; injections are selected on efficiency > 1% in all configurations (78% kept), which biases toward lower SNR.
