# Transformers with Physics-Informed Encodings and Simulation-Based Inference for Robust Detection of Eccentric Binary Black Holes in Pulsar Timing Array Data

Dandapat et al., 2026 · [arXiv:2607.03904](https://arxiv.org/abs/2607.03904)

**Evidence: C.** Not peer-reviewed; 0 citations in 0.3 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

Two-stage NPE for an eccentric SMBHB in 10 pulsars (L = 400 residuals each). (1) Phase predictor: per-time-step linear mix of the 10 pulsars → conv stem (stride 4) → 4-layer Transformer (D=128) → per-token (cos φ, sin φ) upsampled to full rate, plus log-SNR regression. Loss = 8·L_vonMises(−cos of angle error) + 0.1·circular smoothness (wrapped Δφ²) + 0.05·spectral consistency (L2 between normalised |FFT| of e^{iφ̂} and e^{iφ}) + 0.05·MSE(log SNR). (2) Hierarchical Transformer conditioner: per-pulsar patch embedding (patch 20, d=128, 8 heads, 2 layers), added sinusoidal PE plus a "physics-informed phase encoding" = sinusoidal features of the predicted, token-averaged wrapped phase, each PE scaled by a *learnable scalar gate*; mean-pool per pulsar → 2-layer cross-pulsar attention → mean pool → context h → RealNVP coupling flow (primary) or CNF. For the high-D run, phase conditioning is masked so it only enters the scale/shift (or velocity component) of (log n0, e0, log M).

## What they find

Phase predictor: mean wrapped phase error 8.3°, median 5.3°; SNR regression R² 0.93 (lin) / 0.96 (log). Mean log posterior density at truth (standardised space; default 5×10^4 realisations, SNR ∈ [20,30]): CNF −0.715 → 0.650 with predicted phase; DNF −1.197 → 0.805. CNF better than DNF without phase, gap closes with phase; CNF ~3× slower. With 4×10^5 training realisations the no-phase and predicted-phase posteriors become nearly identical and both agree broadly with a Bayesian reference (qualitative, one corner plot). Marginal-coverage MAE 2.09% → 1.35% with phase; worst deviations at the 68% level.

## Implementation details

SNR fixed per realisation by *rescaling the noise σ* to hit a target SNR drawn U(20,30) (phase net: log-uniform 10–100) — the training distribution is SNR-controlled, not prior-controlled. z-score inputs and targets from training stats. AdamW lr 2e-4, wd 1e-4, grad-clip 1.0, batch 128, 120 epochs, best-val-NLL checkpoint; phase net cosine LR, dropout 0.1, 100 epochs. Learned scalar gates on each positional-encoding source let the net down-weight an unreliable channel.

## Caveats

Only 4 targets in most experiments; "Bayesian inference" comparison is a single visual corner plot; no SBC/TARP beyond marginal 1-D coverage; white noise; the gain from physics encodings largely vanishes with 8× more data (authors' own finding). Phase predictor needs the true phase trajectory as a supervised label — a two-stage pipeline with label leakage risk avoided only because it is frozen and trained separately.
