# Opening the Black Box: What Neural Networks Learn from Pulsar Timing Array Data

Alvey, Mitridate, Pieroni, Romano, 2026 · [arXiv:2610.12308](https://arxiv.org/abs/2610.12308)

**Evidence: C.** Not peer-reviewed; 0 citations (OpenAlex, 2026-10-19). Its claims are the
authors' alone: a hypothesis, not a result.

## What they do

A 2-D ("Flatland") PTA toy model where the exact Bayes factor for GWB anisotropy (monopole + quadrupole P2^r, P2^i) is computable. A classifier trained with BCE converges to z* = ln B10 (the pre-sigmoid logit), so a network is "the projection of the log Bayes factor onto its function class". Compare: exact Bayes factor, sky-map frequentist statistic, a linear classifier (quadratic in residuals when fed correlations ρ_ab), and a 1-hidden-layer MLP (width 50).

## What they find

50 pulsars, 10 frequencies (20 samples), input = 1,275 pair correlations; 5×10^4 sims per hypothesis, batch 10^4; linear net by L-BFGS, MLP by AdamW (wd 1e-2, lr 1e-3), early stopping 10/20 epochs. Simple hypotheses: both classifiers match the Bayes-factor ROC; linear weights reproduce the analytic Neyman–Pearson cos(φa+φb+π/4) profile up to normalisation. Composite hypotheses (anisotropy drawn from a zero-mean disk prior): linear classifier = chance (AUC 0.5), sky-map statistic in between, MLP tracks the Bayes factor ROC; the MLP logit tracks ln B10 up to ln B ≈ 2 and then becomes *under-confident* (attributed to the increasingly peaked likelihood needing more data/capacity).

## Implementation details

App. A proof: at w=0 the BCE gradient is ∝ (⟨x⟩_H1 − ⟨x⟩_H0); if the prior-averaged mean of the input features is identical under both hypotheses (zero-mean residuals; zero-mean prior on the anisotropy phase), a linear model learns exactly nothing (weights → 0 as n grows). Information that lives only in second-order (or prior-symmetric) structure needs explicit nonlinear/multiplicative features. Composite case: simulations resampled every epoch.

## Caveats

Toy model, classification (detection), not posterior estimation; small networks; no flows.
