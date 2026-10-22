# Complex-valued vs. real-valued neural networks for classification perspectives: an example on non-circular data

Barrachina, Ren, Morisseau, Vieillard & Ovarlez, 2021, ICASSP ·
[arXiv:2009.08340](https://arxiv.org/abs/2009.08340)

**Evidence: B.** Peer-reviewed (ICASSP 2021); 53 citations in ~6 yr (Semantic Scholar,
2026-10-19).

## What they do

A controlled synthetic test of whether a CVNN beats a real network (RVNN) when the class
information lives **only in the relation between real and imaginary parts**. Each example is
128 i.i.d. features drawn from a non-circular complex Gaussian `CN(0, σ², τ)`; the classes
differ only in the pseudo-variance `τ = σ_X² − σ_Y² + 2iσ_XY` (non-circularity), not in the
variance. Three binary datasets: A (equal variances, correlation ρ = ±0.3 between Re and Im),
B (swapped Re/Im variances 1/2 vs 2/1, ρ = 0), C (both). Dropping either Re or Im makes the
classes identical, so this is a pure Re–Im-relationship task. A parametric ML classifier
thresholding the estimated `τ` gives the accuracy ceiling.

## What they find

Median test accuracy over 30 Monte-Carlo trials (error = 1.57·IQR/√n), CVNN vs RVNN:

| Dataset | 1 hidden layer | 2 hidden layers |
|---|---|---|
| A (ρ = ±0.3) | 96.16 vs 94.48 | 97.83 vs 95.82 |
| B (variances) | 97.39 vs 96.65 | 98.89 vs 97.83 |
| C (both) | 99.67 vs 99.48 | 99.90 vs 99.87 |

- The CVNN wins on all three with non-overlapping error bars; the gap is largest (≈2 points)
  when the signal is only in the Re–Im correlation, and grows with |ρ| from 0.2 up to ≈0.7,
  then both saturate as |ρ| → 1.
- **Overfitting**: without dropout the 2-layer RVNN falls from 97.8 % to 90.15 %, the CVNN
  only to a median 93.48 % (with occasional bad outlier runs and higher variance). With
  dropout, all CVNN runs on A stayed above 97.30 %, all RVNN runs below 96.30 %.
- The RVNN learns faster in the first epochs; the CVNN overtakes later.
- A **polar RVNN** fed "the amplitude and phase" (§4.4: a raw angle, not its sin/cos)
  instead of (Re, Im) overfit badly even with more dropout and never beat the Cartesian
  RVNN. One paragraph, no numbers, and the input's 2π wrap is untreated.

## Implementation details

- MLPs: 1 hidden layer (64 complex units) or 2 (100, 40). The RVNN takes the 256 stacked
  (Re, Im) inputs and **doubles every hidden width** (128; 200, 80): a *feature-equivalent*
  match that gives the RVNN **more** real parameters than the CVNN (a complex dense layer
  `n→m` has `2nm` real params; the real one `2n→2m` has `4nm`). The CVNN won anyway.
- CVNN activation: Type-A (split) ReLU on Re and Im separately; softmax on the output
  magnitude. Dropout 0.5 on hidden layers.
- Plain SGD, lr 0.01, 300 epochs, batch 100; Glorot-uniform init with zero biases (the
  CVNN's halved fan-in/out gives it slightly larger init bounds). No precision discussion.

## Caveats

- Synthetic, tiny MLPs, binary classification, a hand-picked task where the signal is a
  second-order Re–Im statistic. The paper notes learning rate, feature size and dataset size
  were not explored.
- The matching is by feature count, not parameters; a parameter-matched RVNN (√2× narrower
  than the feature-matched one) was not tested, so the result reads as "the complex
  constraint regularises", not as a capacity effect.
- The same group's tutorial (Barrachina et al. 2023, arXiv:2302.08286, 27 citations, preprint)
  reports a ≈4-point accuracy gain for a CV-MLP over an RV-MLP with doubled widths on chirp
  (radar waveform) classification after a Hilbert transform, with less overfitting, but
  flags that the models were not equivalent and the RVNN may have been penalised by the
  `|·|`-softmax output.
