# SE(3) diffusion model with application to protein backbone generation

FrameDiff; Yim et al., 2023, ICML 2023 · [arXiv:2302.02277](https://arxiv.org/abs/2302.02277)

**Evidence: A.** Peer-reviewed (International Conference on Machine Learning); 373 citations in
3.7 yr (Semantic Scholar, 2026-10-19).

## What they do

Independent processes on SE(3)^N: Brownian motion on SO(3) (IGSO(3) kernel, series truncated at 2000 terms, inverse-CDF sampling) and an OU process on centred R³. The rotation DSM weight is chosen so a trivial prediction has expected loss 1. Auxiliary backbone and pairwise-distance losses (w=0.25) apply only for t < T/4. Sampling is a geodesic random walk with noise scale ζ and early truncation.

## What they find

- Designability scTM>0.5 by noise scale: ζ=1.0 → 49%, ζ=0.5 → 74%, ζ=0.1 → 75%. Lower noise improves designability and costs diversity (0.75 → 0.53).
- Ablation (ζ=1, 500 steps; columns self-cond, L2D, Lbb, Ldsm, LF): all on 49%; partial removals 39 / 42 / 22 / 16 / 0%.
- **Swapping the DSM rotation loss for the Frobenius loss L_F of prior work is among the ablations.** The column-to-row mapping was garbled in extraction, so it cannot be stated reliably which row is 0%.

## Implementation details

- Backbones destabilise in the last steps, so sampling truncates early.
- From memory, not verified in fetched text: logarithmic rotation schedule σ(t) = log(t e^{σmax} + (1−t)e^{σmin}) with σ∈[0.1,1.5], translation β∈[0.1,20], ε=0.01.

## Caveats

Proteins only. The ablation mapping is unclear.
