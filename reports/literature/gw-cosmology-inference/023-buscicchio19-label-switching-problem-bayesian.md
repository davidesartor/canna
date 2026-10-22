# The Label Switching Problem in Bayesian Analysis for Gravitational Wave Astronomy

Buscicchio et al., 2019; PRD 100, 084041 · [arXiv:1907.11631](https://arxiv.org/abs/1907.11631)

**Evidence: B.** Peer-reviewed (Physical Review D); 19 citations in 7.2 yr (Semantic Scholar,
2026-10-19).

## What they do

Impose the identifiability constraint x₁ ≤ x₂ ≤ … ≤ x_K on one per-component parameter via a *bijection* φ from the unit hypercube to the ordered hypertriangle with constant Jacobian 1/K!, so a black-box sampler sees a flat prior and a single mode: x'_i = 1 − ∏_{j≤i}(1−x_j)^{1/(K+1−j)} (corrected Eq. A13 of the MultiNest/PolyChord "sorted" prior). Sorting after proposing is *not* a fix (many-to-one); rejecting unordered points wastes a fraction 1−1/K!. For separable non-flat priors, first map through the prior CDF.

## What they find

(1) K = 1..4 bivariate Gaussian mixture for the 10 GWTC-1 BBH masses: evidences identical with and without the map (e.g. K=2: −78.37 vs −78.30; K=4: −84.58 vs −84.2, discrepancy attributed to CPNest under-estimating the multimodal Z), multimodality removed. (2) **6 overlapping LISA GBs**, 1 yr, SNR 10 each, frequencies spaced 2/yr apart around 1 mHz in a 12-bin band (heterodyned/downsampled), A/E channels, noise-free injection, 7 params each (log₁₀A, f, λ, sin β, cos ι, ψ, Φ) = 42-D: ordering by f via the map removes the 720 modes; CPNest recovers all six; neighbouring sources (4,5) show genuine posterior correlation from overlap.

## Implementation details

Five-line numpy implementation (cumprod of (1−x)^{1/(K−i)}). Constraint parameter chosen as the one that best separates components (frequency for GBs, mean log m₁ for mixtures).

## Caveats

Ordering removes the K! symmetry but not the genuine bimodality when two components are near-equal in the ordering parameter (posterior mass piles at the x_i = x_{i+1} boundary); number of sources assumed known; a single noise-free injection.
