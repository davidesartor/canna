# Theoretical Insights for Diffusion Guidance: A Case Study for Gaussian Mixture Models

Wu, Chen, Li, Wang, Wei, 2024, arXiv; venue not stated · [arXiv:2403.01639](https://arxiv.org/abs/2403.01639)

**Evidence: B.** Peer-reviewed (International Conference on Machine Learning); 53 citations in
2.6 yr (Semantic Scholar, 2026-10-19).

## What they do

With exact scores, classifier guidance and CFG coincide for a GMM under the OU forward process. The paper proves that guidance strength η ≥ 0 (a) raises the posterior class confidence along the trajectory and (b) never increases the differential entropy of the output (DDIM, via Fokker–Planck).

## What they find

- Confidence: DDIM gives ≥ the unguided value at every t (Thm 3.3), with final confidence 1−O(η⁻¹ log η) (Thm 3.6). DDPM holds a.s. with a much slower rate, 1−O(η^{−e^{−T}}·polylog).
- Entropy reduction (Thm 4.2) needs only non-degenerate Σ and positive weights. It is unproven for DDPM (open problem).
- Discretised Euler: the confidence gain survives but becomes marginal for large steps. The entropy result needs small steps.
- Counterexample (Fig. 3): three collinear components with guidance toward the middle one **split the guided class into two side clusters**, and coarse steps (0.1 vs 0.04) bring the split on at lower η. Results are plots only, 10⁴ samples per η.

## Implementation details

Step size and guidance strength interact: coarse solvers plus guidance distort more.

## Caveats

Isotropic-covariance GMM with near-orthogonal centres (Assumption 3.1). The bounds may be loose. There are no realistic-data experiments.
