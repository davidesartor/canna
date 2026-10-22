# Sampling-Based Accuracy Testing of Posterior Estimators for General Inference

Lemos et al., 2023, ICML 2023 · [arXiv:2302.03026](https://arxiv.org/abs/2302.03026)

**Evidence: B.** Peer-reviewed (International Conference on Machine Learning); 95 citations in
3.7 yr (Semantic Scholar, 2026-10-19).

## What they do

for each simulation, draw a reference point θ_r. Let f be the fraction of posterior samples closer to θ_r than the truth θ* is. Coverage at level 1 − α is the empirical CDF of f. This uses only samples, so it works for diffusion and flow-matching posteriors with no density.

## What they find

- Gaussian toy: TARP sits on the diagonal for the correct case up to D = 1000. It separates over- and under-confident cases. It detects a position-dependent bias, built so HPD cannot see it, that HPD misses.
- Uninformative estimator p̂ = p(θ) (1-D conjugate model, n_x = 50, σ_x = 0.1, 500 sims): HPD and prior-drawn TARP both look perfect. TARP with θ_r = x₀ + U(0,1) exposes it.
- Lensing (16×16 source = 256 params, 32×32 image, 500 sims × 1000 posterior samples): the exact score-based sampler is calibrated, and the approximate-likelihood sampler is visibly biased. Both use 300 Euler–Maruyama steps.

## Implementation details

- Normalise every parameter to [0, 1] before taking L2 distances. TARP is not invariant to the metric's scaling, so the choice of chart matters.
- Draw θ_r uniformly in the hypercube or from the prior. The results are robust to the θ_r distribution and to L1 vs L2, except in the prior-only failure mode.
- Make θ_r depend on x to make the test sufficient. For example, centre θ_r on a cheap estimate from the data plus noise.
- The curve shapes read differently from HPD curves. In TARP, an under-confident posterior pushes f toward 0.5 (an S-curve), while an over-confident one pushes f toward 0 or 1 (an inverted S). Do not read TARP curves with HPD intuition.
- Reference code: github.com/Ciela-Institute/tarp.

## Caveats

- A single scalar statistic over the whole parameter vector. It does not say which parameter is wrong, and the distance is dominated by whichever coordinates have the largest normalised spread.
- Euclidean distance on circles and spheres is ill-posed unless the chart is embedded. The proof assumes Euclidean U ⊂ ℝⁿ.
- Sufficiency is asymptotic: as α → 1 the balls must shrink, so detecting small local errors takes many sims and samples.
- Detection power is never quantified (no power-vs-N study).
