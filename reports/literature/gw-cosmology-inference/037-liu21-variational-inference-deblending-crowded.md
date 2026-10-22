# Variational Inference for Deblending Crowded Starfields

Liu et al., 2021, JMLR 2024; arXiv v3 Aug 2023 · [arXiv:2102.02409](https://arxiv.org/abs/2102.02409)

**Evidence: B.** Peer-reviewed (Journal of machine learning research); 12 citations in 5.7 yr
(Semantic Scholar, 2026-10-19).

## What they do

- **Tiling.** Split the image into R×R tiles, with R chosen so that P(3 or more stars in a tile) is small.
- **Per-tile network.** A CNN/ResNet sees each tile padded by neighbouring pixels. It outputs:
  - a categorical over Ñ ∈ {0..N_max};
  - for each possible count n, a "triangular array" of n sets of logit-normal location and log-normal flux parameters.
- **Transdimensional handling.** The posterior is a mixture over the count. Given the count n, use row n of the array. This handles the transdimensional space without RJ-MCMC.
- **Training.** Simulate full images and catalogues from the prior. The loss is cross-entropy on the per-tile count plus Gaussian NLL on the matched sources.
- **Permutations.** Within a tile, sources are a set, so q sums over all n! permutations (at most 3! = 6).

## What they find

- **M2 SDSS 100×100 crop (1114 HST stars):**
  | Method  | F1   | TPR / PPV   |
  |---------|------|-------------|
  | StarNet | 0.50 | 0.53 / 0.48 |
  | PCAT    | 0.44 | 0.55 / 0.37 |
  | DAOPHOT | 0.31 | 0.20 / 0.65 |
  - Over 10 refits, StarNet's F1 ranges 0.50–0.55.
  - Estimated N: StarNet mean 1462 (90% interval 1430–1497), PCAT 1672 (1664–1680). Neither covers the truth of 1114.
- **Speed:**
  - Training takes 1 h on one RTX 2080 Ti: 400 epochs × 200 images of 100×100, with Adam.
  - Inference on a 1000×1000 or 4000×2000 image takes under 1 s. PCAT needs 30 min per 100×100.
- **Prior robustness (App D):** F1 stays at 0.48–0.51 for μ ∈ [0.08, 0.20] and α ∈ [0.25, 1.0].
- **Two-star deblending (App C.1):** near-perfect at separation 1.5 px, which is below half the PSF FWHM (4.2 px). The MAP finds 2 stars at δ ≥ 1.2 px.
- **Forward KL vs ELBO (Sec 5):**
  - ELBO with REINFORCE does not converge.
  - Reparameterised ELBO, made possible by summing out N, gets stuck: two stars collapse onto one location, where the gradient vanishes.
  - Forward KL always recovers all 4 stars.

## Implementation details

- **Choosing R and padding:**
  - Pick the tile size so each tile holds few sources (N_max = 3).
  - Pad the input by about the PSF width: 3 px for SDSS with R = 2, 5 px for DECam with R = 10.
  - Less padding leads to more over-counting at tile edges (App C.3).
- **Tile-interior mask:** append a one-hot "inside tile" channel to the input so the network knows which sources it is responsible for.
- **Fixed-shape output:** one parameter set per hypothesised count (the triangular array). Output size grows as N_max², which is fine for small N_max.
- **Forward KL vs reverse KL:** prefer forward KL (NPE-style) for multi-source location problems. Its loss stays quadratic in the location error, so there are no flat ELBO plateaus where two sources sit on one peak.
- **Prior-sensitivity check:** refit warm-started from the base network and confirm the metric spread matches refit-to-refit noise.
- **Count-coverage test:** compute coverage on synthetic images, and also on synthetic images with sources kept away from tile edges, to isolate the effect of the factorisation.

## Caveats

- **Edge over-counting.** Spatial independence across tiles over-counts sources near tile edges: P(N = 1) tends to slightly above 1/2 at the boundary. On a synthetic 100×100 image the true N = 1195 lies below the 0.01 percentile of q(N).
  - Coverage is close to nominal only when sources are kept at least 0.1 px from the edges (App C.2).
  - So the tile-factorised posterior on the *total* count is badly overconfident and biased.
- **Simple family.** The family is mean-field within a tile (diagonal Gaussians), so there are no source–source correlations.
- **Variable N_max is a hard cap.**
- **Background errors cause misspecification on real data.**
- **Crowded fields fail.** In the M2 core (15k stars per 100×100), F1 = 0.13.
