# Conditional Localization Test (CoLT) for Neural Posterior Estimators

Chen, Bansal & Scott, 2025, NeurIPS 2025 · [arXiv:2507.17030](https://arxiv.org/abs/2507.17030)

**Evidence: B.** Peer-reviewed (Neural Information Processing Systems 38, pp. 164309–164347); 0 citations
on its arXiv record, 1.3 yr old (Semantic Scholar, 2026-10-22; the NeurIPS record may count apart).

## What they do

CoLT tests whether q(θ|x) equals p(θ|x), using simulated pairs (θ*, x) ~ p(θ, x) and as many samples from q(·|x) as needed. It needs no density of q.

- **Ball-probability rank.** Pick a centre θ_l(x) and a metric d_ϕ. For a true pair (θ*_i, x_i), draw M samples θ̃_ij ~ q(·|x_i) and compute

  U_i = (1/M) Σ_j 1[d_ϕ(θ_l(x_i), θ̃_ij) ≤ d_ϕ(θ_l(x_i), θ*_i)].

  U_i is the fraction of q's mass inside the ball around θ_l(x_i) that just reaches θ*_i. If q = p, U_i is uniform for any centre and metric.
- **Learned localization.** θ_l(x) is a neural network in x. The embedding ϕ(θ) is a second network, with d_ϕ the Euclidean distance between embeddings. Both are trained to make the U_i as non-uniform as possible.
- **Theory.** Thm. 1: if the ball probabilities match for all centres and radii, then q = p (under a doubling condition on the metric). Thm. 2: maximising the deviation of U from uniformity over centres, embeddings and levels equals an integral probability metric built from indicator balls, which they call ACLD.
- **Test.** After training, compute U_i on fresh pairs and run a one-sample Kolmogorov-Smirnov test against Uniform(0, 1). The KS statistic is also a distance estimate.
- **Variants.** "CoLT ID" fixes ϕ to the identity, so the balls are Euclidean. "CoLT Full" learns ϕ.

## What they find

- **Toy tree** (a 2-D branching mixture, perturbation α). At α = 1.5, CoLT has power 0.877 and C2ST 0.000.
- **Gaussian benchmark** (x ~ N(1_m, I_m), p(θ|x) = N(W₁x, |W₂ᵀx| Σ) with Toeplitz Σ, ρ = 0.9). Perturbations: mean shift, covariance scaling (1 + α)Σ, anisotropic covariance, heavier tails, extra modes, mode collapse. Both CoLT variants match or beat C2ST and consistently beat SBC and TARP. CoLT ID does well on covariance scaling and extra modes. Power curves are in figures only; I did not extract numbers for covariance scaling.
- **Curved (latent-Gaussian manifold) benchmark.** CoLT Full has a clear advantage, especially for tail changes.
- **Blind prior**, q(θ|x) = p(θ), power at 5% (mean ± stderr over 3 seeds):

  | Method | x = 3, θ = 3 | x = 10, θ = 10 | x = 50, θ = 10 | x = 100, θ = 100 |
  |---|---|---|---|---|
  | CoLT ID | 1.000 | 1.000 | 1.000 | 1.000 |
  | CoLT Full | 0.975 ± 0.014 | 0.778 ± 0.222 | 0.693 ± 0.307 | 0.452 ± 0.260 |
  | C2ST | 1.000 | 1.000 | 0.847 ± 0.038 | 0.122 ± 0.007 |
  | SBC | 0.052 | 0.028 | 0.048 | 0.040 |
  | TARP | 0.053 | 0.047 | 0.035 | 0.068 |

  Type-I error in the same setting is 0.020–0.077 for all methods (C2ST 0.077 at the smallest size).
- **Diffusion posterior.** Treating the 20th reverse-diffusion step as truth, the CoLT statistic rises monotonically as the sample is taken from earlier steps (15 to 19).

## Implementation details

- **Networks.** ϕ, θ_l and the C2ST baseline are all 3-layer MLPs with 256 hidden units. ϕ's output dimension is dim(θ).
- **Training loss.** Sinkhorn divergence between the empirical U_i and uniform, used only for training. The indicator is differentiated with a straight-through estimator. The test itself uses KS.
- **Budget.** N = 100 true pairs and K = 500 q-draws per x for training. Power is measured on 200 fresh batches with the same budget. α = 0.05, 3 seeds.
- **Optimiser settings.** Mostly 1000 epochs at learning rate 1e-5 (25 epochs for mean shift). The blind-prior case uses learning rate 1e-3 for CoLT.
- **Baselines.** C2ST uses one q-draw per x. SBC is per-dimension KS with Bonferroni. TARP uses random reference points.
- **Code.** github.com/TianyuCodings/CoLT. Benchmarks in github.com/TianyuCodings/NPTBench.

## Caveats

- The learned centre and metric add inductive bias. If they are under-parameterised or poorly trained, real discrepancies are missed (the authors' own limitation).
- The rank statistic degrades with few q-draws per x.
- The ACLD value is hard to interpret in high dimensions.
- All benchmarks are synthetic, with low-dimensional x (at most 100). Raw long observations are not tested.
- The training set (N = 100 pairs) is small. How the networks scale to large N or complex x is not shown.
- I read the arXiv v1 text ("under review"), not the camera-ready version.
