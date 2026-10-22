# Generalization in diffusion models arises from geometry-adaptive harmonic representations

Kadkhodaie et al., 2023, ICLR 2024 · [arXiv:2310.02557](https://arxiv.org/abs/2310.02557)

**Evidence: A.** Peer-reviewed (International Conference on Learning Representations); 213
citations in 3.0 yr (Semantic Scholar, 2026-10-19).

## What they do

- Two UNet/BF-CNN denoisers trained on disjoint face subsets converge to the same score once N is large: a memorisation → generalisation transition.
- The inductive bias is shrinkage in a geometry-adaptive harmonic basis (GAHB). These are Jacobian eigenvectors that oscillate along contours and in smooth regions. For the optimal denoiser, ∇f*(y) = Cov[x|y]/σ².

## What they find

- Faces at 80×80: N = 1 and 10 memorise (train PSNR has unit slope, test PSNR is flat). N = 1000 is transitional. By N = 10⁵, train and test error match at every noise level, and the S1 and S2 models produce nearly identical samples from the same noise.
- Geometric C^α images: the empirical PSNR slopes match the optimal α/(α+1) (bandlet theory).
- A 5-D manifold of disks (position, size, two intensities): suboptimal. Extra GAHB eigenvectors with non-negligible eigenvalues survive, so the PSNR slope stays below 1 at small σ. The suboptimality grows as σ shrinks relative to manifold curvature.
- Pixel-shuffled faces: substantially worse, because the bias toward local harmonics no longer matches the data.

## Implementation details

- Diagnostic: eigendecompose the denoiser Jacobian. Its trace and spectrum estimate the posterior covariance; ∇f* ∝ Cov[x|y].
- KL is bounded by the integrated MSE gap ∫(MSE − MSE*)σ⁻³dσ.

## Caveats

- Image denoisers only. There is no formal definition of the GAHB class.
- It says nothing about Gaussian scores; the result is about harmonic adaptive bases for images.
