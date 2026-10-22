# Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains

Tancik et al., 2020, NeurIPS 2020 · [arXiv:2006.10739](https://arxiv.org/abs/2006.10739)

**Evidence: A.** Peer-reviewed (Neural Information Processing Systems); 4100 citations in 6.3
yr (Semantic Scholar, 2026-10-19).

## What they do

In the NTK regime the training residual decays along kernel eigenvector $i$ at rate $\eta\lambda_i$ ($Q^\top(\hat y_t-y)\approx -e^{-\eta\Lambda t}Q^\top y$); a coordinate MLP's NTK eigenvalues fall off fast with frequency, so high-frequency components effectively never converge. Mapping inputs $\gamma(v)=[a_j\cos 2\pi b_j^\top v, a_j\sin 2\pi b_j^\top v]_j$ makes the composed NTK stationary, $h_{NTK}(h_\gamma(v_1-v_2))$, with a bandwidth set by the $b_j$, turning the MLP into a tunable reconstruction filter.

## What they find

Table 1 (PSNR; IoU for 3-D shape), no mapping → basic $[\cos 2\pi v,\sin 2\pi v]$ → positional enc. → Gaussian RFF: natural images 19.32 → 21.71 → 24.95 → 25.57; text images 18.40 → 20.48 → 27.57 → 30.47; 3-D occupancy IoU 0.864 → 0.892 → 0.960 → 0.973; CT Shepp 16.75 → 23.31 → 26.89 → 28.33; MRI 26.14 → 28.58 → 32.23 → 34.51; NeRF 22.41 → 23.16 → 25.28 → 25.48. The scale (std) of the sampled frequencies, not the distribution family (Gaussian/uniform/log-uniform/Laplace), predicts test error: too small = underfit (slow high-frequency convergence), too large = aliasing/overfit (Fig. 4, 1-D 1/f^α signals; Fig. 9 same in 2-D). 16 random features ≈ dense 1024-feature basis. Even the single-frequency "basic" wrap gives shift invariance: a plain MLP fits a centred Gaussian bump but not shifted copies; one cos/sin pair fixes it (Fig. 14).

## Implementation details

256 frequencies, 4×256 ReLU MLP, Adam lr 1e-3 (1e-2 for no-mapping/basic), 1–2k iterations (NeRF 5e5). Best scales: σ_g = 10 (natural images), 14 (text), 4–5 (CT), 5 (MRI); positional-encoding σ_p 3–6. Deeper nets need fewer features (each feature's spectral support "spreads" with depth). *Learning* $b_j$ by Adam does not help: they barely move from initialisation (App. A.3), so the scale must be set by hand/validation.

## Caveats

low-dimensional (1–3-D) coordinate inputs only; the NTK argument is for wide nets / small lr; high-dimensional inputs are explicitly out of scope.
