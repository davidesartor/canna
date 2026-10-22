# On Embeddings for Numerical Features in Tabular Deep Learning

Gorishniy et al., 2022, NeurIPS 2022 · [arXiv:2203.05556](https://arxiv.org/abs/2203.05556)

**Evidence: A.** Peer-reviewed (Neural Information Processing Systems); 343 citations in 4.6 yr
(Semantic Scholar, 2026-10-19).

## What they do

embed each scalar feature separately before the backbone mixes features.
- **PLE with bins [b_{t-1}, b_t):** e_t = 0 if x < b_{t-1}; e_t = 1 if x ≥ b_t; otherwise e_t = (x - b_{t-1})/(b_t - b_{t-1}).
  - It is a "thermometer" code with a linear ramp inside the active bin.
  - It is lossless, and with T = 1 it reduces to the scalar itself.
  - The first and last bins extrapolate linearly (e_1 ≤ 0, e_T ≥ 1).
  - Linear(PLE(x)) = v_0 + Σ_t e_t v_t, i.e. one learned vector per bin, weighted by e_t.
- **Bin edges:**
  - quantile-based: b_t = Q_{t/T}(train), with zero-width bins dropped;
  - target-aware: the leaves of a single-feature decision tree fit to the target, in the style of C4.5.
- **Alternative, Periodic:** concat[sin 2πc_i x, cos 2πc_i x], with trainable c_i ~ N(0, σ). σ is the crucial hyperparameter.

## What they find

11 GBDT-friendly tabular datasets; 15 seeds per config.
- **PLE vs plain MLP:**
  - California Housing RMSE: MLP 0.495 → MLP-Q 0.464 → MLP-T 0.447 → MLP-T-LR 0.439.
  - Adult accuracy: 0.854 → 0.859 (Q) / 0.864 (T) / 0.868 (T-LR).
  - On some datasets PLE does not help (FB: MLP-Q 5.766 vs MLP 5.686; MI flat).
- **Ensembles (Table 6), average rank:** MLP 8.5 → MLP-Q-LR 5.1 / MLP-T-LR 5.1 → MLP-PLR 3.0, vs CatBoost 3.6. The best module on average is PLR (periodic → linear → ReLU), not PLE.
- **Ablation of the encoding shape (Table 8):** piecewise-linear vs binary (thermometer) vs one-blob.
  - The piecewise-linear version is best or tied on most datasets.
  - CH: binary drops 0.854 → 0.815.
  - GE: one-blob drops 0.653 → 0.613.
- **Preprocessing (Table 9):** PLE makes the model insensitive to the choice of preprocessing.
  - CA, plain MLP: none 1.118 vs quantile 0.495.
  - CA, MLP-Q: none 0.463 vs quantile 0.464.
- **Fourier features that mix features before the periodic map (Tancik-style, D.2)** are worse than the plain MLP. Per-feature embedding matters.

## Implementation details

- **Bin count:**
  - tuned per dataset, UniformInt[2, 256], shared across features;
  - trees tuned on leaves [2, 256], min items per leaf [1, 128], min gain LogU[1e-9, 0.01];
  - linear output dim [1, 128].
  - "Using too many bins can lead to overfitting" (the synthetic task of D.1 kept improving with more bins).
- **Attention models:** one linear layer after PLE, not shared across features, is "enough" and also supplies feature identity.
- **Not worth the cost:** heavy modules (Q-LRLR, PLRLR, AutoDis).
- **Training setup:** AdamW, no LR schedule, early stopping with patience 16.

## Caveats

- Tabular data only, with tens to hundreds of features and ≤1.2M rows. Small classification and regression models.
- No generative models, no continuous-time conditioning, no high-dynamic-range signals.
- Bins come from training-set quantiles or trees. The paper never tests fixed analytic grids (e.g. arcsinh-spaced), or bin counts above 256.
- Gains are dataset-dependent, and PLE loses on some datasets.
