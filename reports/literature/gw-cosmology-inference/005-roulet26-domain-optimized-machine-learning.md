# A domain-optimized machine-learning tool for gravitational wave inference

Roulet, Crisostomi et al., 2026, arXiv; code "labrador" · [arXiv:2604.08897](https://arxiv.org/abs/2604.08897)

**Evidence: B.** Peer-reviewed (Physical Review D); 3 citations in 0.5 yr (Semantic Scholar,
2026-10-19).

## What they do

"Build a normalizing flow by hand" so the residual learning problem is close to a standard normal, invariant to the source.
- **Data side:**
  - Fit a ~10-parameter phenomenological reference waveform by maximising the likelihood: per-detector amplitude, phase and time, plus two PN phase principal components.
  - Heterodyne the data by that fit, which standardises the pose for *all* major parameters at once.
  - Compress with relative-binning weights (~100 frequencies), then whiten and keep ~50 SVD components.
  - The network's context is the reference parameters p* plus the SVD coefficients.
- **Parameter side:** θ → θ_T (analytic coordinates that remove degeneracies) → θ_F (fold 16 discrete symmetries; restore the label ℓ with a classifier) → θ_R (probit for bounded parameters, affine decorrelation from the data, then a small NN predicting μ(d) and the Cholesky of Σ(d), to whiten).
- A small RQ-NSF then learns p(θ_R|d).

## What they find

- **Cost:**
  - 10⁷ simulations take 100 CPU-core-days, dominated by the 0.9 s reference fit per simulation.
  - NSF training takes 10 h on a V100; the rescaler 1 GPU-h. The rest is under 1 CPU-h.
  - Inference runs >5000 samples/s on one CPU core; IS runs at 800 weighted samples/s.
  - Some overfitting is visible, and it **gets worse with larger networks**.
- **Calibration:** P–P plots are uniform, with KS p-values computed on weighted ECDFs and n_eff.
- **Efficiency:** median 1%. A few percent of events fall below 10⁻⁴ and a few percent exceed 10%. Reaching n_eff ≈ 1000 takes a median of 2 min on one core.
- **Real data:** O4a events agree with nested sampling after IS. Some events rail against the Mc = 50 training boundary.
- **Counterweight reweighting:** n_eff/N rises from 0.011 (naive reweighting) to 0.42, ~40×.
- **Comparison table:** DINGO has 1.3×10⁸ parameters and trains 16–18 days on a V100; DINGO-IS reaches 10%, DINGO-BNS 17%, Dingo-LISA (3.6×10⁸ parameters, 10 d on an A100) 22%, AMPLFI 6×10⁶ parameters. labrador trades efficiency for a ~600× smaller model.

## Implementation details

- **Fit the reference in orthonormal coordinates.** Use QR/SVD of PN phase profiles weighted by the SNR² per bin, W² ∝ Ā²/S·Δf. In these coordinates the log-likelihood Hessian is ∝ identity, an isotropic Gaussian with σ_c = 1/SNR. Amplitude and phase per detector are maximised analytically, the rest by differential evolution.
  - Training-set fits start from the injected phase. The maximisation erases that information, so there is no leakage.
- **Heterodyning = GNPE pose standardisation for every parameter of the reference.** It is exact for transformations that are also symmetries of Gaussian noise, i.e. phase and time shifts. Nothing is lost, because p* is kept as context.
- **No windowing on 32 s segments.** The noise is periodic and signals wrap around, which the heterodyne undoes. This is equivalent to arbitrarily long data.
- **SVD hygiene.** Centre (heterodyned signals have nonzero mean) and whiten by the empirical std of the residual noise per feature. Choose α_max by Wiener-filter signal power (99.9%), not raw variance.
- **Feature design for p*.**
  - Use relative quantities: t_H − t_L, cos and sin of (φ_H − φ_L), network amplitude A = √(A_H² + A_L²), and ratios A_k/A.
  - Drop the overall time and phase, which makes the model exactly equivariant to them.
- **Coordinates.**
  - Extrinsic (Roulet et al. 2022): detector-network sky angles plus arrival time, phase and inverse amplitude ("chirp distance") at the reference detector. Phase and time are taken *relative to the reference fit*.
  - Intrinsic, low mass: Fisher-PCA (μ₁, μ₂), plus q and C_2z = (χ_2z − min)/(max − min). This gives a uniform conditional posterior and no irregular boundaries.
  - Intrinsic, high mass: Mc, log q, χ_eff, C_2z.
- **Folding:** e.g. (cos ι)_F = −|cos ι|. Unfold by sampling ℓ ~ p(ℓ|θ_R, d) from an XGBoost softmax classifier.
- **Rescaler.**
  - **Probit beats logit** for bounded parameters, because a uniform posterior maps to N(0, 1).
  - Periodic parameters are first centred on a predicted circular mean, then probit-mapped.
  - Then an affine least-squares decorrelation from the data.
  - Then a 4×32 SiLU MLP predicting μ(d) and Σ^{-1/2}(d) (log-diagonal plus lower triangle), trained with a Gaussian cross-entropy.
- **Simulation proposal ≠ inference prior.**
  - Simulate uniform in inverse SNR (D̂ ~ U(D̂₅₀, D̂₈)), log-uniform in Mc and q. Keep simulations with reference SNR in [8, 50].
  - Train with weights w_i = e^{−λ(d_i)}·p(θ_i)/p̃(θ_i), λ(d) = α₁μ(d) + α₂σ(d). Here μ and σ are XGBoost regressors for the mean and std of the log prior ratio given d, and α₁, α₂ are chosen to maximise the n_eff of the training set.
  - This is valid because any positive data measure ν(d) leaves the minimiser of the cross-entropy unchanged. The flow directly outputs posteriors under the physical prior.
- **Quasi-random training set.** Draw on a scrambled Halton sequence by inverse-CDF and keep batches *contiguous* (no reshuffling) to preserve low discrepancy. This reduces overfitting relative to an iid set of the same size. The validation set is iid.
- **PSD mismatch at inference.** Whiten with a smoothly modified PSD S̃ = r(f)·S, where the spline r restores the training PSD at coarse resolution and keeps the sharp lines. This removes the linear bias.

## Caveats

- Aligned spin, (2,2) mode only, fixed PSD.
- A median ε of 1% is modest. Accuracy is traded for training cost.
- The heterodyne fit must be run for *every training simulation* (expensive, and rules out on-the-fly augmentation).
- Folding needs known discrete symmetries.
- The counterweight trick changes ν(d), which is fine for the minimiser but changes which data the finite-capacity network prioritises.
