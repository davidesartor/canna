# Dark Energy Survey Year 3 results: likelihood-free, simulation-based wCDM inference with neural compression of weak-lensing map statistics

Jeffrey, Whiteway, Gatti et al., 2024, MNRAS / arXiv v1 · [arXiv:2403.02314](https://arxiv.org/abs/2403.02314)

**Evidence: B.** Peer-reviewed (Monthly notices of the Royal Astronomical Society); 51
citations in 2.6 yr (Semantic Scholar, 2026-10-19).

## What they do

- A production SBI analysis of the DES Y3 weak-lensing maps for wCDM. Pipeline: summaries (pseudo-C_ℓ EE/BB, peak counts over 8 smoothing scales including cross-maps, or the map itself through a CNN), then MSE neural compression to one number per target parameter, then NLE with pyDELFI over (Ω_m, S8, w).
- Every other parameter, about 10³ nuisances in total (hyperrank n(z) realisations, IA, shear bias, h, n_s, Ω_b h², m_ν), is *implicitly marginalised*: it is drawn from its prior in each sim and left out of the conditioning. This uses the marginal-NLE argument of Jeffrey & Wandelt 2020.
- The simulation suite is Gower Street: 791 full-sky pkdgrav3 runs, 4 DES footprints each (3088 mocks), with 3 noise realisations each for training.

## What they find

| Summary | Ω_m | S8 | w |
|---|---|---|---|
| C_ℓ | 0.352 (+0.035/−0.053) | 0.807 ± 0.026 | < −0.661 |
| C_ℓ×Peaks | 0.340 (+0.030/−0.042) | 0.807 ± 0.023 | < −0.740 |
| C_ℓ×CNN | 0.283 (+0.020/−0.027) | 0.804 (+0.025/−0.017) | < −0.803 |
- C_ℓ×CNN improves the FoM over C_ℓ by 2.26× for (S8, Ω_m) and 2.48× for (Ω_DE, w).
- Validation:
  - Inference on the mean compressed data recovers the mean truth.
  - TARP coverage, with 50 leave-one-out NLE refits and 9×10³ samples each, matches nominal.
  - Injecting baryons and source-galaxy-bias variations (b = 0.5 and 1.5) into independent CosmoGridV1 sims shifts the posterior by less than 0.3σ.
  - NDE-ensemble agreement on the real data, run blind.
- There is no ablation of the individual tricks: no single-network versus 12-network ensemble numbers, and no MSE versus VMIM comparison.

## Implementation details

- **MSE compressor, one network per parameter.** MSE depends only on the per-parameter marginal posterior, so each parameter gets its own network.
- **Sigmoid output** on rescaled parameters, to keep summaries in a bounded domain.
- **12-member ensembles averaged.** The paper says this was "clearly superior to training once", without numbers. For the CNN, 4 networks per sky patch × 3 patches, weighted by sky fraction.
- **Data augmentation:**
  - Gaussian input noise with σ = 10⁻³ × the per-bin std.
  - Several shape-noise realisations per N-body mock.
  - The compressor is trained on *different noise realisations* from those used for NLE, to avoid overfitting leakage. The authors checked that reusing the early-stopping validation set for NLE does not shift the posterior.
- **Compressor training settings:**
  - MLP: 10 × 256 with LayerNorm and ReLU.
  - Adam, lr 10⁻⁴ with exponential decay.
  - At most 200 epochs, with early stopping on MSE over a held-out noise realisation.
  - Training took about 1 h per parameter on 12 A100s.
- **NLE** is an ensemble of 4 MAFs (3/5/6 MADEs, 2 × 40–50 hidden). MAFs were preferred to MDNs because they handle hard prior edges better (w = −1).
- **Active learning of simulation parameters:** (Ω_m, σ8) for later sims were drawn from the current posterior (computed blind, inside the code), with a hard exclusion zone around already-used points. Placement was Sobol/Halton quasi-random rather than i.i.d.
- **What failed:**
  - Adding an extra compressed A_IA summary tightened S8, which proves the S8 compression was not sufficient. But the posterior then became so tight that the NDE-ensemble test failed, because there were too few sims in that region. They dropped it.
  - Injecting PSF contamination at map level was swamped by star shot noise.

## Caveats

- MSE compression is explicitly non-optimal, as section 7.6 demonstrates.
- Flat-sky patches lose large-angle information; the C_ℓ concatenation recovers part of it.
- Coverage was tested only on the CNN combination, with figures shown and no scalar metric.
- Dark-matter-only simulations, with baryons tested only by post-hoc injection.
