# Sensitivity Analysis of Simulation-Based Inference for Galaxy Clustering

Modi et al., 2023, MNRAS · [arXiv:2309.15071](https://arxiv.org/abs/2309.15071)

**Evidence: B.** Peer-reviewed (Monthly notices of the Royal Astronomical Society); 19
citations in 3.1 yr (Semantic Scholar, 2026-10-19).

## What they do

- Factorial misspecification study. The forward model is gravity (Quijote N-body or FastPM PM) × halo finder (FoF or Rockstar) × HOD (5-parameter Zheng07 or 10-parameter Zheng07ex), giving 6 working combinations.
- The NPE is trained on one combination and tested on another, swapping one component at a time.
- Summaries: P_ℓ (79×3) and B0 (1,980 triangles) to k = 0.5, with a DESI-LRG-like n̄ = 4e-4.

## What they find

- Gravity: NPE trained on FastPM is unbiased and calibrated on N-body test data. PM costs 200 CPU-h per sim against 5,000 for N-body.
- Halo finder: Ω_m is always fine. σ8 is fine from P_ℓ but biased from B0: FoF-trained NPE under-predicts σ8 on Rockstar data, and vice versa.
- HOD: training on the complex model and testing on the simple one is fine (nested model). Training on the simple model and testing on the complex one gives bias, mild for P_ℓ and severe for B0, which "almost always" gives wrong posteriors for both parameters.
- Component-level summary differences (10–20% in the quadrupole and bispectrum from the halo finder) do not predict inference bias. Only end-to-end tests do.

## Implementation details

- Data: 1,500 / 200 / 300 cosmologies for train / validation / test, × 20 HOD draws each = 40k catalogues per forward model.
- Cosmology-dependent HOD mass priors (log M_min etc. centred on values that hit the target n̄) to raise sample efficiency over one broad prior.
- Extra shot noise S_n ~ U[1e3, 1e4] is added to P0 and marginalised. Conclusions are unchanged without it.
- SNPE-C (`sbi`) MAFs. 400 networks per statistic from a WandB search; top-10 uniform mixture ensemble.
- SBC rank histograms on in-distribution test data, which the authors flag as necessary but not sufficient.
- A bounded σ8 prior creates a residual slope from prior edges, not misspecification.

## Caveats

- Only 2 parameters are reported, and only P_ℓ and B0.
- Rockstar is incompatible with PM by default.
- FoF concentrations are analytic, so FoF-based assembly bias is not real assembly bias.
