# Complete parameter inference for GW150914 using deep learning

Green & Gair, 2020, MLST 2, 03LT01 (2021) · [arXiv:2008.03312](https://arxiv.org/abs/2008.03312)

**Evidence: A.** Peer-reviewed (Machine Learning: Science and Technology); 117 citations in 6.2
yr (Semantic Scholar, 2026-10-19).

## What they do

the first NPE to give full 15-D precessing-BBH posteriors on real data (GW150914, H1+L1) that are close to bilby/dynesty. It uses a neural spline flow conditioned on SVD-compressed whitened strain, with all extrinsic parameters applied on the fly during training.

## What they find

- Close agreement with bilby on GW150914. The NPE puts more weight on a secondary θ_JN mode and a small secondary sky mode.
- A 100-injection P-P plot passes the KS tests.
- Sampling runs at about 5000 samples/s.
- Training takes about 6 days on one Quadro P4000: 500 epochs, batch 512.

## Implementation details

- **Flow**: 15 rational-quadratic spline coupling steps with K = 9 bins. The conditioner is a residual net of 10 blocks × 512 units with batch norm and ELU.
- **Data**: whitened FD strain (20–1024 Hz, 8 s), projected onto a 100-component SVD basis built from waveforms.
  - Raising n_SVD slightly *hurt*, because the higher components are mostly noise.
  - White noise is added directly to the reduced-basis coefficients; this is valid because the basis is orthonormal, so the transform is a rotation.
- **Training set**: 10⁶ precomputed intrinsic waveforms. Extrinsic parameters (d_L rescaling, antenna projection, time shift) are applied at train time, giving effectively unlimited augmentation.
  - Time shifts in the reduced basis use precomputed time-translation operators with cubic interpolation.
- **Standardisation**: θ and the data are standardised. The fixed PSD is estimated from the 1024 s before the event.
- **Proxy prior**: trained on a d_L prior uniform up to 1000 Mpc, then **reweighted** to the physical prior.
- **Optimiser**: lr 2e-4 with cosine annealing to 0; no overfitting observed.

## Caveats

- The network is tied to one PSD, and the paper proposes PSD conditioning as the fix (later done in DINGO).
- One event, two detectors.
- Small but visible excess support in secondary modes.
