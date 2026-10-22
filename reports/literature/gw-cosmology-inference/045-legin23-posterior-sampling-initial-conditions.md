# Posterior Sampling of the Initial Conditions of the Universe from Non-linear Large Scale Structures using Score-Based Generative Models

Legin et al., 2023, MNRAS 527, L173 · [arXiv:2304.03788](https://arxiv.org/abs/2304.03788)

**Evidence: B.** Peer-reviewed (Monthly Notices of the Royal Astronomical Society Letters); 24
citations in 3.5 yr (Semantic Scholar, 2026-10-19).

## What they do

Learn the conditional posterior score of z = 127 initial-condition fields (128³) given the noisy z = 0 dark-matter density field (Gaussian noise with σ = 0.1). Sample with a VE reverse SDE. The posterior is implicitly marginalised over cosmology because training spans the Quijote Latin hypercube.

## What they find

- Sample power spectra are within 1% of the truth for all k except within about 10% of the Nyquist frequency.
- Samples explain 92% of the true initial-field variance. This is consistent with the information destroyed by halo collapse: a Press-Schechter collapse fraction of about 12%, compared with 16% measured.
- Cross-correlation C(k) is far above the optimal linear filter.
- Generalises to the fiducial Planck cosmology, which was not in the training set.
- Coverage test: the voxel-wise histogram of (x_true − μ_samples)/σ_samples is consistent with N(0, 1) (fig. 5).
- Cost: minutes per sample on one GPU, compared with about 100 N-body runs per independent BORG sample.

## Implementation details

- Conditioning by channel concatenation of y with x_t (Batzolis+21 justify this as valid for conditional scores).
- Network: NCSN++/RefineNet U-net with 5 levels, 2 BigGAN ResNet blocks per level, 32/64 channels, dropout 0.1.
- Training: Adam lr 2e-4, gradient clipping at norm 1, batch 8 on 4 H100s, about 400 epochs (about 24 h), 1,900 training and 100 test simulations.
- Sampling: VE SDE, σ_max = 100, σ_min = 0.01, N = 1000 Euler-Maruyama steps; 100 samples per test field.
- **Per-sample variance normalisation of the target field** (unit variance) improved convergence. The physical variance is restored afterwards from the predicted power spectrum, which means the network does not have to learn a cosmology-dependent amplitude.

## Caveats

- The coverage test checks only the first two voxel-marginal moments, using the posterior's own σ.
- Tested on idealised DM fields with simple noise.
- Only about 2,000 training simulations, so there is a risk of overfitting at 128³.
