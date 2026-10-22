# Decoding Long-duration Gravitational Waves from Binary Neutron Stars with Machine Learning: Parameter Estimation and Equations of State

Hu, Irwin et al., 2024, ApJL 987, L17 (2025) · [arXiv:2412.03454](https://arxiv.org/abs/2412.03454)

**Evidence: B.** Peer-reviewed (Astrophysical Journal Letters); 12 citations in 1.9 yr
(Semantic Scholar, 2026-10-19).

## What they do

NPE for 3 h long BNS signals in next-generation detectors (1 ET + 2 CE, including the Earth-rotation response). Data compression has three stages: adaptive multibanding, then heterodyning by the leading 0PN phase, then SVD. A normalizing flow conditioned on a ResNet/ViT embedding is trained on narrow local chirp-mass priors. IS then makes the result exact. The paper also infers the EOS with a separate flow on a 12-D autoencoder latent.

## What they find

- Compression takes ~12M frequency points (5 Hz start) to ~6000 points in 81 bands, then to 128 SVD bases. Reconstruction mismatch has a median of 1e-8 and a maximum of 1e-6.
- The model has ~160M parameters, 96M of them in the NSF. Training takes 2–3 weeks on one A100, on 64M (low-SNR model) or 100M (high-SNR model) waveforms.
- Sampling takes under 1 s on an RTX 3080Ti. IS costs O(1)–O(10) CPU-hours, against ~3500 CPU-hours for bilby on an SNR-35 event.
- P–P tests pass even before IS. The initial samples are biased in the tidal parameters (the authors present Fig. 3 as a bad case), and IS removes the bias.
- The EOS examples are at SNR 39 and SNR 390.

## Implementation details

- **Separate models per SNR regime:** SNR 20–50 and 200–500. Each uses a very narrow Mc prior: U[2, 2.1] (low SNR) and U[1.3, 1.31] (high SNR). The authors say covering a wide Mc and SNR range together is "extremely challenging" because the posterior is far narrower than the prior.
- **SNR is sampled uniformly and d_L rescaled to hit it**, "reducing outliers". The resulting distance-prior mismatch is corrected by the IS weights.
- **Adaptive multibanding:** ~64 points per band, with safety factor α = 2 on the 3.5PN time-to-merger τ(f).
- **0PN heterodyne before SVD,** so that 128 bases suffice.
- **Reference jitter:** δM ~ U[±0.0005] (low SNR) or U[±0.0001] (high SNR), with M̃ given as context, so the network learns the offset from the reference.
- **Cached intrinsics, fresh extrinsics:**
  - The 12 intrinsic parameters are drawn once up front and their SVD projections cached.
  - Extrinsics, t_c and SNR are drawn on the fly every epoch, so the expensive waveforms are reused.
- **φc is left out of the flow:** 16 of 17 parameters are inferred while φc still varies in the data, so φc is implicitly marginalised.
- **EOS:** a RealNVP flow on the 12-D latent of a convolutional autoencoder (CUTER EOS set).

## Caveats

- Each SNR band and Mc slice needs its own very expensive model (weeks on an A100). Amortisation is local, not global.
- The pre-IS tidal bias shows that the flow alone is not accurate at the required precision.
- IS costs CPU-hours rather than seconds.
- Results are for a handful of injections. No IS-efficiency statistics over a population are reported.
