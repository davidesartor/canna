# Real-time gravitational-wave science with neural posterior estimation

Dax et al., 2021, PRL 127, 241103 · [arXiv:2106.12594](https://arxiv.org/abs/2106.12594)

**Evidence: A.** Peer-reviewed (Physical Review Letters); 232 citations in 5.3 yr (Semantic
Scholar, 2026-10-19).

## What they do

DINGO, the first fully amortised BBH NPE (15-D, real GWTC-1 events). It rests on three additions:
1. **PSD conditioning**: the flow is q(θ | d, S_n).
2. **An SVD-initialised embedding network.**
3. **GNPE**: per-detector coalescence times are handled by training on data time-shifted by blurred proxies τ_I = t_I + δt_I, with κ uniform over ±1 ms. Inference then Gibbs-iterates between τ_I and θ for ~30 iterations.

## What they find

- 8 GWTC-1 events. The mean JSD to LALInference MCMC is 0.0009 nat, against 0.0007 nat between LALInference seeds and a 0.002 nat "indistinguishable" threshold.
- 1000-injection P-P plot, combined p = 0.46.
- 50k samples with 30 GNPE iterations take about 20 s, versus O(day) for standard samplers.
- About 1.3·10⁸ parameters; training takes about 10 days on one A100.
- Feeding the wrong PSD as context gives a max JSD of 0.03 nat (t_c, d_L, sky), so PSD conditioning is required at this accuracy.

## Implementation details

- **Input**: whitened FD strain (Re, Im) plus the inverse ASD (more stable than the PSD at spectral lines; scaled by 10²³), 24,096 inputs per detector.
- **Embedding**:
  - The first layer is linear, *initialised* to the first 200 complex SVD vectors of 50k prior waveforms. The ASD projection is initialised to zero and added on.
  - A residual MLP follows: 6 blocks each at widths 1024, 512, 256 and 128 (48 layers, batch norm + ELU), giving 128 features.
- **Flow**: 30 RQ-spline coupling transforms, each with 5 residual blocks, with random permutations between transforms.
- **Training set**: 5·10⁶ intrinsic waveforms stored in SVD form (mismatch 2·10⁻⁵ at the 99.9th percentile). Extrinsic parameters (t_c, α, δ, d_L, ψ) and noise are drawn on the fly, making the set effectively infinite. No overfitting.
- **Distance-prior partitioning**: they "found it difficult to train ... over the entire relevant range of luminosity distance", so separate networks cover [100, 2000], [100, 6000] and [100, 1000] Mpc.
- **Two-stage curriculum**:
  - Pretrain 300 epochs on a fixed average PSD with the SVD layer frozen, at lr 3e-4.
  - Fine-tune 150 epochs with sampled PSDs and the layer unfrozen, at lr 3e-5.
  - Batch 4096, Adam, lr halved when validation stalls for 10 epochs.
  - The loss jumps when fine-tuning starts and then recovers to just above the pretraining loss.
- **GNPE start**: a separate network supplies the initial t_I for GNPE.

## Caveats

- GNPE costs latency from its Gibbs iterations, and it is dropped in DINGO-BNS.
- No IS correction in this paper (it arrives later with DINGO-IS).
- Stationary Gaussian noise; the prior is partitioned by distance.
