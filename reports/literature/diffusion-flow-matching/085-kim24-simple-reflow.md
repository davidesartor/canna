# Simple ReFlow: Improved Techniques for Fast Flow Models

Kim et al., 2024, venue not stated in the text · [arXiv:2410.07815](https://arxiv.org/abs/2410.07815)

**Evidence: B.** Peer-reviewed (International Conference on Learning Representations); 38
citations in 2.0 yr (Semantic Scholar, 2026-10-19).

## What they do

- The design space of ReFlow, restricted to choices that preserve the FM minimiser:
  - any positive weight w(x_t, t);
  - any time density with full support;
  - any loss ||φ(x) − φ(y)||² with φ an invertible linear map (Prop. 1).
- Within these choices: per-sample loss normalisation, an exponential time density, a high-pass-filter-preconditioned MSE, lower dropout, forward and projected coupling pairs, and a sigmoid time grid with DPM-Solver(r).

## What they find

- Final: CIFAR10 2.23, AFHQv2 2.30, FFHQ 2.84, ImageNet-64 3.49. With guidance: 1.98 / 1.91 / 2.67 / 1.74.
- Baselines: OT-CFM 4.44 at 100 NFE; FM at 100 NFE 2.96; minibatch-OT FM gets worse with larger batches (3.16–3.29).
- Ablations, CIFAR10 / AFHQ / FFHQ:
  | change | CIFAR10 | AFHQ | FFHQ |
  |---|---|---|---|
  | baseline | 2.83 | 2.87 | 4.28 |
  | + w = 1/sg[L(x_t, t)] | 2.61 | 2.74 | 3.83 |
  | + time density ∝ 10^t | 2.62 | 2.69 | 3.77 |
  | + φ = I + 10·HPF | 2.58 | 2.55 | 3.69 |
  | + optimal dropout (0.09/0.09/0.03) | 2.43 | 2.53 | 3.17 |
  | + forward / projected pairs | 2.38 | 2.44–2.47 | 3.13 |
  | + sigmoid κ = 20 grid with DPM-Solver r = 0.4 | 2.23 | 2.30 | 2.84 |
- Weights: 1/t and 1/t² help over uniform, but the per-sample normaliser is best.
- Time densities: lognormal is much worse (3.28 / 3.20 / 4.48), so t near 1 (noise) must be emphasised.
- Losses: PH and LPIPS give wrong marginals in a 2-D GMM test.
- ODE grids: the EDM grid is worse because t_N < 1 (6.78 on FFHQ).

## Implementation details

- The loss weight w = exp(−f_ψ(x_t, t)) comes from a small net that tracks log-loss (EDM2-style uncertainty weighting), and needs to be x_t-dependent.
- Spatial loss preconditioning by an invertible linear map keeps the optimum unchanged while reweighting gradient directions.
- The ReFlow (OT-like) denoiser needs higher Lipschitz capacity, hence less dropout.
- Discretise time densely at both ends.

## Caveats

- Small images, built on EDM initialisation. Two of the seven fixes, the HPF loss and the forward pairs, are specific to images and data.
- Guidance's effect on the marginals is unclear.
