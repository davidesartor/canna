# Rethinking Complex-Valued Deep Neural Networks for Monaural Speech Enhancement

Wu et al., 2023, Interspeech 2023 · [arXiv:2301.04320](https://arxiv.org/abs/2301.04320)

**Evidence: B.** Peer-reviewed (Interspeech); 13 citations in 3 yr (Semantic Scholar,
2026-10-19).

## What they do

The paper is a controlled comparison of complex-valued speech enhancement networks against
real-valued counterparts with **matched parameter counts**. One complex weight counts as two
real ones, so the real models get about √2× the channels or units. Both kinds of model see the
same complex STFT, which the real model takes as stacked (Re, Im) channels. Training and test
data are the DNS Challenge 2020 data at −5/0/5 dB SNR. The pairs are:

- a linear stack;
- an LSTM, plus a "quasi-complex" LSTM;
- DCUNET vs a real U-Net (RUNET);
- GCRN vs a complex GCRN, which tries several complex gating variants;
- DCCRN-E vs a real DCCRN.

Every model uses split ReLU (applied separately to Re and Im). The paper also reports MACs, not
only parameter counts.

## What they find

- **No gain from complex arithmetic at matched size.** The real model ties or slightly wins in
  every pair:
  - **DCUNET vs RUNET** (3.10M vs 3.12M parameters): PESQ 1.64/1.92/2.27 vs 1.70/2.00/2.36;
    SI-SDR 10.80/13.79/16.74 vs 10.87/13.86/16.84. The complex model costs 56.69G MACs
    against 19.87G.
  - **LSTM**: complex 1.65/1.95/2.29, quasi-complex 1.71/2.02/2.35, real 1.69/2.00/2.34 PESQ
    (23.35M vs 23.62M parameters; 5.90G vs 2.98G MACs).
  - **Linear stacks**: 406 complex vs 512 real units at 0.59M parameters. Real is slightly
    better at half the MACs (59.88M vs 119.59M).
  - **GCRN**: real 9.25M parameters / 1.72G MACs gives PESQ 1.57/1.87/2.24 and SI-SDR
    8.30/11.29/14.13. The best complex variant (8.83M / 2.57G) gives PESQ 1.57/1.88/2.27 and
    SI-SDR 7.65/10.87/13.82. That variant gates on magnitude, F_g = (σ(|F2|) − 0.5)·2.
  - **DCCRN-E vs real DCCRN** (3.67M vs 3.64M; 14.38G vs 4.84G MACs): PESQ 1.78 vs 1.80 and
    SI-SDR 11.01 vs 11.06 at −5 dB; on the DNS blind set, PESQ 2.92 vs 2.89 and DNSMOS 3.30
    vs 3.33.
- **Complex models lose more as they shrink.** SI-SDR at −5/0/5 dB, complex GCRN vs real GCRN:
  - ~2.3M parameters: 6.60/9.75/12.58 vs 7.25/10.26/12.94;
  - ~0.6M parameters: 4.14/6.80/8.73 vs 5.82/8.94/11.85.
- **Optimisation**: DCCRN was less robust to the learning rate than its real twin.
- **Cost**: across all pairs, complex models need about 2–3× the MACs for the same parameters.

## Implementation details

- **Input**: complex STFT. The complex models take it as complex tensors; the real models take
  it as 2 channels (Re, Im).
- **Matching rule**: parameters counted in real numbers, so a complex layer of width C is
  matched by a real layer of width about √2·C.
- **Activations**: split ReLU in all complex models.
- **Gating**: the best complex GCRN gate uses the magnitude |F2| through a shifted sigmoid, not
  a split-complex sigmoid.

## Caveats

- Speech data only, with perceptual metrics (PESQ, DNSMOS) plus SI-SDR. There is no direct
  metric for phase error.
- Short workshop-length paper; each comparison uses a single training run with no reported
  variance.
- Only split-ReLU nonlinearities are tested on the main pairs. modReLU and cardioid
  activations are not compared at matched size.
