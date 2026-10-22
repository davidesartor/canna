# Complex Ratio Masking for Monaural Speech Separation

Williamson, Wang & Wang, 2016, IEEE/ACM TASLP 24 ·
[DOI:10.1109/TASLP.2015.2512042](https://doi.org/10.1109/TASLP.2015.2512042)

**Evidence: A.** Peer-reviewed (IEEE/ACM TASLP); 840 citations in 10.5 yr (Semantic Scholar,
2026-10-19; OpenAlex 798).

## What they do

The paper introduces the complex ideal ratio mask (cIRM): the complex S/Y per T-F bin, applied
by complex multiplication. A real DNN predicts the mask's Re and Im parts as two linear output
heads.

- **Network**: 3 × 1024 ReLU layers, linear outputs, 3-frame output context.
- **Inputs**: real "complementary" features (AMS, RASTA-PLP, MFCC, a 64-channel gammatone
  cochleagram, plus deltas), spliced over 5 frames.
- **Alternative inputs tried**: noisy magnitude, magnitude + phase, and noisy RI. All were
  "not as good as the complementary set".
- **Target compression**: Re and Im of the cIRM are compressed with
  K(1 − e^{−C·M}) / (1 + e^{−C·M}), with K = 10 and C = 0.1, to bound the target.
- **Loss**: MSE on compressed Re and Im.
- **STFT**: 40 ms Hann window, 50% overlap, 640-point FFT, 16 kHz.
- **Data**: IEEE corpus, one male speaker. Noises: speech-shaped (SSN), cafe, babble, factory.
  Training at −3/0/3 dB, testing at −6..6 dB.

## What they find

- **Masks compared at −3 dB with SSN noise**, PESQ:
  - cRM 2.51;
  - phase-sensitive mask (PSM) 2.34;
  - ideal ratio mask (IRM, magnitude-only) 2.23;
  - complex matrix factorisation (CMF) 1.98.

  With babble noise the result is mixed (PSM 2.46 vs cRM 2.44).
- **Overall**: cRM gives PESQ 2.62 / STOI 0.81 / frequency-weighted SNR 8.08. Direct STFT
  estimation (regressing the spectrum) gives 1.92 / 0.68 / 3.68.
- **Listening tests**: cRM preferred over IRM 89%, over PSM 69%, over time-domain
  reconstruction 85%.
- **Phase vs RI**:
  - The phase spectrum "looks rather random". Extensive attempts to estimate phase directly had
    "no success".
  - The Re and Im components of the mask "show clear structure", which is the stated reason
    for predicting the mask in Cartesian form.

## Implementation details

- The Cartesian mask is bounded by a saturating compression; the compressed values are inverted
  at inference.
- The network is a plain real MLP.

## Caveats

- Small single-speaker corpus with a feed-forward DNN; pre-deep-learning-era features.
- The direct-spectrum baseline is a weak, unbounded regression. Later work (Tan & Wang,
  file 46) found direct RI mapping beats cIRM masking when the network is a larger GCRN.
- The numbers come from the PMC author manuscript, read via a fetched summary.
