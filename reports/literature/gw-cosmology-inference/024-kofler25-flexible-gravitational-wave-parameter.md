# Flexible Gravitational-Wave Parameter Estimation with Transformers

Kofler et al., 2025, PRL 136, 251401 (2026) · [arXiv:2512.02968](https://arxiv.org/abs/2512.02968)

**Evidence: B.** Peer-reviewed (Physical Review Letters); 10 citations in 0.9 yr (Semantic
Scholar, 2026-10-19).

## What they do

Dingo-T1 replaces DINGO's residual-network embedding with a transformer encoder over frequency-segment tokens.
- Multibanded FD data and PSD are cut into 16-bin segments per detector, giving 69 tokens per detector and 208 tokens including one learnable summary token.
- The model is trained with *data-based* token masking (missing detectors, f_min/f_max changes, PSD notches), so one model handles any detector/frequency configuration.
- The summary token is linearly projected to 128-D and conditions the same RQ-spline flow as DINGO-IS.

## What they find

- 48 O3 BBHs across 17 analysis configurations with one model; it replaces 94 fixed-setting DINGO models.
- Median IS efficiency on real events is 4.2%, versus 1.4% for the ResNet-NPE baseline (5.2% versus 1.8% excluding glitch-affected events). HL-only reaches 11%.
- On injections, median efficiency is 26.9% / 6.8% / 3.3% for 1 / 2 / 3 detectors. **Efficiency falls as posteriors narrow**: "posteriors constrained by more detectors are narrower and therefore more challenging to learn for normalizing flows."
- Random masking does worse than data-based masking: 15.7% vs 26.9% (1 detector) and 2.5% vs 4.2% (real events).
- Still below GNPE-level efficiency (~10% on HL), because no time-translation equivariance is used.

## Implementation details

- **Tokenizer**: FC layers plus a 512-wide residual block. The segment bounds (f_k, f_{k+1}) and a one-hot detector ID are injected through a **gated linear unit inside the tokenizer**, *instead of* additive positional encodings. They argue additive encodings suit continuous, non-uniform frequency poorly and multiply design choices.
- **Encoder**: 8 pre-LN layers, d_model 1024, 16 heads, FFN 2048. Pre-LN lets training run without LR warmup.
- **Token size**: 16 bins. 4 or 8 bins gave no gain at longer sequence length.
- **Context size**: anything from 56 to 256 dims barely changes the loss.
- **Optimiser**: AdamW (β₁ 0.8, β₂ 0.99), lr 1e-3, weight decay 0.005, ReduceLROnPlateau (halve after 10 epochs).
- **Scale**: batch 16,384 with AMP. 2.5·10⁷ intrinsic waveforms, extrinsic parameters on the fly. 183 epochs ≈ 9.4 days on 8×A100.
- **Masking mix**:
  - Detector drops: 60% / 30% / 10% for 0 / 1 / 2 detectors.
  - Frequency-range updates on 25% of samples.
  - Notches on 10% of samples.
  - About 25% of tokens masked on average; 15–30% was tried, and too much masking hurts 3-detector performance.
- **Phase**: the coalescence phase φ_c is **not inferred by the network**. It is reconstructed analytically in post-processing, and that step dominates the 5–10 min inference time.
- **IS**: importance sampling runs in the native uniform-frequency domain, which corrects the coarse token-level masking.
- **Diagnostic**: ε ≈ 0 on every configuration involving one detector exposed a bad Hanford PSD.

## Caveats

- Fixed 8 s duration; no amortisation over resolution or prior.
- GNPE is dropped, so efficiency is below the GNPE state of the art.
- The Bilby comparison covers only one event.
