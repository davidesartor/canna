# F50 (2026-10-18): STFT tokens launched, XS-stft and B-stft against XS-rope and B-rope

- Each 16-bin token becomes its band's 16 time samples, encoded as asinh amplitude + gated
  phase. Also `decay_adamw: false`.

User's proposal (after F49, setting F46–F49 aside): whitened data span ~3 decades (noise ~1 per
bin, loudest bins 125–630 on B), poorly served by arcsinh re/im; φ₀ and ψ are better
represented in a short-time Fourier transform.

**Design** (`network.stft_tokens`, `LisaFlow(stft=True)`):
- Same tokens, bands and RoPE positions (4 on XS, 256 on B, 16 rfft bins each). Each becomes 16
  time samples, T_obs/16 = 45.7 days apart, by an orthonormal inverse DFT of its bins:
  critically sampled, invertible, no leakage, noise stays white. A tapered overlapping STFT
  would leak and duplicate, so not used.
- Per complex sample of A, E, T: asinh|z| (log-like loud, linear near 0, noise the only scale;
  a fixed raw-TDI floor would sit at a different depth per window) and the gated phase z / √(1
  + |z|²) (unit phasor when loud, 0 where noise). Smooth at 0; together they give z.
- 144 scalars per token (96 before): amplitude 128 bins on [0, asinh 1000], 0.06 wide (4× finer
  than re/im); phase 64 bins on [−1, 1]. At |z| = 300 one noise σ is 0.1 phase bin and 5% of an
  amplitude bin vs 1.4% of an re/im bin: 4–7× finer.
- In this basis sub-bin f₀ is the phase slope across a token, φ₀ (with ψ's phase) its
  intercept, the sky the annual modulation (8 samples per year). φ₀ is not new information; the
  STFT separates intercept from slope and modulation.
- `decay_adamw: false` so a dead φ₀ can't again be F48's decay. If φ₀ returns the two changes
  can't be separated; if it stays dead, both are ruled out.
- Not included: template, Fisher, F49's helicity coordinates. B-fd-helicity not launched.

**Configs:** `XS-stft` (1.2M steps, ~12 h), `B-stft` (200k, ~11 h), against `17-lisa-XS-rope`
and `18-lisa-B-rope`. Tests: `tests/lisa/test_stft_tokens.py`.

**Predictions (before the result):**
1. φ₀ at epoch ~50: cos/sin φ₀ velocity spread ≥ 20% of target (0.3–0.5% since F39); within 3×
   of the MCMC where it pins it (XS q0.10 s0–s2 at 30–61×, B q0.10 s0 at 47×).
2. ψ within 1–2× of the MCMC on loud sources (now 2.3–10×).
3. Loudest sky (XS 16–17×, B-rope 66–106×) narrows; no change points back at the bf16 cast
   (F49).
4. f₀ and found at least B-rope's (215/240, 0.035 bins loud) and XS-rope's (0.0106 bins loud);
   a regression means the slope is harder to read than the bins' leakage.
