# F42 (2026-10-14): B's floor is not the bf16 cast; XS and B relaunched with the fixes, with and without RoPE

- Evaluating B-sfplus's x in float32 changes nothing: per source float32/bf16 = 0.99–1.00.
  F41's bound matched the floor by coincidence.
- Four full runs launched with the fixes kept, two of them adding RoPE in frequency.

**Float32 test** (14507218; the same 20 injections and keys as the bf16 scorecard):

| SNR | n | median f₀ width (bins), bf16 / float32 | per source, float32 / bf16 [16–84%] |
|---|---|---|---|
| < 15 | 13 | 4.08 / 4.15 | 0.99 [0.90–1.05] |
| 15–40 | 11 | 3.61 / 3.58 | 0.99 [0.95–1.04] |
| 40–100 | 22 | 3.10 / 3.19 | 0.99 [0.98–1.03] |
| ≥ 100 | 34 | 3.36 / 3.37 | **1.00** [0.98–1.04] |

A bf16-trained network could in principle fail to use float32 at eval (F21), but nothing moves.
Remaining candidates for B's ~3-bin floor:
- the f₀ PLE bin (12 frequency bins on B);
- how sources find their tokens: learned absolute token positions against a source coordinate
  in log f (RoPE targets this);
- training.

**Launched** (from `d041316`, TREX reset onto the squashed history, old head on
`trex-pre-squash-2`; no ablations, at the collaborator's request). All use
`slurm/lisa-local.sbatch` (in-job eval draws, optimizer state left behind) and keep the F39
schedule-free fixes and NorMuon, 1/L1 weighting off (F40), SNR-sorted sources with
reconstruction loss 0.1, 16 bins per token.

| job | config | kept as |
|---|---|---|
| 14509892 | XS-fd | `lisa-XS-fixed` |
| 14509893 | B-fd | `lisa-B-fixed` |
| 14509894 | XS-fd-rope | `lisa-XS-rope` |
| 14509895 | B-fd-rope | `lisa-B-rope` |

**RoPE in frequency** (the user's proposal): each source token sits at its current f₀, each
data token at its centre bin, in bins from the window start (`WindowGeometry.source_bins`,
`token_bins`). q and k are rotated after the qk-norm in every attention, angles in float32, 32
periods per head log-spaced from 2 bins to twice the window. A source's score on a token then
depends on f₀ − f_token.
- Unlike R1's additive position features, nothing is added to the residual streams; sources
  stay labelled by SNR.
- Smoke run: trains cleanly, 27 s per 100-step epoch at batch 64. Tests:
  `tests/lisa/test_rope.py`.

**Readout:** each pair with vs without RoPE, and the fixed recipe vs XS-sfplus (F40) and XS-fd
(F33): loud floor, sources found, calibration, MCMC windows (XS). B's rule stays < 1 bin.
