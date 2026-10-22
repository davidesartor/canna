# F25 (2026-10-06): B0 — B costs 1.57 s per step at 512 wide and 2.57 s at 768; the eval needs chunking

- One B step is 41× (512) and 43× (768) an XS step; memory is no constraint at batch 256.
- The eval OOMed and now pushes draws in chunks of 256; B scorecards use `--n_random 50`.
- Default proposal: `B-late` (512) for 200k steps, ~3.6 days. The user decides.

**Training benchmark** (jobs 13875733 `B-late`, 13875737 `B-late-768`; 300 steps, one A100,
batch 256):

| width | first epoch (incl. compile) | steady epoch (100 steps) | per step | peak GPU memory |
|---|---|---|---|---|
| 512 | 224 s | 157 s | **1.57 s** | 18.6 GiB of 59 |
| 768 | 334 s | 257 s | **2.57 s** | 29.4 GiB of 59 |

The XS/S fit's 1.5 s held. 59 GiB is JAX's default 75% cap.

| steps (20% cooldown) | 512 | 768 |
|---|---|---|
| 100k | 44 h, 2 slots | 71 h, 3 slots |
| 150k | 65 h, 3 slots | 107 h, 5 slots |
| 200k | 87 h, 4 slots | 143 h, 6 slots |

A 1M-step XS run took 11 h (512) or 17 h (768), so B affords roughly a tenth to a fifth of XS's
steps.

**Eval.** Job 13875840 ran out of memory: `sample_posterior` pushed all 1024 draws through 2048
tokens at once (one 27.3 GiB buffer). Fixed by chunks of 256 (`eval.CHUNK`), memory only; a
test checks it against the unchunked transport to float32 rounding. The first chunk is exactly
the scorecard's 256 draws.

**Scorecard** (job 13875770): ~78 s per injection at 768, so the default 10 + 200 injections
would take ~4.5 h. **B uses `--n_random 50`**: 60 injections, ~1.3 h, ~240 sources.

**Budget.** At fixed GPU budget the 512 gets 1.64× the steps of the 768. At equal steps on XS
the 768 bought +11 of 840 sources and 11–15% sharper widths below SNR 100 (F24), but B is far
short of 1M steps and each doubling of steps was worth ~20% in width plus the coarse gains
(F12). **Default: `B-late` (512) for 200k steps, 4 chained slots**; the same-cost alternative
is `B-late-768` for ~120k.
