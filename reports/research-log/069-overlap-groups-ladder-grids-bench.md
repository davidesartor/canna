# F69 (2026-11-18): no single parameter group is behind the flows' overlap offset, the mock ladder on the catalogue prior, sampler grids agree, and the full-band step at R = 128

**Outcome.**
- **The overlap offset is not one parameter.** F68 guessed "phases most likely" for the flows'
  +0.15–0.22 overlap rank offset (mock +0.04–0.05). Fixing one group of every draw's FLOW coordinates
  to the truth and re-ranking (`outputs/analysis-261117/overlap_groups_flow.py`, TREX 18078029, the six
  selected models and their Gaussian mocks on the same 1000 validation injections, read from the
  scorecards' saved samples; the unfixed "drawn" group reproduces the scorecard offsets within 0.002):
  - frequencies (x_f, x_fdot) or sky (l, b) fixed: mock offset unchanged (−0.02 to 0.00), flow's
    RISES to +0.24–0.30 (with phases too; +0.02 to +0.15 over drawn). Phases raise the mock's too
    (+0.04–0.10).
  - frequencies + phases fixed together: flow +0.21–0.24, less than either alone, so a drawn wrong f0
    carries a compensating phase: each draw is a coherent fit displaced from the truth along a
    direction that mixes the groups.
  - amplitudes are right: flow rank offsets of |z±| −0.01 to +0.02, own SNR 0.00 to +0.05.
  - only below snr_hat 7 does a group help: moduli |z±| (their ratio = inclination) take 0.01–0.06
    off the flow's offset while the mock's moves −0.01 to +0.04: faint polarisation is part of it.
  The first attempt (`overlap_groups.py`, 18032971) fixed PHYSICAL coordinates; every group raised
  the offset to +0.33–0.36 because physical phi0 is referenced at t = 0 and the flow's phases at
  mid-window (phi0 + pi f0 T): a draw one bin off in f0 with the true phi0 is ~pi out of phase. Kept
  as overlap-groups-<cfg>.json, not used.
- **Mock ladder rerun on catalogue-prior injections** (narrow-B and full-B at S = 4; ladder-mocks
  18032973 + ladder-post 18077399 after fixing fisher_probe.py's stale `matched` key; the reference
  comparison against the packed narrow-4 joint reference written as one-seed chain files). The prior
  mock still passes TARP and the coordinate ranks (min of 8 p = 0.030) and fails overlap and
  log-likelihood. The Gaussian mock is now rejected by EVERY test (most sources faint, linearisation
  fails) and next to the chains it is 0.54–0.65× in log A, sin iota, psi, phi0 even at SNR >= 7. Only
  6 of 800 validation sources reach SNR 40, so the loud-floor mock's strata tests cannot run; only the
  width ratio sees its floor (9.4× ideal vs 3.6× Gaussian, 16 sources at SNR >= 40). The table's width
  column now reads SNR >= 40 (was >= 100: one source), its MCMC column the 187 sources at SNR >= 7.
  fisher-scorecard/build_page.py kept only the old 64-window records (n_eval == 256): now 512.
- **Sampler grids** (TREX 18032972, narrow-4 and full-8 main runs, same sources as the 11-13 uniform
  sweep): with the midpoint rule the power-2 and log grids give 1.00–1.03× the uniform 128-step width
  from 16 steps up, both bands, so F60's grid sensitivity is gone on these runs. Euler converges from
  below: 0.78 (narrow) / 0.92 (full) at 32 uniform steps, 0.90 / 0.97 at 128.
- **Full-band step time at R = 128** (TREX 18033011, A100 80GB PCIe, size B, bf16 backbone): 73 ms per
  step at the training batch (64 windows x 16 draws), the batch draw 2.0 ms = 3% (R = 1024: 124 ms,
  9%); matches the 72–108 ms/step the source-count runs measured. R = 1024 records kept in
  a100-bf16backbone-full.R1024.jsonl.

**Not done.** prior-f16 rerun (18033012, 96 GB, chunk 10000) still running at 9 h on 18 Nov 00:00
UTC, labelfree 18033013 waits on it. Fisher forecast (app:evaluation-fisher) and the rest of
reports/mcmc still on the previous prior. `numbers_evaluation.py` and `numbers_priors.py` cannot run
whole on the laptop (reports/prior/*/data is gitignored and not regenerated); only their ladder and
bench blocks were rerun (outputs/analysis-261117/ladder_only.py).
