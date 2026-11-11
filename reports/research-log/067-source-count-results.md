# F67 (2026-11-11): the source-count runs are in; narrow-2 matches the joint MCMC, every doubling of S widens the flow, the full band loses its loud sources above 10 mHz, and the data-space ranks see all of it without the truth

**Outcome.**
- All six runs of [F66](066-source-count-runs-launched.md), their 24 cooldown forks and 30 evals
  finished on TREX (forks + evals + SBC scorecards by the manager job, last task 11 Nov 08:23 UTC).
  Five main runs trained to their end with every loss finite; full-4 diverged at epoch 51 (see
  below). full-16, the last, trained cleanly: worst single-step frame NLL 1.85, catalog NLL still
  falling at epoch 64.
- **Narrow band.** narrow-2 = the joint MCMC: every width ratio 1.00–1.17 at SNR 7–40 (f₀ 1.03),
  92% found at SNR 7–15, 7% of loud f₀ draws off the MCMC peak (MCMC itself 5%). Each doubling of
  S widens and fattens the tails: narrow-4 f₀ 1.17–1.23×, sky 1.32×, 16% off-peak; narrow-8 f₀
  2.1× (7–15) / 1.4× (15–40), 66% found at 7–15, 29% off-peak, and its faint f₀ is 5.6× the MCMC.
- **Full band.** Close to the MCMC only below SNR 15. Loud (SNR ≥ 40) f₀ is 9.7×, 10.7× and 23.6×
  the MCMC for S = 4, 8, 16 (best model each); log ḟ 7.7 / 9.1 / 12.8×, α 7.0 / 8.6 / 14.8×, A and
  ι within 1.7 / 2.0 / 2.9×. Found at SNR ≥ 40: 80 / 66 / 50%. The misses are above 10 mHz: there
  the share found falls from 63% (S=4) to 6% (S=16), below 10 mHz only from 94% to 82%.
- **Calibration hides all of it.** Every one of the 30 evals covers 0.66–0.69 at 68% and 0.94–0.95
  at 95% after relabelling, including the full-band models that are 10–24× too wide on loud
  sources (as the mock ladder predicted).
- **The validation pass sees it without the truth.** On 1000 prior injections per model the
  coordinate ranks and TARP pass or fail at the per-cent level, like the Gaussian mock. The
  data-space ranks reject every model at p < 0.001, always the same way: the truth overlaps the data
  more than the draws do, and the truth's log-likelihood sits above nearly every draw's. The
  log-likelihood rank offset (mean rank/M − ½) is +0.31 for narrow-2 (29% of ranks in the middle
  half) and +0.42 to +0.48 for every other model (2–10% in the middle half). Excess width in each of
  8S coordinates adds up to a large likelihood loss, which is what this test measures.
- **Longer runs help, and the cooldown matters most on the full band.** Catalog NLL falls with
  the fork epoch everywhere (narrow-2 −1.25 → −1.38 from e4 to e32; full-16 −0.63 → −0.89, loud f₀
  59 → 24×). full-4's uncooled main run (best epoch 43) has loud f₀ 17× against 9.7× for its e32
  fork. Beyond epoch 32 the main runs gain on two configurations (narrow-4, full-8) and lose on three
  (narrow-2, narrow-8, full-16), by a few hundredths of a nat: inside the unmeasured seed spread.

**Best model per configuration** (lowest catalog frame + flow NLL among main and forks; flow/MCMC
width, median over sources of the SNR band; off-peak = share of SNR ≥ 15 f₀ draws farther than
max(0.5 bin, 5 MCMC widths) from the MCMC median, MCMC's own in brackets).

| config | model | cat. NLL | f₀ 7–15 | f₀ 15–40 | f₀ ≥40 | α 15–40 | off-peak | found 7–15 / 15–40 / ≥40 |
|---|---|---|---|---|---|---|---|---|
| narrow-2 | fork e32 | −1.379 | 1.04 | 1.03 | 1.1 | 1.17 | 0.07 (0.05) | 0.92 / 0.97 / 1.00 |
| narrow-4 | main | −1.135 | 1.17 | 1.23 | 1.6 | 1.32 | 0.16 (0.09) | 0.79 / 0.91 / 1.00 |
| narrow-8 | fork e32 | −0.974 | 2.11 | 1.41 | 1.6 | 1.84 | 0.29 (0.05) | 0.66 / 0.84 / 0.92 |
| full-4 | fork e32 | −1.200 | 1.64 | 1.48 | 9.7 | 1.90 | 0.32 (0.02) | 0.62 / 0.71 / 0.80 |
| full-8 | main | −1.105 | 1.69 | 2.19 | 10.7 | 2.69 | 0.40 (0.02) | 0.67 / 0.69 / 0.66 |
| full-16 | fork e32 | −0.887 | 3.27 | 2.37 | 23.6 | 2.90 | 0.49 (0.00) | 0.56 / 0.68 / 0.50 |

All 30 evals: `outputs/runs/261103-summary-2026-11-11.txt`.

**Validation pass** (scorecard.npz; γ-test p, all strata pooled; strata by the draws' median own
SNR in the paper's Table `tab:results-validation`).

| model | coords (min p) | TARP slot | TARP joint | overlap | own SNR | log-lik | log-lik offset | middle half |
|---|---|---|---|---|---|---|---|---|
| narrow-2 e32 | 0.110 | 0.835 | 0.764 | <0.001 | 0.042 | <0.001 | +0.31 | 0.29 |
| narrow-4 main | 0.032 | 0.815 | 0.288 | <0.001 | <0.001 | <0.001 | +0.43 | 0.10 |
| narrow-8 e32 | 0.001 | 0.135 | 0.033 | <0.001 | <0.001 | <0.001 | +0.47 | 0.03 |
| full-4 e32 | 0.016 | 0.100 | 0.536 | <0.001 | 0.019 | <0.001 | +0.42 | 0.09 |
| full-8 main | 0.135 | 0.206 | 0.036 | <0.001 | <0.001 | <0.001 | +0.47 | 0.02 |
| full-16 e32 | 0.151 | 0.001 | 0.014 | <0.001 | <0.001 | <0.001 | +0.48 | 0.02 |

The Gaussian (Fisher) mock of the mock ladder also rejects the log-likelihood rank, so p alone does
not convict; the offset's growth with S and band, in step with the width ratios, does. A null offset
from the Gaussian mock on these strata is still missing.

**full-16 in detail.** Faint (SNR < 7) f₀ 2.0× the MCMC (like narrow-8's 5.6×, milder) and the
faint sky 0.81–0.85×, i.e. *narrower* than the chains; full-16's chains are the least converged of
the six references (R̂ up to 1.038), and a faint sky is multimodal, so check the chains before
blaming the flow. Wall time 63.8 h + queue waits, two self-requeues; one requeue segment's `uv`
step was recorded FAILED 15:0 at the requeue signal (the normal requeue path), the run resumed.

**full-4's divergence** (6 Nov, already in the session notes): non-finite loss at step 2 509 855,
epoch 51, just before the cooldown, after isolated single-step spikes (frame NLL 10⁴ → 10¹⁷, grad
norm to 10¹⁸) from epoch 48, F65-like. The manager never evaluates a failed run, so its best
checkpoint (epoch 43) was evaluated and scored by hand. full-8 and full-16, same code, did not
spike.

**Wall time** (sacct segments in the `.out` logs): narrow 32–34 h with one requeue each, full-8/16
66.0 / 63.8 h with two; full-4 52.9 h to its divergence.

**Data, pages, scripts.**
- Runs (no checkpoints/snapshots/best) local in `outputs/runs/261103-*`, numbered 35–40
  (`35-lisa-narrow-B-n2` … `40-lisa-full-B-n16`); Slurm logs, manager dir and the analysis logs in
  `outputs/trex-logs/source-count-261103/`. TREX keeps the checkpoints and snapshots (~12 GB per run).
- Per eval: `corner.html` (flow + Fisher + joint MCMC), `validation.html` (now with the scorecard),
  per run `run.html`. Built on TREX (CPU job 17449558, 6 min) from a checkout of the runs' code plus
  the six MCMC references of origin/oneshot (`reports/mcmc/data/eval-*.npz`).
- Analysis: `outputs/runs/261103-{score_evals,vs-mcmc,validation,summary,pages,validation-pages}.py`
  → `261103-{scores,vs-mcmc,validation}-2026-11-11.json`, `261103-summary-2026-11-11.txt`;
  paper input `paper/data/runs/source-count.json` from `paper/scripts/import_runs.py`.

**Next.**
- Whether the frame or the flow is the wide part on loud full-band sources above 10 mHz (needs the
  checkpoints, on TREX).
- A null for the validation offset: the Gaussian mock's log-likelihood rank offset on these strata.
- full-16's faint sky: rerun or extend its chains before reading the 0.81× as a flow error.
- Seed spread on one cheap configuration (narrow-2) before reading fork differences of < 0.05 nat.
