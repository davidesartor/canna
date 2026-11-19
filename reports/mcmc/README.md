# MCMC reference: the joint chains the flow is scored against (updated 2026-11-19)

The paper's reference posteriors are joint MCMC chains over all sources of each eval window, one reference per band
and source count. Their settings, health, convergence, cost and the Fisher cross-check are in the paper's
appendices (`app:mcmc`, `app:fisher`, `app:results-mcmc`); this folder holds the data and the scripts behind them.
Findings F67-F71 in [`../research-log/`](../research-log/000-takeaways.md) give the history.

## Takeaways

- **Six references, one per problem.** Narrow band S = 2, 4, 8 and full band S = 4, 8, 16, each on its 128 eval
  injections with one seed: 768 joint chains. 64 walkers x 6 temperatures (up to 10), 100 000 burn-in, 300 000
  kept, thin 100, started from the injection's Fisher draws; 1024 cold draws kept per injection.
- **Healthy by the usual diagnostics.** R-hat <= 1.073 (the worst is full S = 4; narrow band <= 1.005), ESS >= 1097.
  The chains' own coverage of the true f0 at 68% is 0.67-0.72.
- **Sound from SNR 7 up, not below.** Long chains of narrow S = 4 (injections 16, 48, 80, 112; four seeds x
  400 000 iterations, kept from the start) settle for the loud sources: read without labels, their f0 widths move
  by 0.97-0.99x between the first and last quarter and end 0.93x the reference's. Below SNR 7 the chains drift
  from their Fisher start: faint sources spread towards the prior (0.79 -> 1.12x its f0 width), sources at SNR 3-7
  are slowly lost. The f0-ordered labels make the strict drift test fail even where the physics is settled.
- **Crowded problems: the prior is the reference below SNR 3, SNR 3-7 is bracketed.** On narrow S = 8 and full
  S = 16 the reference never leaves its Fisher start for faint sources (f0 width 0.19x and 0.60x the prior's).
  Chains with every SNR < 7 source started from the prior (narrow S = 8: four injections, two seeds, 400 000
  iterations; full S = 16: four injections, two seeds, 200 000) find no faint source more often than chance, and
  at SNR 3-7 the two starts bracket the answer from opposite sides. At SNR >= 7 they confirm narrow S = 8's
  reference (1.11x), but on full S = 16 they end 1.2-1.7x wider than the reference and still widening, so its
  loud ratios are upper bounds.
- **Fisher is a cross-check only.** From SNR 40 up its median width per parameter is 0.89-0.96x the chains', a fair
  proxy from SNR 15 up; below SNR 5 it is none (f0 0.18x the chains'); on the full band at SNR 5-15 it is 2.2-2.5x
  wider in f0, partly because the reference is too narrow there. Its log fdot draws are unbounded for faint
  full-band sources and are not used.
- **Cost.** 400 000 iterations on 2 CPU cores: 4.3-5.4 h for narrow S = 4, 8.8-9.5 h for S = 8, 33-40 h for full
  S = 16. jexplore leaks memory per chunk, so long chains need generous memory (16 GB for four sources; the full
  S = 16 prior-started chains ran with 96 GB).

## Current files

| file | what |
|---|---|
| `data/eval-<problem id>.npz` | the six references (`canna.lisa.mcmc pack`): per injection 1024 pooled cold draws relabelled onto the truth and folded, 256 Fisher draws, the truth, R-hat, ESS and the chains' own SBC/TARP |
| `eval-<problem id>-corner.html`, `-validation.html` | the MCMC alone on the eval catalogue: corner pages with Fisher as the reference, and its SBC ranks, TARP and scores |
| `data/convergence-261113/summary.json`, `curves.npz` | the narrow S = 4 long-chain convergence study (`analyse_convergence.py`) |
| `data/convergence-261113/labelfree-*.json` | label-free readings (`label_free_widths.py`) of those long chains and of the prior-started chains of narrow S = 8 and full S = 16, against the reference and the reported flow (`-e32` = read on the e32 fork's draws) |

Problem ids: narrow S = 2 `a9a1998079`, S = 4 `cc4b9e9aa0`, S = 8 `ad9cb6a159`; full S = 4 `13653a71f0`, S = 8
`2ca7ffaadb`, S = 16 `6a6e86da4a`. The id hashes the problem's numeric fields and the eval seed, so a problem
change means new chains. The long-chain traces themselves are large and stay outside git (`outputs/mcmc/`).

The paper reads the packed references through `paper/scripts/mcmc_reference.py` (`numbers_mcmc.py`, `fig_mcmc.py`,
`fig_mcmc-posterior.py`) and the label-free JSONs through `numbers_mcmc.py`; each run's `corner.html` and
`validation.html` overlay the same draws.

## How they are made

- **Reference chains**: `canna.lisa.mcmc run`, one array task per (injection, seed) via
  [`slurm/lisa-mcmc.sbatch`](../../slurm/lisa-mcmc.sbatch), cached as `outputs/mcmc/joint-<problem id>/`; then
  `canna.lisa.mcmc pack` (CPU, a few minutes) writes `data/eval-<problem id>.npz`.
- **Long chains**: [`long_chain.py`](scripts/long_chain.py) runs the same sampler from iteration 0 at a fixed
  thinning ([`long_chain.sbatch`](scripts/long_chain.sbatch)); [`analyse_convergence.py`](scripts/analyse_convergence.py)
  scores burn-in and kept length ([`analyse.sbatch`](scripts/analyse.sbatch)).
- **Prior-started chains**: [`long_chain_prior.py`](scripts/long_chain_prior.py), `long_chain.py` with the walkers
  of the SNR < 3 sources started from the prior instead of the Fisher draws.
- **Label-free reading**: [`label_free_widths.py`](scripts/label_free_widths.py), post-processing only: per draw a
  loud source (SNR >= 7) takes the slot nearest its true f0, faint ones keep their rank and are compared with the
  f0-ordered prior, and "found" is the share of draws with a slot within 0.5 bins of the true f0.

**Method in short.** Each source is sampled in (ln f0, ln fdot, ln A, alpha, delta, psi, cos iota, phi0) under the
training prior (the catalogue Gaussian in log f0, log fdot, log A with f0 in the window, uniform angles and
cos iota, the Galactic sky), independently across sources, with one extra cut: log fdot at most 5 standard
deviations above its conditional mean. The sampler moves in a box (alpha, psi, phi0 on full periods centred on the
truth; ln fdot and ln A out to 8 conditional standard deviations). The likelihood is the window likelihood in double
precision, every source's response added into the window. The sampler is jexplore's parallel-tempered ensemble:
per iteration a temperature swap, then a differential-evolution or stretch move. Widths are 1.4826 x MAD of the
draws relabelled onto the truth and folded by the exact symmetries psi -> psi + pi and (psi, phi0) ->
(psi + pi/2, phi0 + pi) (`mcmc.fold`).

## Archive: the October page

[`mcmc.html`](mcmc.html) and the data it inlines are the record of the 2026-10-24 to 2026-10-31 chains on the
previous prior and the old narrow band (problems `ad8a87a575`, log Mc chart, and `244bc20185`): `data/draws.json`,
`data/convergence/`, `data/joint-*/chains.json` and `data/fisher-diagonal.json`, built by
[`collect_chains.py`](scripts/collect_chains.py), [`export_draws.py`](scripts/export_draws.py),
[`fisher_diagonal.py`](scripts/fisher_diagonal.py) and [`build_page.py`](scripts/build_page.py). The paper no
longer reads any of them. Rebuilding the page on the current references would need `export_draws.py` and
`fisher_diagonal.py` updated (log fdot in place of log Mc, the six problems) and rerun as jobs. The October
takeaways (converged, seed-stable chains in four windows; joint vs narrowband chains agreeing to a median 1.006)
are in this file's git history.
