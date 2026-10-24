# MCMC reference: can the chains the flow is scored against be trusted? (2026-10-24)

## Takeaways

- **Every reference chain is converged and seed-stable.** The 8 joint chains (windows 9, 5, 2, 7 ×
  2 seeds) have R̂ ≤ 1.00004 and ESS ≥ 138 716; the 80 narrowband chains R̂ ≤ 1.00015 and
  ESS ≥ 72 334. Seed-to-seed widths agree within 3% for 95% of joint and 99.4% of narrowband
  source-parameters (worst 17% and 3.8%).
- **Holding the other sources at their truth barely matters.** Joint ÷ narrowband width has a
  median of 1.006, and 80% of the 128 joint source-parameters lie in 0.9–1.1. The exceptions sit
  in six sources, worst in window 2: its SNR-7 source's f₀ (3.15×) and its SNR-314 source's
  longitude (2.51×), where the joint chain hops between sky modes.
- **The paper's 100 000 burn / 300 000 keep / thin 100 passes every kept-draw check, but only
  windows 9 and 7 pass the strict burn-in test.** Over the kept draws ESS ≥ 572 253,
  |R̂ − 1| ≤ 6.4e-4 and median drift ≤ 0.070 half-widths (tolerance 0.1). Windows 9 and 7 settle
  by 60 000 and 20 000 iterations; windows 5 and 2 never settle within 400 000, so the criterion
  recommends no setting. fml's 20 000 / 60 000 / 20 passes in window 7 only.
- **τ is an upper bound.** τ_max is 26–34 iterations, within 2× the trace's thinning of 20, and
  the slowest parameter is always the cold log L. The ESS values are lower bounds; no conclusion
  depends on resolving τ.
- **The Fisher forecast is a good start point, not a reference.** Above SNR 40, Fisher ÷ MCMC
  lies in 0.8–1.25 for 88% of narrowband (27 sources) and 90% of joint (9 sources)
  source-parameters, without log Mc. It misses by more than 2× for four sources at SNR 48–137,
  along the log A–sin ι and ψ–φ₀ degeneracies (up to 4.0×). Below SNR 40 it spans 0.35–2.63;
  log Mc, which the prior dominates, 0.60–1.48. The Fisher f₀ width is a median 1.06× the ideal
  √3/(πρ) bins (at most 1.55×).
- **The averaged Hessian is not always positive along Mc.** In 8 of the 80 eval sources of both
  bands the 32-realisation average has a non-positive Mc diagonal; the prior precision fixes it,
  and `fisher_samples`' eigenvalue floor at the prior binds in 9 of the 20 injections.
- **Cost.** A 400 000-iteration joint chain takes 4.4–5.1 h on 2 CPUs and needs 16 GB: jexplore
  leaks about 64 MB per 2000-iteration chunk, so 8 GB runs out of memory.

## What the page shows

[`mcmc.html`](mcmc.html), interactive and standalone; every number on it is
computed in the page from the data it inlines. The sidebar picks a window (blue: the four with
joint and long chains), a source (loudest first), a parameter, which drives every card, and the
parameter it is paired with in the corner. The tiles summarise that choice: the window and
source SNRs, the chains' R̂ and ESS, joint ÷ narrowband and Fisher ÷ MCMC widths of the
parameter, and the window's burn test. All state lives in the URL hash (`#window=9&param=3&…`,
also `theme=light|dark`), so a view can be linked.

1. **Do the joint and narrowband chains agree?** A corner of the eight coordinates (offsets from
   the truth, in narrowband widths) with the Fisher 1σ/2σ ellipses, one enlarged pair or
   marginal, and each set's width per coordinate against the narrowband chain, with the prior's
   width below. Sets (joint per seed, narrowband, Fisher) toggle; a corner cell picks the pair.
2. **Are the long chains past burn-in?** For the four long-chain windows: the walkers' 16–84% band
   or one walker per seed over 400 000 iterations, for the parameter or the cold log L, with the
   autocorrelation and its τ; then median and width drift against burn-in and ESS and R̂ − 1
   against kept length, with the tolerances drawn and a button to the worst series. The table
   scores the paper's and fml's settings per window.
3. **How far does the Fisher forecast follow the MCMC?** Fisher ÷ MCMC width against source SNR
   for every eval source, against the joint or the narrowband chain, with the ideal f₀ width;
   then each source's averaged Hessian diagonal over the prior's precision, for the narrow or
   the full band, and a table of each injection's non-positive diagonals and floored
   eigenvalues.
4. **What this means**: the takeaways, recomputed from the inlined data.

## Method

**Narrowband chains** are the single-source reference of `canna.lisa.mcmc`: one source free,
the others subtracted at their truth, 2 seeds per source of the 10 narrow-B eval windows,
cached under `outputs/mcmc/<problem id>/`.

**Joint chains** ([`joint_mcmc.py`](scripts/joint_mcmc.py)) sample all 4 × 8 parameters of one
window in one ensemble against the same noise the flow saw (`eval_injections` at the config's
eval seed). The prior is the training one from `config.load`: uniform log f₀ on the window, the
chirp-mass density, the galactic sky mixture, uniform sin ι and 2π boxes for the angles, and
the log-uniform per-source SNR prior as a log A box plus an SNR cut. The likelihood is
−Σ|o − h|²/P over the window from one `source_segments` call. The sampler is jexplore's
parallel-tempered ensemble: 64 walkers × 6 temperatures (1–10), a temperature swap then DE or
stretch per iteration, started from the window's Fisher draws; 100 000 burn, 300 000 kept,
thin 100. Chains go to `outputs/mcmc/joint-<problem id>/`, keyed on a hash of the problem's
numeric fields and the eval seed; the priors enter only by name, so change a prior's shape and
rerun with `--force`. [`collect_chains.py`](scripts/collect_chains.py) saves every chain's R̂,
ESS, τ, wall time and widths to `data/joint-<id>/chains.json`.

**Comparisons.** Each set is relabelled onto the injected sources (`match_sources`) and folded
by the likelihood's exact symmetries ψ → ψ + π and (ψ, φ₀) → (ψ + π/2, φ₀ + π) (`mcmc.fold`).
Widths are 1.4826 × MAD.

**Burn-in.** [`long_chain.py`](scripts/long_chain.py) runs the joint sampler unchanged for
400 000 iterations from iteration 0, 4 seeds per window, thinned at 20, folded and in f₀ bins.
[`analyse_convergence.py`](scripts/analyse_convergence.py) computes per parameter (32
coordinates plus the cold log L) τ (emcee's estimator, Sokal window c = 5), ESS = walkers ×
kept / τ, split R̂ over 8 half-ensembles, and the drift of each marginal's median (in half 68%
widths) and 68% width against the other seeds' pooled second halves. A window passes at burn b
when b ≥ 10 τ_max and every 20 000-iteration segment from b on drifts less than 0.1; the keep is
the shortest with ESS ≥ 2000 per seed, R̂ < 1.01 and the same drift tolerance. ESS 2000 puts the
Monte Carlo error of a width near 2.6% per seed, below the flow/MCMC ratios being resolved.

**Fisher.** The Fisher draws are `injections.fisher_samples`: the Hessian of −log L at the truth
averaged over 32 noise draws, plus the prior precision, eigenvalues floored at the prior;
[`fisher_diagonal.py`](scripts/fisher_diagonal.py) records its diagonal, the floored eigenvalues
and the f₀ width against √3/(πρ). Agreement means a ratio in 0.8–1.25; faint means SNR < 40.

[`export_draws.py`](scripts/export_draws.py) thins every set to 400 draws per seed and source,
stored as integer hundredths of the narrowband width, plus prior draws and the long-chain
traces; [`build_page.py`](scripts/build_page.py) writes `mcmc.html` with `data/`
inline.

## How to rebuild

From the repo root, after `source slurm/unity.env` (or `trex.env`). Everything but the page is
JAX, so it runs as jobs.

```bash
# narrowband chains, if not cached
sbatch $CANNA_SBATCH_CPU --array=… slurm/lisa-mcmc.sbatch narrow-B
# joint chains (8 tasks, ~4-6 h each), then their health and widths
sbatch $CANNA_SBATCH_CPU reports/mcmc/scripts/joint_mcmc.sbatch narrow-B
sbatch $CANNA_SBATCH_CPU reports/mcmc/scripts/collect.sbatch narrow-B
# long chains (16 tasks, 16 GB, ~5 h each, 4 at a time), then the convergence analysis
sbatch $CANNA_SBATCH_CPU reports/mcmc/scripts/long_chain.sbatch narrow-B
sbatch $CANNA_SBATCH_CPU --dependency=afterok:<long-chain job> reports/mcmc/scripts/analyse.sbatch narrow-B
# draws and traces, then the page
sbatch $CANNA_SBATCH_CPU reports/mcmc/scripts/export.sbatch narrow-B
# Fisher diagonal of both bands (~4 min), then the page
sbatch $CANNA_SBATCH_CPU reports/mcmc/scripts/fisher.sbatch
```

Each sbatch header has a smoke-test line. The page alone is numpy and runs on a login node:

```bash
bash -c 'ulimit -v 4000000; uv run --frozen python reports/mcmc/scripts/build_page.py'
```

It prints every number the takeaways quote.

## Data

- `data/draws.json`: per eval window, the folded draws (joint per seed, narrowband per seed,
  Fisher, prior), their widths and medians, and the source SNRs.
- `data/joint-<problem id>/chains.json`: every narrowband and joint chain's R̂, ESS, τ, seconds
  and widths.
- `data/convergence/summary.json`, `curves.npz`: the criterion, per-window burn/keep/thin, the
  scored paper and fml settings, and every drift, ESS and R̂ curve.
- `data/convergence/traces.json`: thinned long-chain traces and autocorrelations.
- `data/fisher-diagonal.json`: the precision diagonal, prior diagonal, floored eigenvalues and
  f₀ widths of every eval injection of both bands.

The paper's MCMC, convergence and Fisher numbers and figures read these files.
