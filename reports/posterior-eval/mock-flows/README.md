# Mock flows: stand-in posteriors with known defects (2026-11-02)

## Takeaways

- Seven mock posteriors run through the repo's own eval pipeline (working-tree src, narrow-B config: 1-3 mHz, 4 sources, seed 0), in the exact format of a real run's `eval/<tag>/`, so any eval page can be tested against a known answer. Sources are f0-sorted in draws and truth, as the flow emits them. Narrow band only; no full-band mocks were rebuilt.
- Since the visible-window prior the eval grid reaches loud sources: 128 injections, 1312 scorecard sources (1111 at SNR 0-15, 185 at 15-40, 15 at 40-100, 1 above 100). The validation stream (200 injections) still has no source above SNR 100, so every "loud" column is empty.
- `fisher` is a calibrated reference only in f0 and fdot. Over all 800 validation slots the p-values pass for f0 (0.141), fdot (0.422), `f0_plus_fdot` (0.255), `f0_minus_fdot` (0.251) and the joint average rank (0.117); lat, im_zp, re_zm, im_zm, `abs_zp/zm`, overlap, own SNR, TARP and band depth reject (p<0.001). Flow-over-Fisher f0 width 1.04, scorecard in68 0.66 and in95 0.93; the width over ideal grows with SNR (1.07 at 0-15, 1.47 at 15-40, 3.54 at 40-100).
- `wide` (f0 width 3.10 x Fisher, in68 1.00) and `narrow` (0.52, in68 0.40, in95 0.65) fail every per-slot rank test and TARP (p<0.001). `narrow` keeps the joint average rank (p=0.006), so only the per-coordinate ranks, TARP, band depth and the log-likelihood rank catch it; `wide` rejects it too (p<0.001).
- `biased` (f0 +1 sigma) fails f0 (p<0.001) and keeps fdot (0.422) and the f0 sums (0.255, 0.251); in68 falls to 0.47.
- `prior` is the clean pass: f0 p=0.689, lon 0.356, lat 0.101, TARP 0.546, average rank 0.916 (joint 0.775, TARP 0.853), band depth 0.294. It fails only overlap (p<0.001), fdot (0.03, marginal), band depth joint (0.026) and the joint log-likelihood rank (p<0.001). Only the data-space ranks catch it (see [../data-space-checks/](../data-space-checks/)).
- `phases-prior` is not separable from `fisher` on the SBC ranks (both reject the z± ranks and the sky), because `fisher` already rejects them; it differs in the joint average rank (0.606 vs 0.117) and re_zp (<0.001 vs 0.049).
- `loud-floor` rejects the same quantities as `fisher` (f0 p=0.622 against 0.141): its width cap starts at source SNR 18.4, and its in68 0.68 / in95 0.94 barely differ from `fisher`, so it tests the floor only through the 16 sources above SNR 40.

## What the page shows

`mock-flows.html`: what each mock is, the width over the Fisher forecast per mock, the scorecard's per-SNR-band width and coverage, and the SBC gamma-test p-values per rank quantity (all slots or the loud snr_hat >= 100 stratum).

## Method

`scripts/mock_eval.py` replays `eval.main`, `scorecard.main` and `validate.sample` with the flow's sampler replaced by a mock, using the same candidates, noise keys and draw keys as a real run of the band's `<band>-B` config (seed 0), with sources sorted by f0 in the truth and in every draw. `eval_injections`, `fisher_samples`, `sort_by_f0`, `score`, `validate.quantities` and `validate.summarise` are the repo's own. `src/` is untouched.

The mock base is a linearised Gaussian posterior in flow coordinates (flat, near-Gaussian): for each observation, `C = (F + P)^-1` with `F` the exact expected Fisher at the truth (jacobian of `clean_signal` through `flow_to_physical`) and `P` the Gaussian prior precision (the flow-coordinate covariance of the eval's 1024 prior candidates). The centre is the linearised MAP of the actual noisy data, `x0 + C (score + P (mu - x0))`. This is exactly calibrated in a linear-Gaussian model; it is approximate here because of non-Gaussian priors (galactic sky mixture, uniform x_f, the SNR >= 7 truncation), low-SNR non-Gaussianity and sky multimodality, and because the Fisher is taken at the truth (a mock may use the truth; a flow may not). Not to be confused with the repo's `fisher` key in `draws.npz` (`fisher_samples`), which is centred on the truth and so is not a calibrated reference; it is only the width yardstick.

The ladder (one draw per key, batching by `jax.vmap`; extra streams `fold_in(k, 100)` and `fold_in(k, 101)`):

1. `fisher`: the base, draws from N(centre, C).
2. `wide`: every width x3 (same centre and correlations).
3. `narrow`: every width x0.5.
4. `biased`: f0 centre shifted +1 marginal sigma on every source.
5. `prior`: prior draws for the same f, ignoring the data (the flow's own base draws; an identity flow).
6. `phases-prior`: `fisher` with the phase of z+ and z- redrawn uniform on the circle, radii kept (research log F52).
7. `loud-floor`: per-coordinate width scaled by `max(1, min(rho_s / 18.4, prior_sd / sigma))`, where `rho_s` is the source's SNR at the centre. The floor comes from B's loud f0 floor of ~0.03 bins (research log F44, F48): the monochromatic bound `sqrt(3) / (pi rho)` reaches 0.03 bins at rho = 18.4, so widths stop shrinking above that, giving ~55x at SNR 1000, between the loudest source's sky (~20x MCMC, F56) and phi- (~100x, F58). The cap at the prior width keeps prior-dominated coordinates sane.

Validation injections are the repo's stream (`split(fold_in(key(seed), 1), n)`, 127 draws, `tarp_ref` from `key_tarp`), so TARP and the strata match a real run's.

## How to rebuild

From the repo root, `PYTHON=<venv python>` exported (CPU only, never on a login node; the sbatch runs the working tree's `src`):

    sbatch $CANNA_SBATCH_CPU reports/posterior-eval/mock-flows/scripts/mock.sbatch narrow --mocks fisher,wide,narrow,biased
    sbatch $CANNA_SBATCH_CPU reports/posterior-eval/mock-flows/scripts/mock.sbatch narrow --mocks prior,phases-prior,loud-floor

Each job took ~14 min (jobs 66528333 and 66528334). The sbatch rebuilds the page; `python reports/posterior-eval/mock-flows/scripts/build_page.py` alone does it from `data/`.

## Data

`data/narrow/<mock>/` for the seven mocks above, ~20 MB each (gitignored; draws are packed, see Format). Counts per mock are in its `mock.json`: 128 eval injections + 200 random scorecard injections, 200 validation injections.

## Format

Each `data/<band>/<mock>/` has the layout of a real `outputs/runs/<run>/eval/<tag>/` (a mock is told apart by `epoch: -1`, `ode_steps: 0`, `weights: mock`); N = injections, S = 4 sources.

- `eval.yaml`: `epoch: -1, weights: mock, ode_steps: 0, bf16: none, backend: cpu, n_random, n_validation`.
- `draws.npz` (as `eval.main`, packed by `canna.lisa.samples.pack_draws`; `samples.load` unpacks): `draws` and `fisher` `(128,1024,4,8)`; `truth` `(128,4,8)`; `f`, `snr` `(128,)`; `source_snr` `(128,4)`; `epoch`. `draws` are physical `[f0, fdot, A, lon, lat, psi, iota, phi0]`, sources sorted by f0 like `truth`; `fisher` is the repo's truth-centred forecast, identical across mocks.
- `scorecard.npz` (as `scorecard.main`), three groups in one file:
  - stage: `epoch`, `ode_steps`, `n_random` `<i8 ()`; `t_obs` `<f8 ()`.
  - f0 rows, each `card_`-prefixed: `f0`, `x_f`, `snr`, `width`, `ideal`, `offset`, `rank` `<f8 (128+n_random)*4`; `f0_window` `(.,2)`; `in68`, `in95`, `is_eval` bool; `injection` `<i8`. Rows are source-major within an injection; the first 128 injections are the eval ones.
  - validation samples, packed (`canna.lisa.samples.load` unpacks): `f` `(N,)`; `f0_bins`, `overlap`, `source_snr` `(N,128,4)`; `log_likelihood`, `snr` `(N,128)`; `tarp_ref` `(N,4,8)`; `x` `(N,128,4,8)` (row 0 of the draw axis is the truth). Beside them `validate.summarise`'s scores: `rank_<q>` `<i8 (N,4)`, `joint_rank_<q>` `(N,)`, `snr_hat`, `joint_snr_hat`, `energy`, `crps`, `joint_energy`, `width_ratio`, `draws`.
- `mock.json`: description, parameters (`wide`, `narrow`, `floor_bins`, `floor_snr`), counts, and a `checks` block (width over Fisher per SNR band, scorecard per SNR band, gamma-test p, bias and middle-half share per rank set, all and loud).
