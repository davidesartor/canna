# Mock flows: stand-in posteriors with known defects (2026-10-30)

## Takeaways

- Seven mock posteriors run through the repo's own eval pipeline (src at 8310cb3), in the exact format of a real run's `eval/<tag>/`, so any eval page can be tested against a known answer.
- At HEAD's default prior every eval source is faint: `snr_hat` of the eval grid has median 0.05 and maximum 2.6, so all injections fall in the 0-15 SNR stratum and the 15-40, 40-100 and >=100 strata (and every "loud" column) are empty. The ladder is read at sub-threshold SNR only until `visible_snr` (window centres by the visible prior) is the default; rerun `mock.sbatch` then.
- `fisher` is no longer a calibrated reference there. Narrow (200 validation injections, 4 sources): f0 p=0.055, fdot p=0.78, f0+fdot p=0.26, f0-fdot p=0.86 pass, but lon, lat, the z± coordinates, `abs_zp/zm`, own SNR, TARP and the joint ranks reject (p<0.001), and the flow-over-Fisher f0 width is 1.79 (scorecard in68 0.75, in95 0.99). Faint sources are non-Gaussian in the flow coordinates (posterior close to the prior), which the linearised Gaussian cannot follow. Full band (100 injections): f0 p=0.29, fdot p=0.95, f0+fdot p=0.48; sky, z±, TARP and the joint ranks reject; f0 width ratio 4.0.
- `wide`, `narrow`, `biased` fail every rank test they should (p<0.001 on f0, the sky and z± coordinates, TARP, band depth); `biased` passes fdot (narrow p=0.58, full 0.86) and fails f0 and every combination containing it (`f0_plus_fdot`, `f0_minus_fdot`). `narrow` keeps the joint average rank (p=0.55 narrow, 0.53 full), so only the per-coordinate ranks, TARP, band depth and the log-likelihood rank catch it.
- `prior` is the clean pass here: narrow f0 p=0.69, lon 0.36, lat 0.18, TARP 0.24, joint average rank 0.72, band depth 0.81 (joint 0.026); it fails only fdot (p=0.009, narrow), `overlap` (p=0.003) and the joint log-likelihood rank (p=0.001). Full: lat p=0.016 and re_zp 0.044 are marginal, the log-likelihood rank and overlap (p=0.007) reject. Plain SBC and TARP pass it, as the literature says; only the data-space ranks catch it (see [../data-space-checks/](../data-space-checks/)).
- `phases-prior` is no longer separable from `fisher` on the SBC ranks (both reject the z± ranks, lon and lat), because `fisher` already rejects them at this SNR.
- `loud-floor` is identical to `fisher` here (same p-values and widths): its width cap starts at source SNR 18.4 and no source reaches it, so it tests nothing until loud sources exist.

## What the page shows

`mock-flows.html`: what each mock is, the width over the Fisher forecast per mock, the scorecard's per-SNR-band width and coverage, and the SBC gamma-test p-values per rank quantity (all slots or the loud snr_hat >= 100 stratum).

## Method

`scripts/mock_eval.py` replays `eval.main`, `scorecard.main` and `validate.sample` with the flow's sampler replaced by a mock, using the same candidates, noise keys and draw keys as a real run of the band's `<band>-B` config (seed 0). `eval_injections`, `fisher_samples`, `match_sources`, `score`, `validate.quantities` and `validate.summarise` are the repo's own. `src/` is untouched.

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

From the repo root, after `source slurm/unity.env` (CPU only, never on a login node):

    sbatch $CANNA_SBATCH_CPU reports/posterior-eval/mock-flows/scripts/mock.sbatch narrow --mocks fisher,wide,narrow,biased
    sbatch $CANNA_SBATCH_CPU reports/posterior-eval/mock-flows/scripts/mock.sbatch narrow --mocks prior,phases-prior,loud-floor

Each job takes ~8 min (4 or 3 mocks). Full ran with `--n-random 50 --n-validation 100` (smaller files), the same four and three mocks per job; the full default counts would also fit in 1 h per job. The sbatch rebuilds the page; `uv run --frozen python reports/posterior-eval/mock-flows/scripts/build_page.py` alone does it from `data/`.

## Data

`data/<band>/<mock>/` for band in `narrow`, `full` and the seven mocks above, ~54 MB each (draws.npz ~48 MB, scorecard.npz 3.2 MB narrow / 1.6 MB full, which carries the validation samples and scores). `validation.npz` and `validation_scores.npz` in the dirs are unused leftovers of the 2026-10-24 layout; nothing reads them and they can be deleted. Counts per mock are in its `mock.json`. Narrow: 64 eval-grid + 200 random scorecard injections, 200 validation injections. Full: 64 + 50 and 100.

## Format

Each `data/<band>/<mock>/` has the layout of a real `outputs/runs/<run>/eval/<tag>/` (a mock is told apart by `epoch: -1`, `ode_steps: 0`, `weights: mock`); N = injections, S = 4 sources.

- `eval.yaml`: `epoch: -1, weights: mock, ode_steps: 0, bf16: none, backend: cpu, n_random, n_validation`.
- `draws.npz` (as `eval.main`): `draws`, `matched`, `fisher` `<f8 (64,1024,4,8)`; `truth` `(64,4,8)`; `f`, `snr` `(64,)`; `grid` `(64,2)` (window index, SNR rank within it: 8 windows × 8 SNR ranks); `source_snr` `(64,4)`; `epoch` `<i8 ()`. `draws` are physical `[f0, fdot, A, lon, lat, psi, iota, phi0]`; `matched` is `match_sources` of them; `fisher` is the repo's truth-centred forecast, identical across mocks.
- `scorecard.npz` (as `scorecard.main`), three groups in one file:
  - stage: `epoch`, `ode_steps`, `n_random` `<i8 ()`; `t_obs` `<f8 ()`.
  - f0 rows, each `card_`-prefixed: `f0`, `x_f`, `snr`, `width`, `ideal`, `offset`, `rank` `<f8 (64+n_random)*4`; `f0_window` `(.,2)`; `in68`, `in95`, `is_eval` bool; `injection` `<i8`. Rows are source-major within an injection; the first 64 injections are the eval ones.
  - validation samples, packed (`canna.lisa.samples.load` unpacks): `f` `(N,)`; `f0_bins`, `overlap`, `source_snr` `(N,128,4)`; `log_likelihood`, `snr` `(N,128)`; `tarp_ref` `(N,4,8)`; `x` `(N,128,4,8)` (row 0 of the draw axis is the truth). Beside them `validate.summarise`'s scores: `rank_<q>` `<i8 (N,4)`, `joint_rank_<q>` `(N,)`, `snr_hat`, `joint_snr_hat`, `energy`, `crps`, `joint_energy`, `width_ratio`, `draws`.
- `mock.json`: description, parameters (`wide`, `narrow`, `floor_bins`, `floor_snr`), counts, and a `checks` block (width over Fisher per SNR band, scorecard per SNR band, gamma-test p, bias and middle-half share per rank set, all and loud).
