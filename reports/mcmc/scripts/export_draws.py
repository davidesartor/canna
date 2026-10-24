"""Thinned reference draws, Fisher draws, prior widths and convergence traces of every eval window, for the page: data/draws.json and data/convergence/traces.json."""

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.random as jr
import numpy as np
from einops import rearrange

from joint_mcmc import WINDOWS, chain_dir, make_window

from canna.lisa import config as run_config
from canna.lisa.injections import eval_injections, match_sources
from canna.lisa.mcmc import COLUMNS, MCMC_ROOT, cache_dir, fold, problem_id
from canna.lisa.problem import LisaGB
from canna.lisa.simulator import sample_physical

REPORT = Path(__file__).parents[1]
TRACES = MCMC_ROOT / "convergence" / "traces"
# draws are stored as integer hundredths of the source's narrowband MCMC width
QUANT = 100


def width(a: np.ndarray) -> np.ndarray:
    return 1.4826 * np.median(np.abs(a - np.median(a, axis=0)), axis=0)


def fold_sources(p: np.ndarray, truth: np.ndarray, t_obs: float) -> np.ndarray:
    """Draws [N, S, 8] -> COLUMNS coordinates [N, S, 8], each source folded onto its truth."""
    return np.stack(
        [fold(p[:, s], truth[s], t_obs) for s in range(truth.shape[0])], axis=1
    )


def quantise(offsets: np.ndarray, scale: np.ndarray) -> list:
    """Offsets [N, 8] -> per column, integer hundredths of scale [8]."""
    return np.round(offsets / scale * QUANT).astype(np.int64).T.tolist()


def thin(a: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return a[rng.choice(len(a), min(n, len(a)), replace=False)]


def rounded(a, digits: int = 5):
    return np.round(np.asarray(a, np.float64), digits).tolist()


def export_draws(
    problem: LisaGB, config, n_draws: int, n_joint: int, n_prior: int
) -> dict:
    t_obs = problem.t_obs
    injections = eval_injections(problem, config.seed)
    narrow_dir = cache_dir(problem, config.seed)
    joint_dir = chain_dir(problem, config.seed)
    rng = np.random.default_rng(0)
    windows = []
    for j in range(len(injections[0])):
        start = time.perf_counter()
        window = make_window(problem, injections, j)
        truth, f, n_sources = window.truth, window.f, window.truth.shape[0]
        true = fold_sources(truth[None], truth, t_obs)[0]

        # narrowband: one source free, the others subtracted at their truth
        narrow = {}
        for s in range(n_sources):
            for path in sorted(narrow_dir.glob(f"inj{j}_src{s}_seed*.npz")):
                z = np.load(path)
                seed = int(path.stem.rsplit("seed", 1)[1])
                narrow.setdefault(seed, {})[s] = fold(z["physical"], truth[s], t_obs)
        seeds = sorted(narrow)
        scale = np.stack(
            [
                width(np.concatenate([narrow[k][s] for k in seeds]))
                for s in range(n_sources)
            ]
        )

        # joint: every source free, relabelled onto the truth
        joint = {}
        for path in sorted(joint_dir.glob(f"window{j}_seed*.npz")):
            seed = int(path.stem.rsplit("seed", 1)[1])
            physical = thin(np.load(path)["physical"], n_joint, rng)
            joint[seed] = fold_sources(
                match_sources(problem, physical, truth, f), truth, t_obs
            )

        fisher = fold_sources(
            match_sources(problem, window.fisher, truth, f), truth, t_obs
        )

        # the prior of each loudness-sorted slot, folded onto that slot's truth
        prior = np.asarray(
            jax.vmap(sample_physical, in_axes=(None, 0, None))(
                problem, jr.split(jr.key(j), n_prior), f
            )
        )
        prior = fold_sources(prior, truth, t_obs)

        offsets = lambda d, s: d[:, s] - true[s]
        relative = lambda a: rounded(np.asarray(a) / scale, 4)
        pooled = [np.concatenate([narrow[k][s] for k in seeds]) for s in range(n_sources)]
        record = dict(
            j=j,
            f=float(f),
            window_snr=float(window.window_snr),
            snr=rounded(window.snr, 3),
            scale=scale.tolist(),
            prior_width=relative([width(prior[:, s]) for s in range(n_sources)]),
            narrow_width=[
                relative([width(narrow[k][s]) for s in range(n_sources)]) for k in seeds
            ],
            narrow_median=[
                relative([np.median(narrow[k][s] - true[s], axis=0) for s in range(n_sources)])
                for k in seeds
            ],
            joint_width=[
                relative([width(joint[k][:, s]) for s in range(n_sources)]) for k in sorted(joint)
            ],
            joint_median=[
                relative([np.median(offsets(joint[k], s), axis=0) for s in range(n_sources)])
                for k in sorted(joint)
            ],
            fisher_width=relative([width(fisher[:, s]) for s in range(n_sources)]),
            fisher_mean=relative([np.mean(offsets(fisher, s), axis=0) for s in range(n_sources)]),
            fisher_cov=[
                np.cov(offsets(fisher, s) / scale[s], rowvar=False).round(5).tolist()
                for s in range(n_sources)
            ],
            draws=dict(
                narrow=[
                    quantise(thin(pooled[s], n_draws, rng) - true[s], scale[s])
                    for s in range(n_sources)
                ],
                joint=[
                    [
                        quantise(thin(joint[k][:, s], n_draws, rng) - true[s], scale[s])
                        for s in range(n_sources)
                    ]
                    for k in sorted(joint)
                ],
                fisher=[
                    quantise(thin(fisher[:, s], n_draws, rng) - true[s], scale[s])
                    for s in range(n_sources)
                ],
            ),
        )
        windows.append(record)
        print(
            f"window {j}: SNR {window.window_snr:.0f}, {len(seeds)} narrowband and {len(joint)} joint seeds,"
            f" {time.perf_counter() - start:.0f} s",
            flush=True,
        )
    return dict(
        problem_id=problem_id(problem, config.seed),
        config=config.name,
        columns=COLUMNS,
        quant=QUANT,
        joint_windows=WINDOWS,
        windows=windows,
    )


def acf(x: np.ndarray, lags: int) -> np.ndarray:
    """Walker-averaged normalised autocorrelation of x [W, T] at lags 0..lags-1."""
    x = x - x.mean(axis=1, keepdims=True)
    n = x.shape[1]
    spectrum = np.fft.rfft(x, 2 * n, axis=1)
    c = np.fft.irfft(spectrum * np.conj(spectrum), axis=1)[:, :lags].mean(0)
    return c / c[0]


def export_traces(
    problem: LisaGB, config, points: int, band_points: int, lags: int, burn: int
) -> dict:
    directory = TRACES / f"joint-{problem_id(problem, config.seed)}"
    out = {}
    for j in WINDOWS:
        paths = sorted(directory.glob(f"window{j}_seed*.npz"))
        # 32 offsets from the folded truth, then log L: [W, T, 33] per seed
        series, meta = [], {}
        for path in paths:
            z = np.load(path)
            trace = z["trace"]
            series.append(
                np.concatenate(
                    [trace.reshape(*trace.shape[:2], -1), z["loglik"][..., None]], axis=-1
                )
            )
            meta = dict(
                iterations=int(z["iterations"]),
                thin=int(z["thin"]),
                nwalker=int(z["nwalker"]),
                snr=rounded(z["snr"], 3),
                window_snr=float(z["window_snr"]),
                seeds=[int(np.load(p)["chain_seed"]) for p in paths],
            )
        steps, first_kept = series[0].shape[1], burn // meta["thin"]

        # one centre and scale for every seed: the pooled kept draws, every 10th stored step
        kept = np.concatenate(
            [s[:, first_kept::10].reshape(-1, s.shape[-1]) for s in series]
        ).astype(np.float64)
        centre, scale = np.median(kept, axis=0), width(kept)
        unit = lambda a: np.round((a - centre) / scale * QUANT).astype(int)

        pick = np.linspace(0, steps - 1, points).round().astype(int)
        band = np.linspace(0, steps - 1, band_points).round().astype(int)
        seeds = [
            dict(
                walker=unit(s[0, pick]).T.tolist(),
                band=rearrange(
                    unit(np.percentile(s[:, band], [16, 50, 84], axis=0)), "q t d -> d q t"
                ).tolist(),
                acf=[rounded(acf(s[:, first_kept:, d], lags), 3) for d in range(s.shape[-1])],
            )
            for s in series
        ]
        print(f"traces window {j}: {len(seeds)} seeds", flush=True)
        out[str(j)] = meta | dict(
            centre=rounded(centre, 8),
            scale=rounded(scale, 8),
            iteration_walker=(pick * meta["thin"]).tolist(),
            iteration_band=(band * meta["thin"]).tolist(),
            chains=seeds,
        )
    return dict(columns=COLUMNS, quant=QUANT, burn=burn, lags=lags, windows=out)


def main() -> int:
    parser = argparse.ArgumentParser(prog="export_draws", description=__doc__)
    parser.add_argument(
        "--n_draws",
        type=int,
        default=400,
        help="draws per set, seed and source on the page",
    )
    parser.add_argument(
        "--n_joint",
        type=int,
        default=40000,
        help="joint draws per seed matched for the widths",
    )
    parser.add_argument(
        "--n_prior", type=int, default=4000, help="prior draws per window"
    )
    parser.add_argument(
        "--points", type=int, default=200, help="single-walker trace points"
    )
    parser.add_argument(
        "--band_points", type=int, default=100, help="ensemble 16/50/84 trace points"
    )
    parser.add_argument(
        "--lags", type=int, default=25, help="autocorrelation lags, in stored steps"
    )
    parser.add_argument(
        "--burn",
        type=int,
        default=100000,
        help="iterations dropped before the widths and ACF",
    )
    parser.add_argument("--skip", choices=["draws", "traces"])
    args = parser.parse_args()

    config = run_config.load()
    problem = LisaGB(**config.problem)
    data = REPORT / "data"
    if args.skip != "draws":
        record = export_draws(problem, config, args.n_draws, args.n_joint, args.n_prior)
        (data / "draws.json").write_text(
            json.dumps(record, separators=(",", ":")) + "\n"
        )
    if args.skip != "traces":
        record = export_traces(
            problem, config, args.points, args.band_points, args.lags, args.burn
        )
        (data / "convergence" / "traces.json").write_text(
            json.dumps(record, separators=(",", ":")) + "\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
