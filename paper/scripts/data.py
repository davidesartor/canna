"""Inputs for the figures and tables: scorecards, training logs, and the eval injections.

Everything lives under paper/data and is committed, so the paper builds from a clone,
without the cluster or the local outputs/ folder: training logs as per-epoch CSVs, the
scorecards, and the flow and Fisher draws of the eval injections the figures show
(import_data.py makes them from the run outputs). The eval injections themselves are rebuilt
with eval.py's own keys, checked against the stored truths, and cached (gitignored).
"""

import csv
import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from style import DATA  # noqa: E402

SNR_BANDS = [(0, 15), (15, 40), (40, 100), (100, np.inf)]
BAND_LABELS = ["< 15", "15-40", "40-100", r"$\geq$ 100"]

# training-log lineages: (job ids in order, what they are)
LOGS = {
    "XS-1M": [12486854, 13264195, 13378657],  # uniform clock, 500k then continued to 1M
    "XS-late": [13474448],
    "XS-late-768": [13719349],
    "XS-late-768-cool": [13719349, 13780234],  # the 768 run, then its 200k cooldown
    "XS-late-cool": [13474448, 13849421],  # XS-late, then its 200k cooldown
    "XS-aux10": [13395354],  # E1, stopped at epoch ~60
    "B-late": [13882121, 13882123, 13882124, 13882125],  # B1: four chained 24 h jobs
}


def scorecard(name: str) -> dict:
    with np.load(DATA / "scorecards" / f"{name}.npz") as d:
        return {k: d[k] for k in d.files}


def window_coordinate() -> np.ndarray:
    """Each scored source's f0 flow coordinate in [-1, 1], scorecard row order (F15)."""
    return np.load(DATA / "scorecards" / "coord.npy")[0]


# one unit of the f0 coordinate x_f is half the window interior: (64 - 16) / 2 bins on XS
XS_BINS_PER_UNIT = 24.0


def b_coordinate() -> dict:
    """B's scored sources, scorecard row order: f0, its coordinate x_f, the bins one unit of
    x_f spans there, and the bfloat16 cell of x_f in bins (outputs/scorecards/positions_B.py)."""
    with np.load(DATA / "scorecards" / "coord_B.npz") as d:
        return {k: d[k] for k in d.files}


def mcmc(j: int) -> dict:
    """compare_mcmc.py's summary of eval window j: per-source rows, coverage, loudest draws."""
    with np.load(DATA / "mcmc" / f"compare_window{j}.npz") as d:
        return {k: d[k] for k in d.files}


def training_log(run: str) -> dict:
    """Per-epoch medians of a lineage of jobs (data/logs/<job>.csv); later jobs win."""
    rows = {}
    for job in LOGS[run]:
        with open(DATA / "logs" / f"{job}.csv", newline="") as f:
            for r in csv.DictReader(f):
                rows[int(r["epoch"])] = (
                    int(r["total"]), float(r["flow"]), float(r["x"]), float(r["y"]),
                    float(r["aux_weight"]), float(r["lr_scale"]),
                    float(r["seconds"]) if r["seconds"] else np.nan,
                )
    epochs = np.array(sorted(rows))
    cols = np.array([rows[e] for e in epochs])
    return dict(epoch=epochs, total=cols[:, 0], flow=cols[:, 1], x=cols[:, 2], y=cols[:, 3],
                aux=cols[:, 4], lr=cols[:, 5], time=cols[:, 6])


def xs_problem():
    """The XS problem every XS run shares (configs/XS*.yaml, problem block)."""
    import yaml
    import canna.lisa as lisa
    from pathlib import Path

    cfg = yaml.safe_load(open(Path(lisa.__file__).parent / "configs" / "XS-late-768-cool.yaml"))
    return lisa.LisaGB(**cfg["problem"]), cfg


def eval_injections() -> dict:
    """The ten eval injections exactly as eval.py draws them (seed 0), cached."""
    cache = DATA / "eval" / "xs_eval_injections.npz"
    if cache.exists():
        with np.load(cache) as d:
            return {k: d[k] for k in d.files}

    import jax
    import jax.random as jr
    from canna.lisa.eval import N_CANDIDATES, N_QUANTILES

    problem, cfg = xs_problem()
    key_pick, key_window, key_noise = jr.split(jr.key(cfg["seed"]), 3)
    windows = jax.vmap(problem.sample_f)(jr.split(key_window, N_CANDIDATES))
    latents = jax.vmap(problem.sample_physical)(jr.split(key_pick, N_CANDIDATES), windows)
    snrs = np.asarray(jax.lax.map(lambda pc: problem.snr(*pc), (latents, windows)))
    quantiles = np.linspace(1.0 / N_QUANTILES, 1.0, N_QUANTILES)
    chosen = np.argsort(snrs)[np.round(quantiles * (N_CANDIDATES - 1)).astype(int)]
    noise_keys = jr.split(key_noise, N_QUANTILES)

    out = dict(
        window=np.asarray(windows[chosen]),
        truth=np.asarray(latents[chosen]),
        snr=snrs[chosen],
        quantile=quantiles,
        noise_key=np.asarray(jr.key_data(noise_keys)),
        all_snr=snrs,
        all_window=np.asarray(windows),
        all_truth=np.asarray(latents),
    )
    with np.load(DATA / "eval" / "fisher_eval_draws.npz") as d:
        assert np.allclose(d["truth"], out["truth"].reshape(d["truth"].shape)), \
            "rebuilt injections do not match eval.py's"
    np.savez(cache, **out)
    return out


def _injection(name: str, key: str, j: int) -> np.ndarray:
    with np.load(DATA / "eval" / name) as d:
        where = np.flatnonzero(d["injections"] == j)
        assert where.size, f"eval injection {j} is not in data/eval/{name} (import_data.py eval)"
        return d[key][where[0]]


def final_draws(j: int) -> np.ndarray:
    """The final model's 1024 draws of eval injection j, [draw, source, param], physical."""
    return _injection("final_eval_draws.npz", "draws", j)


def fisher_draws(j: int) -> np.ndarray:
    """The Fisher forecast draws of eval injection j, in the truth's labelling."""
    return _injection("fisher_eval_draws.npz", "fisher", j)


def matched_draws(problem, draws: np.ndarray, truth: np.ndarray, f: float) -> np.ndarray:
    """[draw, source, param] relabelled onto the true sources (eval.match_sources)."""
    from canna.lisa.eval import match_sources

    return match_sources(problem, draws, truth, f)
