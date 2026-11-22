"""The six joint references in use (reports/mcmc/data/eval-<problem id>.npz, `mcmc pack`): per eval injection 1024 pooled
cold draws of the joint chains, 256 Fisher draws, the truth and the chains' diagnostics, sources by f0. Used by
numbers_mcmc.py.

Widths are 1.4826 MAD of the draws folded onto the truth's (psi, phi0) branch (mcmc.fold), f0 in bins. The drift column
is taken in fdot itself, not ln fdot, for the widths: Fisher draws can put fdot below zero."""

from functools import cache

import numpy as np

import style
from canna.lisa.mcmc import COLUMNS, fold
from canna.lisa.samples import load as load_samples

# (band, S) -> problem id, as the source-count runs' configs hash (outputs/runs/261103-vs-mcmc.py)
PROBLEMS = {
    ("narrow", 2): "a9a1998079",
    ("narrow", 4): "cc4b9e9aa0",
    ("narrow", 8): "ad9cb6a159",
    ("full", 4): "13653a71f0",
    ("full", 8): "2ca7ffaadb",
    ("full", 16): "6a6e86da4a",
}
T_OBS = 63115200.0
WIDTH_COLUMNS = ["f0 [bins]", "fdot", *COLUMNS[2:]]


def mad(x: np.ndarray) -> np.ndarray:
    return 1.4826 * np.median(np.abs(x - np.median(x, 0)), 0)


@cache
def reference(band: str, n: int) -> dict:
    return load_samples(style.REPO / f"reports/mcmc/data/eval-{PROBLEMS[band, n]}.npz")


def folded(draws: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Draws [N, 8] of one source as offsets from the truth in COLUMNS (f0 in bins), the drift column in fdot."""
    out = fold(draws, truth, T_OBS) - fold(truth[None], truth, T_OBS)[0]
    out[:, 1] = draws[:, 1] - truth[1]
    return out


@cache
def sources(band: str, n: int) -> list[dict]:
    """Per source of every injection with chains: SNR, MCMC and Fisher widths in WIDTH_COLUMNS, R-hat, ESS."""
    d = reference(band, n)
    rows = []
    for j in range(d["mcmc"].shape[0]):
        if not np.isfinite(d["mcmc"][j]).all():
            continue
        for s in range(n):
            truth = d["truth"][j, s]
            wm = mad(folded(d["mcmc"][j, :, s], truth))
            wf = mad(folded(d["fisher"][j, :, s], truth))
            rows.append(dict(injection=j, source=s, snr=float(d["source_snr"][j, s]), mcmc=wm, fisher=wf, ratio=wf / wm))
    return rows
