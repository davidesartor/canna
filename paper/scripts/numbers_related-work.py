"""Numbers of appendix/related-work.tex: this work's problem size, and how crowded its prior windows are."""

import os

import numpy as np

import style

import jax
import jax.numpy as jnp
import jax.random as jr
from canna.lisa import config, network
from canna.lisa.physics import YEAR
from canna.lisa.problem import LisaGB, sample_f
from canna.lisa.simulator import sample_physical

assert not [k for k in os.environ if k.startswith("LISA_")], "stray LISA_* env vars"

WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight")
# Delmond et al. 2026: two sources at 1 yr, close to MCMC at 0.5 uHz, broad and biased at 0.2 uHz
DELMOND_SOURCES, DELMOND_TOBS = 2, YEAR
DELMOND_HARDEST_DF0, DELMOND_EASIEST_DF0 = 0.2e-6, 0.5e-6
CROWDING_WINDOWS, CHUNK = 4096, 256

configs = {band: config.load(f"{band}-B") for band in ("narrow", "full")}
problems = {band: LisaGB(**cfg.problem) for band, cfg in configs.items()}
full = problems["full"]
patch = configs["full"].network.get("patch", 16)
assert configs["narrow"].network.get("patch", 16) == patch


def closest_pair_bins(problem: LisaGB, key) -> np.ndarray:
    """Closest f0 separation per prior window, in the analysis's own 1/t_obs bins."""

    def one(key):
        key_f, key_p = jr.split(key)
        f = sample_f(problem, key_f)
        bins = sample_physical(problem, key_p, f)[:, 0] * problem.t_obs
        gaps = jnp.abs(bins[:, None] - bins[None, :])
        return jnp.min(jnp.where(jnp.eye(problem.n_sources, dtype=bool), jnp.inf, gaps))

    keys = jr.split(key, CROWDING_WINDOWS).reshape(-1, CHUNK)
    batched = jax.jit(jax.vmap(one))
    return np.concatenate([np.asarray(batched(chunk)) for chunk in keys])


hardest = DELMOND_HARDEST_DF0 * DELMOND_TOBS
closest = {band: closest_pair_bins(p, jr.key(0)) for band, p in problems.items()}
assert full.n_sources / DELMOND_SOURCES == 2

numbers = {
    "RelatedWorkSources": f"{full.n_sources}",
    "RelatedWorkSourcesWord": WORDS[full.n_sources],
    "RelatedWorkParameters": f"{full.n_sources * len(network.COORD_RANGES)}",
    "RelatedWorkTobs": f"{full.t_obs / YEAR:.0f}",
    "RelatedWorkPatch": f"{patch}",
    "RelatedWorkFzeroLow": f"{full.f0_range[0] * 1e3:.1f}",
    "RelatedWorkFzeroHigh": f"{full.f0_range[1] * 1e3:.0f}",
    "RelatedWorkSnrLow": f"{full.snr_range[0]:.0f}",
    "RelatedWorkSnrHigh": f"{full.snr_range[1]:.0f}",
    "RelatedWorkDelmondHardest": f"{hardest:.1f}",
    "RelatedWorkDelmondEasiest": f"{DELMOND_EASIEST_DF0 * DELMOND_TOBS:.1f}",
    "RelatedWorkCrowdingWindows": f"{CROWDING_WINDOWS}",
    "RelatedWorkClosestNarrow": f"{np.median(closest['narrow']):.1f}",
    "RelatedWorkClosestFull": f"{np.median(closest['full']):.0f}",
    "RelatedWorkCloserNarrow": f"{100 * np.mean(closest['narrow'] < hardest):.0f}",
    "RelatedWorkCloserFull": f"{100 * np.mean(closest['full'] < hardest):.0f}",
    "RelatedWorkSubBinNarrow": f"{100 * np.mean(closest['narrow'] < 1):.0f}",
}
for band, c in closest.items():
    print(band, "closest-pair quantiles", np.quantile(c, [0.1, 0.25, 0.5, 0.75, 0.9]))
style.save_numbers("related-work", numbers)
