"""Optimal-SNR distribution over prior draws, overlaid for 1/2/4/8 injected sources."""

import os

import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from canna import lisa

from _helpers import SOURCE_COUNTS, map_chunks, problem_name, save_figure, save_text

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
N_DRAWS = int(os.environ.get("SNR_DRAWS", 512))
CHUNK = int(os.environ.get("SNR_CHUNK", 32))
SEED = int(os.environ.get("SEED", 0))


def _snr_samples(n_sources: int, key) -> np.ndarray:
    """Combined optimal SNR for N_DRAWS datastreams of ``n_sources`` binaries."""
    u = jr.uniform(key, (N_DRAWS, n_sources, 8))
    params = lisa.prior_inverse_cdf(u)
    return np.asarray(map_chunks(lisa.optimal_snr, params, CHUNK))


def test_snr_distribution():
    keys = jr.split(jr.key(SEED), len(SOURCE_COUNTS))
    snrs = {n: _snr_samples(n, k) for n, k in zip(SOURCE_COUNTS, keys)}
    for snr in snrs.values():
        assert np.isfinite(snr).all() and (snr > 0.0).all()

    lo = min(snr.min() for snr in snrs.values())
    hi = max(snr.max() for snr in snrs.values())
    bins = np.logspace(np.log10(lo), np.log10(hi), 60)
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(SOURCE_COUNTS)))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (n, snr), c in zip(snrs.items(), colors):
        ax.hist(snr, bins=bins, histtype="step", lw=1.8, color=c, label=f"{n} src")
    ax.axvline(1.0, color="grey", ls="--", lw=0.8)
    ax.set(xscale="log", xlabel="optimal SNR", ylabel="count")
    ax.set_title(f"SNR — {problem_name()} ({N_DRAWS} draws / source count)")
    ax.legend()
    save_figure(fig, "snr")

    lines = [f"SNR — {problem_name()} ({N_DRAWS} draws / source count)"]
    for n, snr in snrs.items():
        quant = np.quantile(snr, QUANTILES)
        qstr = "  ".join(f"q{int(q * 100):02d}={v:.3g}" for q, v in zip(QUANTILES, quant))
        lines.append(f"{n} src: min={snr.min():.3g} max={snr.max():.3g}  {qstr}")
    save_text("snr", "\n".join(lines))
