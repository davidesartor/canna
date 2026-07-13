"""WDM pixel percentiles vs frequency bin (time folded), clean signal vs noisy datastream.

Per frequency bin the arcsinh-compressed WDM pixels are ~iid over time, so we fold
time+draws and show the 0/25/50/75/100 percentile bands per A/E/T channel, one colour
per injected-source count. Top row is the clean signal WDM, bottom the noisy datastream
(signal+noise) WDM -- the latter is what the network actually conditions on.
"""

import os

import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from canna import lisa

from _helpers import SOURCE_COUNTS, map_chunks, problem_name, save_figure, save_text

PCTLS = [0, 25, 50, 75, 100]
N_DRAWS = int(os.environ.get("WDM_DRAWS", 48))
CHUNK = int(os.environ.get("WDM_CHUNK", 4))
SEED = int(os.environ.get("SEED", 0))


def _percentiles(y: np.ndarray) -> np.ndarray:
    """(len(PCTLS), F, C) percentiles, folding draws and time."""
    y = np.asarray(y).reshape(-1, y.shape[-2], y.shape[-1])  # (draws*T, F, C)
    return np.percentile(y, PCTLS, axis=0)


def _wdm_percentiles(n_sources: int, key):
    """Percentile bands for the noisy datastream and clean signal WDM images."""
    keys = jr.split(key, N_DRAWS)
    y, y_clean = map_chunks(
        lambda k: lisa.get_physics_sample(k, n_sources)[4:6], keys, CHUNK
    )  # each (draws, T, F, C)
    return {"datastream (noisy)": _percentiles(y), "signal (clean)": _percentiles(y_clean)}


def test_wdm_pixel_stats():
    keys = jr.split(jr.key(SEED), len(SOURCE_COUNTS))
    bands = {n: _wdm_percentiles(n, k) for n, k in zip(SOURCE_COUNTS, keys)}
    for kinds in bands.values():
        for b in kinds.values():
            assert np.isfinite(b).all()

    kinds = ["signal (clean)", "datastream (noisy)"]
    any_band = next(iter(bands.values()))["signal (clean)"]
    channels = any_band.shape[-1]
    freq = np.arange(any_band.shape[1])
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(SOURCE_COUNTS)))

    fig, axes = plt.subplots(
        len(kinds), channels, figsize=(6 * channels, 4.5 * len(kinds)), squeeze=False, sharex=True
    )
    for r, kind in enumerate(kinds):
        for ch in range(channels):
            ax = axes[r, ch]
            for (n, kd), c in zip(bands.items(), colors):
                p0, p25, p50, p75, p100 = kd[kind][:, :, ch]
                ax.fill_between(freq, p0, p100, color=c, alpha=0.12)
                ax.fill_between(freq, p25, p75, color=c, alpha=0.30)
                ax.plot(freq, p50, color=c, lw=1.3, label=f"{n} src")
            ax.set_title(f"{kind} · {lisa.CHANNEL_NAMES[ch]}")
            if r == len(kinds) - 1:
                ax.set_xlabel("frequency bin")
            ax.legend(fontsize=8)
        axes[r, 0].set_ylabel("arcsinh WDM pixel value")
    fig.suptitle(f"WDM pixel percentiles vs freq — {problem_name()}")
    save_figure(fig, "wdm_pixel_stats")

    lines = [f"WDM pixel percentiles — {problem_name()} ({N_DRAWS} draws)"]
    for n, kd in bands.items():
        for kind in kinds:
            for ch in range(channels):
                p = kd[kind][:, :, ch]  # (Q, F): summarize the band across freq bins
                lines.append(
                    f"{n} src {kind:>18} {lisa.CHANNEL_NAMES[ch]}: "
                    f"med(q50)={np.median(p[2]):.3g}  "
                    f"IQR~{np.median(p[3] - p[1]):.3g}  "
                    f"range~{np.median(p[4] - p[0]):.3g}"
                )
    save_text("wdm_pixel_stats", "\n".join(lines))
