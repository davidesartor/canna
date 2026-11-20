"""One narrow-band training window: each source's TDI-1.5 A response against the noise (2026-11-19: one column, was 2x2 with the full band and the whitened data)."""

import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

import style
from canna.lisa import config
from canna.lisa.injections import sort_by_f0
from canna.lisa.problem import LisaGB, window_power
from canna.lisa.simulator import (
    clean_signal,
    preprocess,
    sample_observation,
    sample_physical,
    sample_window,
    source_segments,
    source_snrs,
)

SOURCE_COLORS = [style.BLUE, style.ORANGE, style.AQUA, style.MAGENTA]
VISIBLE_SNR = 7.0


def draw_window(problem: LisaGB, seed: int):
    """Centre frequency, sources (by f0), data and clean signal of the window drawn from one seed."""
    key_f, key_p, key_n = jr.split(jr.key(seed), 3)
    f = sample_window(problem, key_f)
    p = sort_by_f0(sample_physical(problem, key_p, f))
    return f, p, sample_observation(problem, key_n, p, f), clean_signal(problem, p, f)


def first_seed(problem: LisaGB, accept, start: int = 0) -> int:
    """The first seed from `start` whose window passes `accept`."""
    return next(
        seed
        for seed in range(start, start + 1000)
        if accept(*draw_window(problem, seed)[:2])
    )


style.setup()
problem = LisaGB(**config.load("narrow-B").problem)


def narrow_overlap(f, p) -> bool:
    q = np.sort(np.asarray(p[:, 0]) * problem.t_obs)
    return bool(np.diff(q).min() < problem.response_points / 2)


seed = first_seed(problem, narrow_overlap)
fig, ax = plt.subplots(figsize=(style.COLUMN, 1.9), layout="constrained")
f, p, observation, clean = draw_window(problem, seed)
guard = problem.response_points // 2
ax.axvspan(0, guard, color=style.FILL, lw=0, zorder=0)
ax.axvspan(problem.window_bins - guard, problem.window_bins, color=style.FILL, lw=0, zorder=0)

# each source alone, in units of the noise's rms per bin
idxs, segments = source_segments(problem, p, f)
sigma = np.sqrt(np.asarray(window_power(problem, f))[:, 0])
rho = np.asarray(source_snrs(problem, p, f))
ax.axhline(1.0, **style.GUIDE)
peaks = []
saved = {"f_window": float(f), "snr": rho, "sigma": sigma}
for s, color in enumerate(SOURCE_COLORS):
    at = np.asarray(idxs[s])
    amplitude = np.abs(np.asarray(segments[s, :, 0])) / sigma[at]
    saved[f"bins_{s}"], saved[f"amplitude_{s}"] = at, amplitude
    ax.step(at, amplitude, where="mid", color=color, lw=0.9)
    peak = int(np.argmax(amplitude))
    peaks.append((at[peak], amplitude[peak], rho[s]))
# each SNR centred above its peak; a label close after an unraised one is raised so the two don't collide
previous, raised = -np.inf, True
for x, y, snr in sorted(peaks):
    raised = not raised and x - previous < 0.15 * problem.window_bins
    style.direct_label(
        ax, x, y, rf"$\rho = {snr:.2g}$", dx=0, dy=3 + (8 if raised else 0), ha="center", va="bottom",
        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"),
    )
    previous = x
ax.set(
    xlim=(0, problem.window_bins),
    yscale="log",
    ylim=(min(1e-2, min(y for _, y, _ in peaks) / 10), 8 * max(y for _, y, _ in peaks)),
    xlabel="frequency bin in window",
    ylabel=r"$|\tilde h_{A}| / \sqrt{P_A}$",
)
style.direct_label(ax, problem.window_bins, 1.0, "noise rms", dx=-2, dy=3, ha="right", va="bottom")

style.save(fig, "training-window")
# the plotted curves, in data/ (local only), so restyling never re-simulates
(style.PAPER / "data/figures").mkdir(parents=True, exist_ok=True)
np.savez(style.PAPER / "data/figures/training-window.npz", **saved)
