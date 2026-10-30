"""One training window per band: the whitened data the network sees, and each source's response against the noise."""

import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

import style
from canna.lisa import config
from canna.lisa.injections import sort_by_snr
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
    """Centre frequency, sources (loudest first), data and clean signal of the window drawn from one seed."""
    key_f, key_p, key_n = jr.split(jr.key(seed), 3)
    f = sample_window(problem, key_f)
    p = sort_by_snr(problem, sample_physical(problem, key_p, f), f)
    return f, p, sample_observation(problem, key_n, p, f), clean_signal(problem, p, f)


def first_seed(problem: LisaGB, accept, start: int = 0) -> int:
    """The first seed from `start` whose window passes `accept`."""
    return next(
        seed
        for seed in range(start, start + 1000)
        if accept(*draw_window(problem, seed)[:2])
    )


style.setup()
problems = {
    name: LisaGB(**config.load(f"{name}-B").problem) for name in ("narrow", "full")
}


def narrow_overlap(f, p) -> bool:
    q = np.sort(np.asarray(p[:, 0]) * problems["narrow"].t_obs)
    return bool(np.diff(q).min() < problems["narrow"].response_points / 2)


narrow_seed = first_seed(problems["narrow"], narrow_overlap)
seeds = {
    "narrow": narrow_seed,
    "full": first_seed(
        problems["full"],
        lambda f, p: float(source_snrs(problems["full"], p, f).max()) >= VISIBLE_SNR,
        start=narrow_seed + 1,
    ),
}

fig, axes = plt.subplots(
    2,
    2,
    figsize=(style.FULL, 3.6),
    layout="constrained",
    sharex="col",
    height_ratios=[1, 1.15],
)
for column, (name, problem) in enumerate(problems.items()):
    ax_data, ax_sources = axes[:, column]
    f, p, observation, clean = draw_window(problem, seeds[name])
    bins = np.arange(problem.window_bins)
    guard = problem.response_points // 2
    for ax in (ax_data, ax_sources):
        ax.axvspan(0, guard, color=style.FILL, lw=0, zorder=0)
        ax.axvspan(
            problem.window_bins - guard,
            problem.window_bins,
            color=style.FILL,
            lw=0,
            zorder=0,
        )

    # top: the network's input channel Re A, and the clean signal whitened the same way
    data = np.asarray(preprocess(problem, observation, f))[:, 0]
    signal = np.asarray(preprocess(problem, clean, f))[:, 0]
    ax_data.step(bins, data, where="mid", color=style.MUTED, lw=0.5)
    ax_data.step(bins, signal, where="mid", color=style.INK, lw=0.9)
    ax_data.set(xlim=(0, problem.window_bins))
    ax_data.set_title(
        rf"{name}-B window, $f_w = {float(f) * 1e3:.3g}$ mHz, $K = {problem.window_bins}$"
    )
    if column == 0:
        ax_data.set_ylabel(r"whitened $\mathrm{Re}\,y_A$")

    # bottom: each source alone, in units of the noise's rms per bin
    idxs, segments = source_segments(problem, p, f)
    sigma = np.sqrt(np.asarray(window_power(problem, f))[:, 0])
    rho = np.asarray(source_snrs(problem, p, f))
    ax_sources.axhline(1.0, **style.GUIDE)
    peaks = []
    for s, color in enumerate(SOURCE_COLORS):
        at = np.asarray(idxs[s])
        amplitude = np.abs(np.asarray(segments[s, :, 0])) / sigma[at]
        ax_sources.step(at, amplitude, where="mid", color=color, lw=0.9)
        peak = int(np.argmax(amplitude))
        peaks.append((at[peak], amplitude[peak], rho[s]))
    # label each peak; a label close after an unraised one is raised 10 pt so the two don't collide
    previous, raised = -np.inf, True
    for x, y, snr in sorted(peaks):
        raised = not raised and x - previous < 0.12 * problem.window_bins
        style.direct_label(
            ax_sources,
            x,
            y,
            rf"$\rho = {snr:.2g}$",
            dx=2,
            dy=(3 if y > 1 else -6) + (10 if raised else 0),
        )
        previous = x
    # down to the faintest source's peak: a narrow window's sources often all sit far below the noise
    ax_sources.set(
        yscale="log",
        ylim=(min(1e-2, min(y for _, y, _ in peaks) / 10), None),
        xlabel="bin in window",
    )
    if column == 0:
        ax_sources.set_ylabel(r"$|\tilde h_{A}| / \sqrt{P_A}$ per source")
    style.direct_label(
        ax_sources, problem.window_bins, 1.0, "noise rms", dx=-2, dy=3, ha="right"
    )

for ax, letter in zip(axes.flat, "abcd"):
    style.panel_label(ax, letter)

style.save(fig, "training-window")
