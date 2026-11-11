"""figures/prior-corner.pdf: the per-source prior of each band, (log10 A, log10 f0, log10 fdot, RA, Dec), as a corner plot.

Draws come from the samplers the simulator uses (canna.lisa.physics.sample_sources, sample_sky), f0 truncated to each
band's f0_range; they are saved to data/prior-corner/<band>.npz beside the figure (data/ stays local), and a rerun
with the files present only restyles. Contours enclose 39% and 86% (1 and 2 sigma in 2-D); no points are drawn.
"""

import corner
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import style
from style import plt

from canna.lisa import config
from canna.lisa.physics import sample_sky, sample_sources
from canna.lisa.problem import LisaGB

DRAWS = 200_000
BANDS = {"full": style.BLUE, "narrow": style.ORANGE}  # full first: the narrow band's lines sit on top
LEVELS = (1 - np.exp(-0.5), 1 - np.exp(-2.0))
LABELS = [r"$\log_{10}\mathcal{A}$", r"$\log_{10}f_0$ [Hz]", r"$\log_{10}\dot f$ [Hz/s]", r"$\alpha$ [deg]", r"$\delta$ [deg]"]
DATA = style.PAPER / "data/prior-corner"


def draws(band: str, seed: int) -> np.ndarray:
    """DRAWS prior sources of the band as [log10 A, log10 f0, log10 fdot, RA deg, Dec deg], cached in DATA."""
    path = DATA / f"{band}.npz"
    if path.exists():
        return np.load(path)["x"]
    problem = LisaGB(**config.load(f"{band}-B").problem)
    key_source, key_sky = jr.split(jr.key(seed))
    f0, fdot, amp = np.asarray(sample_sources(key_source, *map(jnp.asarray, problem.f0_range), DRAWS)).T
    ra, dec = np.asarray(jax.vmap(sample_sky)(jr.split(key_sky, DRAWS))).T
    x = np.stack([np.log10(amp), np.log10(f0), np.log10(fdot), np.degrees(ra) % 360, np.degrees(dec)], axis=1)
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x.astype(np.float32), f0_range=np.asarray(problem.f0_range), columns=np.array(LABELS))
    return x


style.setup()
samples = {band: draws(band, seed) for seed, band in enumerate(BANDS)}
# both bands inside every range: the union of their 0.1-99.9% quantiles
ranges = [
    (min(np.quantile(s[:, i], 0.001) for s in samples.values()), max(np.quantile(s[:, i], 0.999) for s in samples.values()))
    for i in range(3)
]
ranges += [(0, 360), (-90, 90)]
fig = plt.figure(figsize=(0.72 * style.FULL, 0.72 * style.FULL))
for band, color in BANDS.items():
    corner.corner(
        samples[band],
        fig=fig,
        labels=LABELS,
        range=ranges,
        bins=40,
        smooth=1.0,
        levels=LEVELS,
        color=color,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=band == "full",
        contour_kwargs=dict(linewidths=0.9),
        hist_kwargs=dict(density=True, lw=0.9),
        max_n_ticks=3,
        labelpad=0.08,
    )
for ax in fig.axes:
    ax.tick_params(labelsize=6.5)
    ax.xaxis.label.set_size(7.5)
    ax.yaxis.label.set_size(7.5)
handles = [plt.Line2D([], [], color=c, lw=1.2) for c in BANDS.values()]
fig.legend(handles, ["full band", "narrow band"], loc="upper right", bbox_to_anchor=(0.97, 0.97), frameon=False, fontsize=8)
style.save(fig, "prior-corner")
style.save_numbers("prior-corner", {
    "PriorCornerDraws": f"{len(samples['full'])}",
    "PriorCornerInner": f"{100 * LEVELS[0]:.0f}",
    "PriorCornerOuter": f"{100 * LEVELS[1]:.0f}",
})
