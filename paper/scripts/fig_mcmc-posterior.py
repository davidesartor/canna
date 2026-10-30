"""Figure: the joint reference MCMC posterior of every source of one eval injection, against the Fisher forecast."""

import json

import numpy as np
from matplotlib.colors import to_rgba
from scipy.ndimage import gaussian_filter

import style
from style import plt

DRAWS = style.REPO / "reports/mcmc/data/draws.json"
INJECTION = 2
# columns of draws.json: f0 [bins], the drift (ln Mc in the old chains, ln fdot now), ln A, alpha, delta, psi, cos iota, phi0
PAIRS = [(0, 1), (2, 6), (3, 4), (5, 7)]
LABELS = [
    r"$\Delta f_0$ [bins]",
    None,
    r"$\Delta\ln\mathcal{A}$",
    r"$\Delta\alpha$",
    r"$\Delta\delta$",
    r"$\Delta\psi$",
    r"$\Delta\cos\iota$",
    r"$\Delta\phi_0$",
]
MASSES = (0.95, 0.68)
BINS = 32


def credible_levels(density: np.ndarray, masses: tuple[float, ...]) -> list[float]:
    """Density thresholds enclosing the given probability masses, highest density first."""
    ordered = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    return [ordered[np.searchsorted(cumulative, m)] for m in masses]


def contours(ax, x, y, extent, filled: bool, color: str, smooth: float) -> None:
    """68% and 95% highest-density contours of a smoothed 2-D histogram."""
    hist, xe, ye = np.histogram2d(x, y, bins=BINS, range=extent)
    density = gaussian_filter(hist.T, smooth)
    xc, yc = (xe[1:] + xe[:-1]) / 2, (ye[1:] + ye[:-1]) / 2
    levels = credible_levels(density, MASSES)
    if filled:
        ax.contourf(
            xc,
            yc,
            density,
            levels=[*levels, density.max() * 1.01],
            colors=[to_rgba(color, 0.35), to_rgba(color, 0.8)],
        )
    else:
        ax.contour(xc, yc, density, levels=levels, colors=color, linewidths=[0.6, 1.0])


# offsets from the truth, quantised in hundredths of draws.json's scale; joint seeds pooled
draws = json.loads(DRAWS.read_text())
window = next(w for w in draws["windows"] if w["j"] == INJECTION)
assert window["joint_width"], f"injection {INJECTION} has no joint chains"
LABELS[1] = {"log Mc": r"$\Delta\ln\mathcal{M}_c$", "log fdot": r"$\Delta\ln\dot f$"}[
    draws["columns"][1]
]
source_snr = np.array(window["snr"])
unit = lambda s: np.array(window["scale"][s]) / draws["quant"]

style.setup()
fig, axes = plt.subplots(
    len(source_snr),
    len(PAIRS),
    figsize=(style.FULL, 5.9),
    constrained_layout=True,
)
for s, row in enumerate(axes):
    mcmc_draws = np.concatenate(
        [np.array(seed[s]).T for seed in window["draws"]["joint"]]
    ) * unit(s)
    fisher_draws = np.array(window["draws"]["fisher"][s]).T * unit(s)
    for ax, (i, j) in zip(row, PAIRS):
        # axis range: the central 98% of both sets, so a mode holding under 1% does not squash the plot
        both = np.concatenate([mcmc_draws, fisher_draws])
        lo, hi = np.percentile(both[:, [i, j]], [1, 99], axis=0)
        pad = 0.08 * (hi - lo)
        extent = [[lo[0] - pad[0], hi[0] + pad[0]], [lo[1] - pad[1], hi[1] + pad[1]]]
        contours(ax, mcmc_draws[:, i], mcmc_draws[:, j], extent, True, style.BLUE, 1.2)
        contours(
            ax, fisher_draws[:, i], fisher_draws[:, j], extent, False, style.ORANGE, 2.0
        )
        ax.axvline(0, color=style.INK2, lw=0.4, zorder=3)
        ax.axhline(0, color=style.INK2, lw=0.4, zorder=3)
        ax.set_xlim(extent[0])
        ax.set_ylim(extent[1])
        ax.set_xlabel(LABELS[i], labelpad=1)
        ax.set_ylabel(LABELS[j], labelpad=1)
        ax.ticklabel_format(style="sci", scilimits=(-2, 3), useMathText=True)
        ax.locator_params(nbins=4)
    snr_text = f"{source_snr[s]:.0f}" if source_snr[s] >= 10 else f"{source_snr[s]:.1f}"
    row[0].set_title(f"source {s + 1}, SNR {snr_text}")
handles = [
    plt.Rectangle((0, 0), 1, 1, color=to_rgba(style.BLUE, 0.6)),
    plt.Line2D([], [], color=style.ORANGE, lw=1.0),
    plt.Line2D([], [], color=style.INK2, lw=0.5),
]
fig.legend(
    handles,
    ["joint MCMC (68%, 95%)", "Fisher forecast (68%, 95%)", "truth"],
    loc="outside upper center",
    ncol=3,
)
style.save(fig, "mcmc-posterior")
