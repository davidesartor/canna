"""Figure: the loudest eval source (SNR 1366), flow against a joint MCMC and the Fisher forecast.

As fig_corner.py's loudest page, with the MCMC added: 4000 draws of jexplore's cold chains
over the window's four sources (lisa_checks/mcmc_xs.py, two seeds), relabelled onto the
source. The MCMC samples one (psi, phi0) copy; each draw is moved to one of the four exact
copies, (psi + k pi/2, phi0 + k pi), at random, so that it is shown on the flow's footing.
"""

import numpy as np

import data
import style
from style import plt
import corner
from fig_corner import LABELS, features

style.setup()
problem, _ = data.xs_problem()
inj = data.eval_injections()
j = 9
z = data.mcmc(j)
s = int(z["loudest"])
f, truth = float(inj["window"][j]), inj["truth"][j]
assert np.allclose(z["truth"], truth), "MCMC summary is for another injection"

mcmc = np.array(z["mcmc_loudest"])
k = np.random.default_rng(0).integers(0, 4, len(mcmc))
mcmc[:, 5] += k * np.pi / 2
mcmc[:, 7] += k * np.pi
mcmc = features(mcmc, truth[s, 0])
flow = features(data.matched_draws(problem, data.final_draws(j), truth, f)[:, s], truth[s, 0])
fish = features(data.fisher_draws(j)[:, s], truth[s, 0])
true = features(truth[s][None], truth[s, 0])[0]

main = np.abs(flow[:, 0]) < 1.0
ranges = []
for c in range(8):
    if c in (5, 7):
        ranges.append((0, 2 * np.pi))
        continue
    lo = min(np.percentile(flow[main, c], 0.5), np.percentile(mcmc[:, c], 0.5))
    hi = max(np.percentile(flow[main, c], 99.5), np.percentile(mcmc[:, c], 99.5))
    pad = 0.08 * (hi - lo)
    ranges.append((lo - pad, hi + pad))
fig = plt.figure(figsize=(style.FULL, style.FULL))
kw = dict(range=ranges, bins=32, smooth=0.9, levels=(0.393, 0.865), plot_datapoints=False,
          label_kwargs=dict(fontsize=7.5), max_n_ticks=3, labelpad=0.08)
corner.corner(flow, fig=fig, labels=LABELS, color=style.BLUE, fill_contours=True,
              hist_kwargs=dict(density=True, lw=1.0), **kw)
corner.corner(fish, fig=fig, color=style.MUTED, plot_density=False, no_fill_contours=True,
              contour_kwargs=dict(linewidths=0.7, linestyles="dashed"),
              hist_kwargs=dict(density=True, lw=0.7, ls="dashed"), **kw)
corner.corner(mcmc, fig=fig, color=style.INK, plot_density=False, no_fill_contours=True,
              contour_kwargs=dict(linewidths=0.9), hist_kwargs=dict(density=True, lw=0.9), **kw)
corner.overplot_lines(fig, true, color=style.RED, lw=0.6)
for ax in fig.axes:
    ax.tick_params(labelsize=5.5, length=1.8)
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    ax.set_rasterized(True)
fig.text(0.62, 0.86, f"loudest eval source: SNR {z['snr'][s]:.1f}\nwindow at {f * 1e3:.4f} mHz\n"
         "blue: flow (final model)\nblack: MCMC, all four sources jointly\n"
         "grey, dashed: Fisher forecast\nred: truth", fontsize=8, color=style.INK, va="top")
style.save(fig, "mcmc")
