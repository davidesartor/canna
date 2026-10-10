"""Figure: single-source posteriors, flow (blue) against the Fisher forecast (grey contours).

The median-SNR (62.4) and the loudest (1366) of the 40 eval sources, final model, 1024
draws relabelled onto the source. The Fisher forecast is a Gaussian at the truth with the
noise-averaged Hessian plus the prior precision (Appendix I): a reference width, not a
posterior. It cannot show the exact degeneracies the flow reproduces, (psi, phi0) ->
(psi + pi/2, phi0 + pi) and psi -> psi + pi.
"""

import numpy as np

import data
import style
from style import plt
import corner

style.setup()
problem, _ = data.xs_problem()
inj = data.eval_injections()
T = problem.t_obs
LABELS = [r"$\Delta f_0$ [bins]", r"$\mathcal{M}$ [$M_\odot$]", r"$\log_{10}\mathcal{A}$",
          r"$\alpha$", r"$\sin\delta$", r"$\psi$", r"$\cos\iota$", r"$\phi_0$"]


def features(p, f0_true):
    return np.stack([(p[:, 0] - f0_true) * T, p[:, 1], np.log10(np.abs(p[:, 2])), p[:, 3],
                     np.sin(p[:, 4]), np.mod(p[:, 5], 2 * np.pi), np.sin(p[:, 6]),
                     np.mod(p[:, 7], 2 * np.pi)], axis=-1)


def page(j, s, name):
    f, truth = float(inj["window"][j]), inj["truth"][j]
    matched = data.matched_draws(problem, data.final_draws(j), truth, f)
    flow = features(matched[:, s], truth[s, 0])
    fish = features(data.fisher_draws(j)[:, s], truth[s, 0])
    true = features(truth[s][None], truth[s, 0])[0]
    # the axes follow the draws that found this source (|df0| < 1 bin); the rest are draws
    # whose slot, matched onto this source, belongs to an unlocalised one in the same window
    main = np.abs(flow[:, 0]) < 1.0
    print(f"{name}: {1 - main.mean():.1%} of the draws lie outside the main mode")
    ranges = []
    for c in range(8):
        if c in (5, 7):
            ranges.append((0, 2 * np.pi))
            continue
        lo = min(np.percentile(flow[main, c], 0.5), np.percentile(fish[:, c], 0.5))
        hi = max(np.percentile(flow[main, c], 99.5), np.percentile(fish[:, c], 99.5))
        pad = 0.08 * (hi - lo)
        ranges.append((lo - pad, hi + pad))
    fig = plt.figure(figsize=(style.FULL, style.FULL))
    kw = dict(range=ranges, bins=32, smooth=0.9, levels=(0.393, 0.865), plot_datapoints=False,
              label_kwargs=dict(fontsize=7.5), max_n_ticks=3, labelpad=0.08)
    corner.corner(flow, fig=fig, labels=LABELS, color=style.BLUE, fill_contours=True,
                  hist_kwargs=dict(density=True, lw=1.0), **kw)
    corner.corner(fish, fig=fig, color=style.MUTED, plot_density=False, no_fill_contours=True,
                  contour_kwargs=dict(linewidths=0.8), hist_kwargs=dict(density=True, lw=0.8), **kw)
    corner.overplot_lines(fig, true, color=style.INK, lw=0.6)
    for ax in fig.axes:
        ax.tick_params(labelsize=5.5, length=1.8)
        ax.grid(False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
        ax.set_rasterized(True)
    import jax.numpy as jnp
    snr = float(np.asarray(problem.snr(jnp.asarray(truth[s])[None, None, :], f))[0])
    fig.text(0.62, 0.86, f"{name} eval source: SNR {snr:.1f}\nwindow at {f * 1e3:.4f} mHz\n"
             "blue: flow (final model)\ngrey: Fisher forecast\nblack: truth",
             fontsize=8, color=style.INK, va="top")
    style.save(fig, f"corner_{name}")


if __name__ == "__main__":
    page(5, 2, "median")
    page(9, 3, "loudest")
