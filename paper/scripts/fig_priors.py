"""Figure: the two non-trivial priors.

(a) the chirp mass of the observationally driven DWD population (Korol et al. 2022);
(b) the matched-filter SNR the amplitude prior actually produces, per source, on the XS rung:
    the prior is log-uniform in the sky-averaged SNR at the window centre, and the full
    response (sky, polarisation, inclination) spreads it around that.
"""

import numpy as np

import data
import style
from style import plt

import jax
import jax.numpy as jnp
import jax.random as jr
from canna.lisa import ChirpMass

style.setup()
fig, axes = plt.subplots(1, 2, figsize=(style.COLUMN, 1.45), constrained_layout=True)

# (a) chirp mass
ax = axes[0]
prior = ChirpMass()
mc = np.asarray(jax.vmap(prior)(jr.split(jr.key(1), 200_000)))[:, 0]
bins = np.linspace(0.1, 1.25, 70)
ax.hist(mc, bins=bins, density=True, color=style.BLUE, alpha=0.85, lw=0)
ax.axvline(np.median(mc), color=style.INK, lw=0.7)
ax.text(0.72, ax.get_ylim()[1] * 0.55, f"median {np.median(mc):.2f}",
        fontsize=6, color=style.INK2)
ax.set(xlabel=r"chirp mass $\mathcal{M}$ [$M_\odot$]", ylabel="density", xlim=(0.1, 1.25))
ax.set_yticks([])
style.panel_label(ax, "a")

# (b) per-source SNR under the XS prior
problem, cfg = data.xs_problem()
key_w, key_p = jr.split(jr.key(7))
n = 2048
windows = jax.vmap(problem.sample_f)(jr.split(key_w, n))
latents = jax.vmap(problem.sample_physical)(jr.split(key_p, n), windows)
per_source = jax.vmap(lambda p, f: problem.snr(p[:, None, :], f))(latents, windows)
rho = np.asarray(per_source).ravel()
ax = axes[1]
edges = np.logspace(0, 3.7, 60)
ax.hist(rho, bins=edges, density=False, weights=np.full(rho.size, 1 / rho.size / np.diff(np.log10(edges))[0]),
        color=style.BLUE, alpha=0.85, lw=0)
ax.axvspan(7, 1000, color=style.BLUE, alpha=0.08, lw=0)
flat = 1 / np.log10(1000 / 7)
ax.plot([7, 1000], [flat, flat], color=style.INK, lw=0.8)
ax.text(80, flat * 1.08, "log-uniform, 7 to 1000", fontsize=5.8, color=style.INK2, va="bottom", ha="center")
ax.set_xscale("log")
ax.set(xlabel=r"source SNR $\rho$", ylabel=r"density in $\log_{10}\rho$", xlim=(1, 5e3), ylim=(0, 0.72))
frac_low = np.mean(rho < 7)
ax.text(0.98, 0.99, f"{frac_low:.0%} below 7, {np.mean(rho > 1000):.0%} above 1000",
        transform=ax.transAxes, ha="right", va="top", fontsize=5.8, color=style.INK2)
style.panel_label(ax, "b")
print(f"chirp mass median {np.median(mc):.3f}, mode bin {bins[np.argmax(np.histogram(mc, bins)[0])]:.3f};"
      f" SNR: {frac_low:.3f} below 7, {np.mean(rho > 1000):.3f} above 1000, median {np.median(rho):.1f}")
style.save(fig, "priors")
