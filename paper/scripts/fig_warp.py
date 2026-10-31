"""Figure: what the velocity MSE and the Gaussian NLL of x1 pay for a too-wide model, and what the sampler returns.

1-D Gaussian model of one coordinate: base N(0, s0^2), posterior N(0, sigma^2), uniform time.
(a) the excess loss of a N(0, sigma_m^2) model under each loss, quadrature against the asymptotes;
(b) the width canna.lisa.sample.sample_posterior returns for the exact Gaussian z1 | z_t, base N(0, 1) in the prior's frame.
Bars: the ideal f0 widths of each band over the prior's s0 = 1/sqrt(3), the widths without a frame, between the SNR range's ends.
"""

import math

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import style
from style import plt
from canna.lisa import config, sample
from canna.lisa.network import COORDS, reference_frame
from canna.lisa.problem import WINDOW_STRIDE, LisaGB, f0_window

style.setup()
steps = config.load("full-B").ode_steps
S0 = 1 / math.sqrt(3)
LOSSES = {"velocity MSE": style.MUTED, "NLL of $x_1$": style.BLUE}
STEP_COUNTS = {
    steps: style.BLUE_RAMP[2],
    4 * steps: style.BLUE_RAMP[4],
    16 * steps: style.BLUE_RAMP[6],
}


def centre_window(name: str) -> tuple[LisaGB, float]:
    """The config's problem and the centre frequency of its middle window."""
    problem = LisaGB(**config.load(name).problem)
    lo, hi = problem.window_index_range
    return problem, ((lo + hi) // 2 * WINDOW_STRIDE + problem.window_bins / 2) / problem.t_obs


def ideal_range(name: str) -> tuple[float, float]:
    """The band's ideal f0 widths over s0 at the loud and the weak end of its visible SNRs: 7 to the prior's 99.99% quantile."""
    problem, f = centre_window(name)
    low, high = (float(x) for x in f0_window(problem, jnp.asarray(f)))
    bins_per_unit = problem.t_obs * math.sqrt(low * high) * math.log(high / low) / 2
    prior = style.REPO / f"reports/prior/f0-fdot-amplitude/data/{name.removesuffix('-B')}.npz"
    snr_lo, snr_hi = 7.0, float(np.quantile(np.load(prior)["snr"], 0.9999))
    ideal = lambda snr: math.sqrt(3) / (math.pi * snr) / bins_per_unit / S0
    return ideal(snr_hi), ideal(snr_lo)


bands = {"narrow band": ideal_range("narrow-B"), "full band": ideal_range("full-B")}


def band_bars(ax) -> None:
    """Each band's ideal-width range as a bar along the bottom of the panel."""
    for row, (name, (loud, weak)) in enumerate(bands.items()):
        height = 0.07 + 0.09 * row
        ax.plot(
            [loud, weak],
            [height, height],
            color=style.INK2,
            lw=1.6,
            transform=ax.get_xaxis_transform(),
            solid_capstyle="butt",
            zorder=2,
        )
        style.direct_label(ax, weak, height, name, xycoords=ax.get_xaxis_transform())


def marginal(eps, sigma, s0=S0):
    s = 1 - eps
    m2 = eps**2 * s0**2 + s**2 * sigma**2
    return m2, (s * sigma**2 - eps * s0**2) / m2


def excess(sigma_m, sigma, nll: bool, s0=S0):
    """Quadrature in z = -ln(1 - s) of a N(0, sigma_m^2) model's excess loss under uniform s."""
    z = np.linspace(0, 60, 600_001)
    eps = np.exp(-z)
    m2, g = marginal(eps, sigma, s0)
    mm2, gm = marginal(eps, sigma_m, s0)
    if nll:
        var = eps**2 * s0**2 * sigma**2 / m2
        var_m = eps**2 * s0**2 * sigma_m**2 / mm2
        density = np.log(var_m / var) + (var + eps**2 * m2 * (gm - g) ** 2) / var_m - 1
    else:
        density = m2 * (gm - g) ** 2
    return np.trapezoid(density * eps, z)


class GaussianFlow(eqx.Module):
    """The exact mean and log std of z1 | z_t of a N(0, 1) base and a N(0, sigma^2) z1, as a LisaFlow whose frame is the prior's."""

    sigma: float
    n_sources: int = eqx.field(static=True)

    def encode(self, y, f):
        return jnp.zeros((self.n_sources, 2 * COORDS)), None

    def __call__(self, z, t, frame, f, cache):
        m2 = (1 - t) ** 2 + t**2 * self.sigma**2
        log_std = jnp.log((1 - t) * self.sigma) - 0.5 * jnp.log(m2)
        return jnp.concat([t * self.sigma**2 / m2 * z, jnp.broadcast_to(log_std, z.shape)], axis=-1)


problem, f = centre_window("narrow-B")


def sampled_width(sigma: float, ode_steps: int, n: int = 1 << 10) -> float:
    z = jr.normal(jr.key(0), (n, problem.n_sources, COORDS))
    flow = GaussianFlow(sigma, problem.n_sources)
    x = sample.sample_posterior(problem, flow, z, jnp.zeros(()), jnp.asarray(f), ode_steps)
    # the coordinates past f0 have prior mean 0: x / sigma_frame is z1 without cancellation
    _, log_sigma = reference_frame(problem, flow.encode(None, f)[0], jnp.asarray(f))
    return float(jnp.std(x[..., 1:] / jnp.exp(log_sigma[..., 1:])) / sigma)


fig, axes = plt.subplots(1, 2, figsize=(style.FULL, 2.0), layout="constrained")

# (a) the excess loss of a model sigma_m wide, the true posterior 1e4 times narrower
ax = axes[0]
widths = np.logspace(-4, -0.5, 9)
fine = np.logspace(-4.2, -0.4, 100)
asymptotes = {
    "velocity MSE": math.pi / 4 * S0**2 * fine,
    "NLL of $x_1$": np.full_like(fine, math.log(1e8) + 1e-8 - 1),
}
for name, color in LOSSES.items():
    quadrature = [excess(w * S0, 1e-4 * w * S0, name != "velocity MSE") for w in widths]
    ax.loglog(fine, asymptotes[name], color=color, lw=0.9)
    ax.loglog(
        widths, quadrature, ls="none", marker="o", ms=3.4, mfc="none", mec=color, mew=0.8
    )
    style.direct_label(
        ax, 3e-3, asymptotes[name][np.searchsorted(fine, 3e-3)], name, dx=0, dy=4, ha="left", va="bottom", color=color
    )
style.direct_label(
    ax,
    0.98,
    0.03,
    "rings: quadrature\nlines: asymptotes",
    dx=0,
    ha="right",
    va="bottom",
    xycoords="axes fraction",
)
band_bars(ax)
ax.set(xlabel=r"model width $w_m/s_0$", ylabel=r"excess loss $\Delta L$", ylim=(1e-6, 300))
ax.set_title("What a too-wide model costs")
style.panel_label(ax, "a")

# (b) the width the sampler returns for the exact Gaussian z1 | z_t; the frame puts it near one
ax = axes[1]
true = np.logspace(-7, 0.5, 31)
ax.axhline(1, **style.GUIDE)
ax.axvline(1, **style.GUIDE)
style.direct_label(ax, 1, 0.97, "frame", dx=3, ha="left", va="top", xycoords=ax.get_xaxis_transform())
for n_steps, color in STEP_COUNTS.items():
    ratio = [sampled_width(w, n_steps) for w in true]
    ax.loglog(true, ratio, color=color)
    style.direct_label(
        ax, true[0], ratio[0], f"{n_steps} steps", dx=-3, ha="right", color=color
    )
band_bars(ax)
ax.set(
    xlabel=r"true width $w/s_0$",
    ylabel="sampled / true width",
    xlim=(1e-10, 10),
    ylim=(1e-6, 3),
)
ax.set_title("What the sampler returns")
style.panel_label(ax, "b")

style.save(fig, "warp")
