"""Figure: where the clock puts the training draws, what the loss pays for a too-wide model, and what the sampler returns.

1-D Gaussian model of the f0 coordinate: base N(0, s0^2) with s0 = 1/sqrt(3), posterior N(0, sigma^2).
(a) Pr(1 - s < eps) under the uniform clock, the late-time mix and a p = 3 power clock;
(b) the excess flow-matching loss of a N(0, sigma_m^2) model, quadrature against the asymptotes;
(c) the width canna.lisa.sample.sample_posterior returns for the exact Gaussian endpoint E[x1 | x_s].
Bars: the ideal f0 widths of each band, between the SNR range's ends.
"""

import math

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import style
from style import plt
from canna.lisa import config, sample
from canna.lisa.network import LOSS_TIME_DECADES
from canna.lisa.problem import WINDOW_STRIDE, LisaGB, f0_window

style.setup()
cfg = config.load("full-B")
mix, floor, p, steps = (
    cfg.late_time_mix,
    cfg.late_time_floor,
    cfg.time_power,
    cfg.ode_steps,
)
S0 = 1 / math.sqrt(3)
CLOCKS = {
    "uniform": style.MUTED,
    "power $p=3$": style.ORANGE,
    "late-time mix": style.BLUE,
}
STEP_COUNTS = {
    steps: style.BLUE_RAMP[2],
    4 * steps: style.BLUE_RAMP[4],
    16 * steps: style.BLUE_RAMP[6],
}


def ideal_range(name: str) -> tuple[float, float]:
    """The band's ideal f0 widths over s0 at the loud and the weak end of its SNR range."""
    run = config.load(name)
    problem = LisaGB(**run.problem)
    lo, hi = problem.window_index_range
    f = ((lo + hi) // 2 * WINDOW_STRIDE + problem.window_bins / 2) / problem.t_obs
    low, high = (float(x) for x in f0_window(problem, jnp.asarray(f)))
    bins_per_unit = problem.t_obs * math.sqrt(low * high) * math.log(high / low) / 2
    snr_lo, snr_hi = run.problem["snr_range"]
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


def excess(sigma_m, sigma, late_share, s0=S0):
    z = np.linspace(0, 60, 600_001)
    eps = np.exp(-z)
    s = 1 - eps
    m2 = eps**2 * s0**2 + s**2 * sigma**2
    mm2 = eps**2 * s0**2 + s**2 * sigma_m**2
    g = (s * sigma**2 - eps * s0**2) / m2
    gm = (s * sigma_m**2 - eps * s0**2) / mm2
    weight = (1 - late_share) * eps + late_share * (z <= -math.log(floor)) / -math.log(
        floor
    )
    return np.trapezoid(m2 * (gm - g) ** 2 * weight, z)


class GaussianFlow(eqx.Module):
    """The exact endpoint E[x1 | x_s] of a N(0, s0^2) base and a N(0, sigma^2) posterior, as a LisaFlow."""

    sigma: float
    s0: float = S0
    prediction: str = eqx.field(static=True, default="x")
    self_condition: bool = eqx.field(static=True, default=True)

    def encode(self, y, f):
        return jnp.zeros(())

    def __call__(self, u, t, y, f, x_hat, cache):
        s = 1 - (1 - t) ** p
        m2 = (1 - s) ** 2 * self.s0**2 + s**2 * self.sigma**2
        return s * self.sigma**2 / m2 * u


def sampled_width(sigma_scaled: float, ode_steps: int, n: int = 1 << 14) -> float:
    u = S0 * jr.normal(jr.key(0), (n, 1))
    flow = GaussianFlow(sigma=sigma_scaled * S0)
    x = sample.sample_posterior(flow, u, jnp.zeros(()), jnp.zeros(()), ode_steps, p)
    return float(jnp.std(x) / (sigma_scaled * S0))


fig, axes = plt.subplots(1, 3, figsize=(style.FULL, 2.0), layout="constrained")

# (a) where the training draws land
ax = axes[0]
eps = np.logspace(-8, 0, 400)
share = {
    "uniform": eps,
    "power $p=3$": eps ** (1 / 3),
    "late-time mix": (1 - mix) * eps
    + mix * np.clip(1 - np.log(eps) / math.log(floor), 0, 1),
}
for name, color in CLOCKS.items():
    ax.loglog(eps, share[name], color=color)
style.direct_label(
    ax,
    3e-7,
    3e-7 ** (1 / 3),
    "power $p=3$",
    dx=0,
    dy=-6,
    ha="left",
    va="top",
    color=CLOCKS["power $p=3$"],
)
style.direct_label(
    ax,
    1e-4,
    3e-3,
    "late-time mix",
    dx=4,
    ha="left",
    color=CLOCKS["late-time mix"],
)
style.direct_label(ax, 1e-3, 1e-3, "uniform", dx=4, dy=-4, va="top", color=style.INK2)
ax.axvline(10**-LOSS_TIME_DECADES, **style.GUIDE)
style.direct_label(
    ax,
    10**-LOSS_TIME_DECADES,
    0.97,
    "loss floor",
    dx=-2,
    ha="right",
    va="top",
    xycoords=ax.get_xaxis_transform(),
)
band_bars(ax)
ax.set(
    xlabel=r"$\epsilon$", ylabel=r"$\Pr(1-s<\epsilon)$", xlim=(1e-8, 1), ylim=(1e-9, 2)
)
ax.set_title("Where training draws land")
style.panel_label(ax, "a")

# (b) the excess loss of a model sigma_m wide, the true posterior 1e4 times narrower
ax = axes[1]
widths = np.logspace(-4, -0.5, 9)
fine = np.logspace(-4.2, -0.4, 100)
mix_asymptote = (1 - mix) * math.pi / 4 * S0**2 * fine + mix * S0**2 * (
    np.log(fine / floor) - 0.5
) / -math.log(floor)
inside = fine > 3 * floor
asymptotes = {
    "uniform": (fine, math.pi / 4 * S0**2 * fine),
    "late-time mix": (fine[inside], mix_asymptote[inside]),
}
for name, late_share in (("uniform", 0.0), ("late-time mix", mix)):
    quadrature = [excess(w * S0, 1e-4 * w * S0, late_share) for w in widths]
    ax.loglog(*asymptotes[name], color=CLOCKS[name], lw=0.9)
    ax.loglog(
        widths,
        quadrature,
        ls="none",
        marker="o",
        ms=3.4,
        mfc="none",
        mec=CLOCKS[name],
        mew=0.8,
    )
style.direct_label(
    ax,
    3e-2,
    math.pi / 4 * S0**2 * 3e-2,
    "uniform",
    dx=4,
    dy=-4,
    ha="left",
    va="top",
    color=style.INK2,
)
style.direct_label(
    ax,
    1e-2,
    mix_asymptote[np.searchsorted(fine, 1e-2)],
    "late-time mix",
    dx=0,
    dy=-8,
    ha="left",
    color=CLOCKS["late-time mix"],
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
ax.set(xlabel=r"model width $\sigma_m/s_0$", ylabel=r"excess loss $\Delta L$")
ax.set_title("What a too-wide model costs")
style.panel_label(ax, "b")

# (c) the width the sampler returns for the exact Gaussian endpoint
ax = axes[2]
true = np.logspace(-7, -0.5, 27)
ax.axhline(1, **style.GUIDE)
for n_steps, color in STEP_COUNTS.items():
    ratio = [sampled_width(w, n_steps) for w in true]
    ax.loglog(true, ratio, color=color)
    style.direct_label(
        ax, true[0], ratio[0], f"{n_steps} steps", dx=-3, ha="right", color=color
    )
band_bars(ax)
ax.set(
    xlabel=r"true width $\sigma/s_0$",
    ylabel="sampled / true width",
    xlim=(1e-10, 1),
    ylim=(1e-6, 3),
)
ax.set_title("What the sampler returns")
style.panel_label(ax, "c")

style.save(fig, "warp")
