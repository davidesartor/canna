"""Figure: why the loss cannot see a loud source's width, and what the warped clock changes.

1-D Gaussian model of one flow coordinate (Appendix F): base x0 ~ N(0, s0^2), posterior
x1 ~ N(0, sigma^2), straight path x_s = (1 - s) x0 + s x1, target u = x1 - x0. A model whose
field is exactly that of a N(0, sigma_m^2) posterior pays an excess flow-matching loss

    dL_p = int_0^1 dt  m^2(s) [g_m(s) - g(s)]^2,     s = 1 - (1 - t)^p,

with m^2 = (1-s)^2 s0^2 + s^2 sigma^2 and g = (s sigma^2 - (1-s) s0^2) / m^2. For
sigma << sigma_m << s0 this tends to C_p s0^(2 - 1/p) sigma_m^(1/p), with
C_p = (2p - 1) pi / (4 p^2 sin(pi / 2p)): pi/4 for the uniform clock, 5 pi / 18 for p = 3.

(a) the share of training samples that land within 1 - s < eps of the end of the path;
(b) the excess loss against the model's width, numerically and from the asymptote.
"""

import numpy as np

import style
from style import plt

style.setup()
S0 = 1 / np.sqrt(3)  # std of U[-1, 1], the f0 coordinate's prior
BINS_PER_UNIT = 24.0  # XS: the 48-bin interior spans 2 flow units


def excess(sigma_m, sigma, p, s0=S0):
    """Excess flow-matching loss of a N(0, sigma_m^2) model, exact quadrature in z = -log(1-t)."""
    z = np.linspace(0, 80 / p, 400_001)
    eps = np.exp(-p * z)  # 1 - s
    s = 1 - eps
    dt = np.exp(-z)  # dt/dz
    m2 = eps**2 * s0**2 + s**2 * sigma**2
    mm2 = eps**2 * s0**2 + s**2 * sigma_m**2
    g = (s * sigma**2 - eps * s0**2) / m2
    gm = (s * sigma_m**2 - eps * s0**2) / mm2
    return np.trapezoid(m2 * (gm - g) ** 2 * dt, z)


def asymptote(sigma_m, p, s0=S0):
    c = (2 * p - 1) * np.pi / (4 * p**2 * np.sin(np.pi / (2 * p)))
    return c * s0 ** (2 - 1 / p) * sigma_m ** (1 / p)


fig, axes = plt.subplots(1, 2, figsize=(style.COLUMN, 1.55), constrained_layout=True)

# (a) where the training samples land
ax = axes[0]
eps = np.logspace(-6, 0, 200)
for p, color, name in [(1, style.ORANGE, "uniform, $p=1$"), (3, style.BLUE, "warped, $p=3$")]:
    ax.loglog(eps, eps ** (1 / p), color=color, label=name)
marks = [(0.18, "0.18"), (0.037, "0.037"), (np.sqrt(3) / (np.pi * 1000), "ideal")]
for w, text in marks:
    e = w / BINS_PER_UNIT / S0
    ax.axvline(e, color=style.AXIS, lw=0.7)
    ax.text(e / 1.35, 1.6e-6, text, rotation=90, fontsize=5.5, color=style.INK2, va="bottom", ha="right")
ax.set(xlabel=r"$\epsilon$ (distance to the end, $1-s$)", ylabel=r"share of samples with $1-s<\epsilon$",
       xlim=(1e-6, 1), ylim=(1e-6, 1.5))
ax.legend(loc="upper left", fontsize=5.8)
style.panel_label(ax, "a")

# (b) the excess loss against the model width, true width far below
ax = axes[1]
widths = np.logspace(-5, -0.7, 22)  # in flow units
for p, color in [(1, style.ORANGE), (3, style.BLUE)]:
    exact = np.array([excess(w, 1e-7, p) for w in widths])
    ax.loglog(widths * BINS_PER_UNIT, exact, color=color, lw=0, marker="o", ms=2.6, mec="none")
    ax.loglog(widths * BINS_PER_UNIT, asymptote(widths, p), color=color, lw=1.0)
    print(f"p={p}: exact/asymptote at 0.037 bins = "
          f"{excess(0.037 / BINS_PER_UNIT, 1e-7, p) / asymptote(0.037 / BINS_PER_UNIT, p):.3f}")
for w, _ in marks[:2]:
    ax.axvline(w, color=style.AXIS, lw=0.7)
r = asymptote(0.18 / BINS_PER_UNIT, 3) / asymptote(0.18 / BINS_PER_UNIT, 1)
r2 = asymptote(0.037 / BINS_PER_UNIT, 3) / asymptote(0.037 / BINS_PER_UNIT, 1)
ax.text(0.97, 0.05, f"warp / uniform:\n×{r:.0f} at 0.18 bins\n×{r2:.0f} at 0.037 bins",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5, color=style.INK2)
ax.set(xlabel=r"model width $\sigma_m$ [XS bins]", ylabel=r"excess loss $\Delta L$ per coordinate")
style.panel_label(ax, "b")
print(f"ratio warp/uniform: {r:.1f} at 0.18 bins, {r2:.1f} at 0.037 bins")
style.save(fig, "warp")
