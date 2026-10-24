"""figures/chart.pdf: what the chart does to amplitude and inclination, and the prior in flow coordinates."""

import json

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import style
from style import plt

BANDS = {"narrow": style.ORANGE, "full": style.BLUE}
LABELS = [
    r"$x_f$",
    r"$x_{\mathcal{M}}$",
    r"$x_l$",
    r"$x_b$",
    r"Re $z_+$",
    r"Im $z_+$",
    r"Re $z_-$",
    r"Im $z_-$",
]

prior = json.loads((style.REPO / "reports/prior/data/prior.json").read_text())
narrow = prior["bands"]["narrow"]
scatter = {k: np.asarray(v) for k, v in narrow["scatter"].items()}

# chart radii of the recorded sources: A/u = rho/kappa, cos(iota) is the stored sin of the complement
amplitude_in_units = scatter["snr"] / scatter["kappa"]
cos_iota = scatter["sin_iota"]
z_plus = np.arcsinh(amplitude_in_units * (1 - cos_iota) ** 2 / 4)
z_minus = np.arcsinh(amplitude_in_units * (1 + cos_iota) ** 2 / 4)
print(
    "median asinh|z+|, |z-| from the scatter",
    np.median(z_plus),
    np.median(z_minus),
    "recorded",
    [r["quantiles"][2] for r in narrow["radius"]],
)

style.setup()
inclination_cmap = LinearSegmentedColormap.from_list("inclination", style.BLUE_RAMP[1:])
fig = plt.figure(figsize=(style.FULL, 3.3), layout="constrained")
top, bottom = fig.subfigures(2, 1, height_ratios=[1.75, 1.0])
axes = top.subplots(1, 3)
order = np.argsort(np.abs(cos_iota))
dots = dict(
    c=np.abs(cos_iota)[order],
    cmap=inclination_cmap,
    vmin=0,
    vmax=1,
    s=2.5,
    lw=0,
    rasterized=True,
)

# (a) amplitude and inclination as drawn
ax = axes[0]
mappable = ax.scatter(amplitude_in_units[order], cos_iota[order], **dots)
ax.set(
    xscale="log",
    xlabel=r"$\mathcal{A}/u(f_w)$",
    ylabel=r"$\cos\iota$",
    ylim=(-1.05, 1.05),
)
style.panel_label(ax, "a")

# (b) the same sources in the chart's warped helicity radii
ax = axes[1]
ax.scatter(z_plus[order], z_minus[order], **dots)
reach = max(z_plus.max(), z_minus.max()) * 1.04
ax.plot([0, reach], [0, reach], **style.GUIDE)
ax.set(
    xlim=(0, reach),
    ylim=(0, reach),
    aspect="equal",
    xlabel=r"$|z_+| = \mathrm{asinh}|\zeta_+|$",
    ylabel=r"$|z_-|$",
)
top.colorbar(
    mappable, ax=axes[:2], label=r"$|\cos\iota|$", shrink=0.8, aspect=25, pad=0.01
)
style.panel_label(ax, "b")

# (c) SNR per unit amplitude against inclination, with the sky-averaged law
ax = axes[2]
grid = np.linspace(-1, 1, 201)
sky_constant = narrow["sky_average"]["galactic"]["C_low"]
law = np.sqrt(sky_constant * (1 + 6 * grid**2 + grid**4) / 3.2)
ax.scatter(
    cos_iota, scatter["kappa"], color=style.BLUE_RAMP[2], s=2.5, lw=0, rasterized=True
)
ax.plot(grid, law, color=style.INK, lw=1.0)
ax.set(
    xlabel=r"$\cos\iota$",
    ylabel=r"$\kappa = \rho\,u(f_w)/\mathcal{A}$",
    xlim=(-1.02, 1.02),
    ylim=(0, None),
)
style.panel_label(ax, "c")

# (d) the prior's marginal of each flow coordinate, per band, network reach shaded beyond
marginal_axes = bottom.subplots(1, 8, sharey=False)
for j, ax in enumerate(marginal_axes):
    for band, color in BANDS.items():
        coord = prior["bands"][band]["flow"][j]
        edges, counts = (
            np.asarray(coord["hist"]["edges"]),
            np.asarray(coord["hist"]["counts"]),
        )
        width = edges[1] - edges[0]
        total = counts.sum() + coord["hist"]["outside"]
        ax.stairs(counts / total / width, edges, color=color, lw=0.9)
        if "reference" in coord["hist"] and band == "full":
            centres = (edges[1:] + edges[:-1]) / 2
            ax.plot(
                centres,
                np.asarray(coord["hist"]["reference"]) / total / width,
                color=style.INK,
                lw=0.6,
                ls="--",
            )
    lo, hi = coord["reach"]
    left, right = ax.get_xlim()
    ax.axvspan(left, lo, color=style.FILL, lw=0)
    ax.axvspan(hi, right, color=style.FILL, lw=0)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom=0)
    ax.set_yticks([])
    ax.set_xlabel(LABELS[j])
    ax.tick_params(axis="x", labelsize=5.5)
style.panel_label(marginal_axes[0], "d")
marginal_axes[0].legend(
    [plt.Line2D([], [], color=c) for c in BANDS.values()],
    list(BANDS),
    loc="lower center",
    fontsize=5.5,
    handlelength=1.0,
)

style.save(fig, "chart")
