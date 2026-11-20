"""figures/priors.pdf: the per-source prior in three panels (2026-11-20, user's layout).

(a) Corner of (log f0, log fdot, log A) over the full band, the catalogue prior's own windows (visible_snr = 0):
    the drawn sources in blue, those at SNR >= 7 in red (contours), and the SNR = 7 line in grey in the joints.
(b) The sky prior in ecliptic (lambda, beta), Mollweide: the bulge and the two discs, each in its own shade.
(c) The orientation angles: the orbit's plane inclined by iota to the sky, psi to the line of nodes, phi0 along
    the orbit.

The drawn sources are cached in data/figures/priors-draws.npz (local only), so restyling never re-simulates.
"""

import json
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Patch

import style

VISIBLE_SNR = 7.0
WINDOWS = 8192
DRAWS = style.PAPER / "data/figures/priors-draws.npz"
REPORT = style.REPO / "reports/prior/f0-fdot-amplitude/data/f0-fdot-amplitude.json"
PRIOR_SHADES = [style.BLUE_RAMP[0], style.BLUE_RAMP[2]]  # the 86% and 39% regions of the drawn sources
SKY_PARTS = {  # component index in physics.SKY_WEIGHTS: (label, colour, top alpha), drawn in this order
    1: ("thick disc", style.AQUA, 0.55),
    2: ("thin disc", style.BLUE, 0.95),
    0: ("bulge", style.VIOLET, 0.95),
}


def draw_sources() -> dict:
    """WINDOWS full-band windows centred by the catalogue's f0 marginal, their sources and each one's SNR alone."""
    import jax
    import jax.random as jr

    from canna.lisa import config
    from canna.lisa.problem import LisaGB, sample_f
    from canna.lisa.simulator import sample_physical, source_snrs

    problem = LisaGB(**config.load("full-B").problem | dict(visible_snr=0.0))

    def one(key):
        key_f, key_p = jr.split(key)
        f = sample_f(problem, key_f)
        p = sample_physical(problem, key_p, f)
        return p, source_snrs(problem, p, f)

    p, snr = jax.lax.map(one, jr.split(jr.key(0), WINDOWS), batch_size=256)
    p, snr = np.asarray(p).reshape(-1, 8), np.asarray(snr).ravel()
    out = dict(
        log_f0=np.log10(p[:, 0]),
        log_fdot=np.log10(p[:, 1]),
        log_a=np.log10(p[:, 2]),
        snr=snr,
        f0_range=np.asarray(problem.f0_range),
    )
    DRAWS.parent.mkdir(parents=True, exist_ok=True)
    np.savez(DRAWS, **out)
    return out


style.setup()
d = dict(np.load(DRAWS)) if DRAWS.is_file() else draw_sources()
x = np.stack([d["log_f0"], d["log_fdot"], d["log_a"]], axis=1)
visible = d["snr"] >= VISIBLE_SNR
report = json.loads(REPORT.read_text())
curves = report["bands"]["full"]["curves"]
curve_f0, curve_a7 = np.asarray(curves["log_f0"]), np.asarray(curves["amp_snr7"])
mean, cov = np.asarray(report["prior"]["mean"]), np.asarray(report["prior"]["cov"])


def a7(log_f0):
    """log10 A at which a source of sky- and orientation-average response reaches SNR 7 (the report's curve)."""
    return np.interp(log_f0, curve_f0, curve_a7)


def conditional_mean(known: dict, want: int, x_known: np.ndarray) -> np.ndarray:
    """E[x_want | x_known] under the catalogue Gaussian; known maps coordinate index -> column of x_known."""
    k = list(known)
    gain = cov[want, k] @ np.linalg.inv(cov[np.ix_(k, k)])
    return mean[want] + (x_known - mean[k]) @ gain


def threshold_gap(coords: dict) -> np.ndarray:
    """log10 A - log10 A_7(f0), the coordinates not given at their conditional mean given those that are."""
    known = {i: v for i, v in coords.items()}
    stacked = np.stack(list(known.values()), axis=-1)
    log_f0 = known[0] if 0 in known else conditional_mean(known, 0, stacked)
    log_a = known[2] if 2 in known else conditional_mean(known, 2, stacked)
    return log_a - a7(log_f0)


def mass_levels(h: np.ndarray, shares=(0.86, 0.39)) -> list[float]:
    """Density levels enclosing the given shares of a 2-D histogram's mass."""
    flat = np.sort(h.ravel())[::-1]
    cum = np.cumsum(flat) / flat.sum()
    return [flat[np.searchsorted(cum, s)] for s in shares]


def smooth(h: np.ndarray) -> np.ndarray:
    k = np.array([1, 4, 6, 4, 1]) / 16
    for axis in (0, 1):
        h = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), axis, h)
    return h


fig = plt.figure(figsize=(style.FULL, 2.55))
sub_a, sub_b, sub_c = fig.subfigures(1, 3, width_ratios=[1.0, 1.25, 0.95], wspace=0.0)

# (a) corner ----------------------------------------------------------------------------------------------
names = [r"$\log_{10} f_0$", r"$\log_{10}\dot f$", r"$\log_{10}\mathcal{A}$"]
lo = np.quantile(x, 0.001, axis=0)
hi = np.quantile(x, 0.999, axis=0)
lo[0], hi[0] = np.log10(d["f0_range"])
for i in (1, 2):  # widen to every visible source
    lo[i], hi[i] = min(lo[i], x[visible, i].min()), max(hi[i], x[visible, i].max())
pad = 0.04 * (hi - lo)
lo, hi = lo - pad, hi + pad
axes = sub_a.subplots(3, 3, gridspec_kw=dict(hspace=0.08, wspace=0.08))
sub_a.subplots_adjust(left=0.2, right=0.97, bottom=0.17, top=0.93)
for r in range(3):
    for c in range(3):
        ax = axes[r, c]
        if c > r:
            ax.set_axis_off()
            continue
        ax.set_xlim(lo[c], hi[c])
        ax.tick_params(labelsize=5, length=2, pad=1)
        ticks = ([-4, -3], [-20, -16], [-24, -23, -22])
        ax.set_xticks(ticks[c])
        if r != c:
            ax.set_yticks(ticks[r])
        if r == c:  # marginal: the prior (blue), the visible sources (red)
            edges = np.linspace(lo[c], hi[c], 41)
            ax.stairs(np.histogram(x[:, c], edges, density=True)[0], edges, color=PRIOR_SHADES[1], fill=True, lw=0, alpha=0.8)
            ax.stairs(np.histogram(x[visible, c], edges, density=True)[0], edges, color=style.RED, lw=0.8)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
        else:  # joint: prior regions, visible sources, the SNR = 7 line
            ax.set_ylim(lo[r], hi[r])
            h, xe, ye = np.histogram2d(x[:, c], x[:, r], bins=60, range=[[lo[c], hi[c]], [lo[r], hi[r]]])
            h = smooth(h)
            levels = mass_levels(h)
            xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
            ax.contourf(xc, yc, h.T, levels=[levels[0], levels[1], h.max() * 1.01], colors=PRIOR_SHADES)
            # the visible sources' 39% and 86% regions, as red outlines (few sources: coarser, smoother)
            hv, xv, yv = np.histogram2d(x[visible, c], x[visible, r], bins=30, range=[[lo[c], hi[c]], [lo[r], hi[r]]])
            hv = smooth(smooth(hv))
            ax.contour(0.5 * (xv[1:] + xv[:-1]), 0.5 * (yv[1:] + yv[:-1]), hv.T, levels=sorted(mass_levels(hv)),
                       colors=style.RED, linewidths=[0.6, 1.0])
            gx, gy = np.meshgrid(np.linspace(lo[c], hi[c], 300), np.linspace(lo[r], hi[r], 300))
            ax.contour(gx, gy, threshold_gap({c: gx, r: gy}), levels=[0], colors=style.INK2, linewidths=0.9)
        if r < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(names[c], fontsize=6, labelpad=1)
        if c > 0 or r == 0:
            if r != c:
                ax.set_yticklabels([])
        else:
            ax.set_ylabel(names[r], fontsize=6, labelpad=1)
axes[0, 2].legend(
    handles=[
        Patch(color=PRIOR_SHADES[1], label="prior, 39 / 86%"),
        plt.Line2D([], [], color=style.RED, lw=1.0, label=rf"SNR $\geq$ {VISIBLE_SNR:.0f}, 39 / 86%"),
        plt.Line2D([], [], color=style.INK2, lw=0.9, label=rf"SNR $=$ {VISIBLE_SNR:.0f}"),
    ],
    loc="upper right",
    fontsize=5.5,
    frameon=False,
    handlelength=1.4,
    borderaxespad=0,
)
style.panel_label(axes[0, 0], "a", x=-0.55, y=1.08)

# (b) sky -------------------------------------------------------------------------------------------------
from astropy import units as u  # noqa: E402
from astropy.coordinates import BarycentricMeanEcliptic, SkyCoord  # noqa: E402

from canna.lisa.physics import (  # noqa: E402
    BULGE_SCALES,
    SKY_WEIGHTS,
    THICK_DISK_SCALES,
    THIN_DISK_SCALES,
)

ax = sub_b.add_subplot(projection="mollweide")
sub_b.subplots_adjust(left=0.1, right=0.98, bottom=0.3, top=0.93)
ra = np.linspace(-math.pi, math.pi, 721)
dec = np.linspace(-math.pi / 2, math.pi / 2, 361)
RA, DEC = np.meshgrid(ra, dec)
gal = SkyCoord(lon=(RA % (2 * math.pi)) * u.rad, lat=DEC * u.rad, frame=BarycentricMeanEcliptic()).galactic  # RA, DEC: ecliptic lambda, beta
l = (gal.l.rad + math.pi) % (2 * math.pi) - math.pi
s = np.sin(gal.b.rad)
xs = np.stack([l, s], axis=-1)
half_width = np.array([math.pi, 1.0])
parts = {
    0: (np.exp(-0.5 * (xs / np.array(BULGE_SCALES)) ** 2) / (math.sqrt(2 * math.pi) * np.array(BULGE_SCALES))).prod(-1),
    **{
        k: (np.exp(-np.abs(xs) / sc) / (2 * sc * (1 - np.exp(-half_width / sc)))).prod(-1)
        for k, sc in ((1, np.array(THICK_DISK_SCALES)), (2, np.array(THIN_DISK_SCALES)))
    },
}
# composite the three parts over white into one RGB image (no stacked translucent meshes, so no moire)
image = np.ones((*RA.shape, 3))
for k, (label, colour, top_alpha) in SKY_PARTS.items():
    density = np.log10(SKY_WEIGHTS[k] * parts[k] + 1e-300)
    top = density.max()
    alpha = top_alpha * np.clip((density - (top - 3.0)) / 3.0, 0, 1)[..., None] ** 1.5
    image = image * (1 - alpha) + np.asarray(to_rgb(colour)) * alpha
# x = -lambda so that lambda grows to the left; reverse the columns to keep x increasing
ax.pcolormesh(-RA[:, ::-1], DEC, image[:, ::-1], shading="auto", antialiased=False, rasterized=True)
ax.grid(color=style.GRID, lw=0.5)
ax.set_xticks(np.radians([-120, -60, 0, 60, 120]))
ax.set_xticklabels([f"{t}°" for t in (120, 60, 0, 300, 240)], fontsize=5, color=style.INK2)
ax.set_yticks(np.radians([-60, -30, 0, 30, 60]))
ax.tick_params(axis="y", labelsize=5, colors=style.INK2)
ax.set_xlabel(r"ecliptic longitude $\lambda$", fontsize=6, labelpad=2)
ax.set_ylabel(r"ecliptic latitude $\beta$", fontsize=6, labelpad=1)
ax.legend(
    handles=[Patch(color=c, alpha=0.8, label=f"{lab} ({SKY_WEIGHTS[k]:.0%})") for k, (lab, c, _) in reversed(SKY_PARTS.items())],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.17),
    ncol=2,
    fontsize=5.5,
    frameon=False,
    handlelength=1.0,
    columnspacing=0.8,
)
style.panel_label(ax, "b", x=0.0, y=1.06)

# (c) orientation -----------------------------------------------------------------------------------------
def draw_orientation(ax, azim=-105, elev=20, psi_deg=30, iota_deg=55, phi0_deg=75):
    """The orbit inclined by iota to the sky plane, psi to its line of nodes, phi0 along it; below the sky plane dashed."""
    psi, iota, phi0 = map(math.radians, (psi_deg, iota_deg, phi0_deg))
    ez = np.array([0, 0, 1.0])
    node = np.array([math.cos(psi), math.sin(psi), 0.0])          # line of nodes, in the sky plane
    in_sky = np.cross(ez, node)                                     # sky-plane direction perpendicular to it
    normal = math.cos(iota) * ez + math.sin(iota) * in_sky          # orbit normal, iota from the line of sight
    in_orbit = np.cross(normal, node)                               # orbit-plane direction perpendicular to the nodes
    if in_orbit[2] < 0:
        normal, in_orbit = normal * [1, 1, 1], -in_orbit  # keep the ascending node at +node
    R, S = 1.0, 1.45

    def arc(u, v, a0, a1, r, n=60):
        a = np.linspace(a0, a1, n)
        return r * (np.cos(a)[:, None] * u + np.sin(a)[:, None] * v)

    # sky plane
    sq = S * np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
    ax.plot_trisurf(sq[:, 0], sq[:, 1], sq[:, 2], color="#e4e4e8", alpha=0.55, shade=False, zorder=0)
    ax.plot(*np.vstack([sq, sq[:1]]).T, color=style.AXIS, lw=0.5, zorder=0.5)
    ax.text(*(np.array([-S, S, 0]) * 0.92), "sky plane", fontsize=5, color=style.INK2, zorder=0.6)
    # the orbit: below the sky plane dashed and drawn first, its plane tinted, above the sky plane solid
    t = np.linspace(0, 2 * math.pi, 361)
    orbit = R * (np.cos(t)[:, None] * node + np.sin(t)[:, None] * in_orbit)
    below = orbit[:, 2] < 0
    ax.plot(*np.where(below[:, None], orbit, np.nan).T, color=style.BLUE, lw=0.9, ls=(0, (2.5, 1.5)), zorder=1)
    disc = np.vstack([np.zeros(3), orbit])
    tri = [[0, i, i + 1] for i in range(1, len(disc) - 1)]
    ax.plot_trisurf(disc[:, 0], disc[:, 1], disc[:, 2], triangles=tri, color=style.BLUE_RAMP[1], alpha=0.35,
                    shade=False, zorder=2)
    ax.plot(*np.where(~below[:, None], orbit, np.nan).T, color=style.BLUE, lw=1.2, zorder=3)
    ax.text(*(R * (math.cos(4.4) * node + math.sin(4.4) * in_orbit) * 1.18), "orbit", fontsize=5.5, color=style.BLUE,
            zorder=3.1)
    # line of nodes: where the two planes meet
    ax.plot(*np.stack([-S * node, S * node]).T, color=style.INK, lw=0.7, zorder=3.2)
    ax.text(*(-S * node * 1.02 + np.array([0, 0, -0.12])), "line of nodes", fontsize=5, color=style.INK2,
            ha="center", zorder=3.3)
    # axes
    for v, lab, length in ((np.array([1.0, 0, 0]), r"$x$", 1.7), (np.array([0, 1.0, 0]), r"$y$", 1.7),
                           (ez, r"$z$, to observer", 1.75)):
        on_top = lab == r"$y$"  # y runs behind L-hat in this view: draw it over, label to its right
        ax.quiver(0, 0, 0, *(length * v), color=style.INK2, lw=0.7, arrow_length_ratio=0.07, zorder=4.5 if on_top else 4)
        nudge = np.array([0.22, 0.0, -0.08]) if on_top else np.zeros(3)
        ax.text(*((length + 0.12) * v + nudge), lab, fontsize=6, color=style.INK2, zorder=4.6)
    ax.quiver(0, 0, 0, *(1.3 * normal), color=style.BLUE, lw=1.0, arrow_length_ratio=0.1, zorder=4.2)
    ax.text(*(1.42 * normal), r"$\hat L$", fontsize=6.5, color=style.BLUE, zorder=4.3)
    # iota between z and L; psi from x to the line of nodes, in the sky plane
    perp = in_sky
    ax.plot(*arc(ez, perp, 0, iota, 0.6).T, color=style.ORANGE, lw=1.0, zorder=5)
    ax.text(*(0.4 * (math.cos(iota / 2) * ez + math.sin(iota / 2) * perp)), r"$\iota$", fontsize=7.5,
            color=style.ORANGE, zorder=5.1)
    ax.plot(*arc(np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), 0, psi, 1.25).T, color=style.ORANGE, lw=1.0, zorder=5)
    ax.text(*(1.38 * np.array([math.cos(psi / 2), math.sin(psi / 2), 0])), r"$\psi$", fontsize=7.5,
            color=style.ORANGE, zorder=5.1)
    # phi0: along the orbit from the ascending node to the binary at the start of the observation
    path = arc(node, in_orbit, 0, phi0, R)
    ax.plot(*path.T, color=style.GREEN, lw=1.6, zorder=6)
    ax.quiver(*path[-6], *(path[-1] - path[-6]), color=style.GREEN, lw=1.6, arrow_length_ratio=1.0, zorder=6)
    ax.scatter(*path[-1], s=12, color=style.INK, zorder=7, depthshade=False)
    ax.text(*(1.2 * R * (math.cos(phi0 / 2) * node + math.sin(phi0 / 2) * in_orbit)), r"$\phi_0$", fontsize=7.5,
            color=style.GREEN, zorder=7.1)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 0.85))
    ax.set(xlim=(-S, S), ylim=(-S, S), zlim=(-1.0, 1.7))
    ax.set_axis_off()


sub_c.set_facecolor("none")
ax = sub_c.add_subplot(projection="3d", computed_zorder=False)
ax.set_facecolor((0, 0, 0, 0))
sub_c.subplots_adjust(left=-0.15, right=1.12, bottom=-0.12, top=1.12)
draw_orientation(ax)
style.panel_label(ax, "c", x=0.1, y=0.93)

style.save(fig, "priors")
