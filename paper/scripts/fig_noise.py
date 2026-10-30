"""The instrument noise: the A/E and T spectra with the two terms of A, and the sky-averaged SNR constant C over the band."""

import json

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import style
from canna.lisa import config
from canna.lisa.problem import LisaGB, noise_psd

style.setup()
full_cfg = config.load("full-B").problem
narrow = LisaGB(**config.load("narrow-B").problem)
full = LisaGB(**full_cfg)
f = np.geomspace(5e-5, 2e-2, 600)
f_mhz = f * 1e3

# each noise term alone, by switching the other off
psd = np.asarray(noise_psd(full, jnp.asarray(f)))
psd_oms = np.asarray(
    noise_psd(LisaGB(**(full_cfg | dict(acceleration_noise=0.0))), jnp.asarray(f))
)[:, 0]
psd_acc = np.asarray(
    noise_psd(LisaGB(**(full_cfg | dict(oms_noise=0.0))), jnp.asarray(f))
)[:, 0]

record = json.loads(
    (
        style.REPO / "reports/prior/f0-fdot-amplitude/data/f0-fdot-amplitude.json"
    ).read_text()
)
sky_average = record["bands"]["full"]["sky_average"]

fig, (ax_psd, ax_c) = plt.subplots(
    1, 2, figsize=(style.FULL, 2.3), layout="constrained", width_ratios=[1.25, 1]
)

# (a) amplitude spectral densities over the two bands
ax_psd.axvspan(*(np.array(full.f0_range) * 1e3), color=style.FILL, lw=0, zorder=0)
ax_psd.axvspan(
    *(np.array(narrow.f0_range) * 1e3), color=style.BLUE_RAMP[0], lw=0, zorder=0
)
ax_psd.loglog(f_mhz, np.sqrt(psd_oms), **style.GUIDE)
ax_psd.loglog(f_mhz, np.sqrt(psd_acc), color=style.MUTED, lw=0.75, ls=(0, (1, 1.5)))
ax_psd.loglog(f_mhz, np.sqrt(psd[:, 0]), color=style.BLUE)
ax_psd.loglog(f_mhz, np.sqrt(psd[:, 2]), color=style.ORANGE)
at = np.searchsorted(f, [1.5e-4, 6e-3, 1.5e-3, 2.2e-3])
style.direct_label(ax_psd, f_mhz[at[0]], np.sqrt(psd[at[0], 0]), "$A$, $E$", dx=4, dy=2)
style.direct_label(ax_psd, f_mhz[at[1]], np.sqrt(psd[at[1], 2]), "$T$", dx=4, dy=-2)
style.direct_label(ax_psd, f_mhz[at[2]], np.sqrt(psd_oms[at[2]]), "OMS", dx=0, dy=-8)
style.direct_label(
    ax_psd,
    f_mhz[at[3]],
    np.sqrt(psd_acc[at[3]]),
    "acceleration",
    dx=0,
    dy=-6,
    ha="center",
)
for left, name in ((0.105, "narrow"), (2.0, "full band")):
    ax_psd.text(
        left,
        0.96,
        name,
        transform=ax_psd.get_xaxis_transform(),
        va="top",
        color=style.INK2,
    )
ax_psd.set(
    xlabel=r"$f$ [mHz]",
    ylabel=r"$\sqrt{S_c(f)}$ [Hz$^{-1/2}$]",
    xlim=(f_mhz[0], f_mhz[-1]),
    ylim=(np.sqrt(psd[:, 0]).min() / 3, np.sqrt(psd[:, 2]).max()),
)
ax_psd.set_title("Noise spectra in fractional frequency")
style.panel_label(ax_psd, "a")

# (b) the measured sky-averaged constant C(f0), isotropic and galactic sky
f0_mhz = np.array(sky_average["f0"]) * 1e3
ax_c.axvspan(
    sky_average["flat_below"] * 1e3, f0_mhz[-1], color=style.FILL, lw=0, zorder=0
)
for sky, color, name in (
    ("isotropic", style.BLUE, "isotropic sky"),
    ("galactic", style.ORANGE, "galactic sky"),
):
    c, se = np.array(sky_average[sky]["C"]), np.array(sky_average[sky]["C_se"])
    ax_c.fill_between(f0_mhz, c - se, c + se, color=color, alpha=0.18, lw=0)
    ax_c.plot(f0_mhz, c, color=color, marker="o", ms=2.4, mew=0)
    style.direct_label(ax_c, f0_mhz[0], c[0], name, dx=0, dy=6, ha="left")
ax_c.set(
    xscale="log",
    xlabel=r"$f_0$ [mHz]",
    ylabel=r"$C = \mathrm{E}[\rho^2]\,S_n / (\mathcal{A}^2 T_{\rm obs})$",
)
style.direct_label(
    ax_c,
    sky_average["flat_below"] * 1e3,
    0.97,
    "$1 + 0.6\\,(f/f_\\star)^2$\napproximate",
    dx=3,
    va="top",
    xycoords=ax_c.get_xaxis_transform(),
)
ax_c.set_title("Sky-averaged SNR constant")
style.panel_label(ax_c, "b")

style.save(fig, "noise")
