"""Figure: the LISA band as the training prior sees it.

(a) the TDI-1.5 A/E and T noise PSDs that whiten the data (fractional frequency);
(b) the amplitude prior: the band between matched-filter SNR 7 and 1000 at each frequency;
(c) why the rungs stop where they do: the response span (annual Doppler sideband plus the
    chirp drift over T_obs) against the half-window each rung can hold.
"""

import numpy as np

import data  # noqa: F401  (sets JAX_PLATFORMS=cpu before jax is imported)
import style
from style import plt

import jax.numpy as jnp
from canna.lisa import LisaGB
from canna.lisa.constants import EARTH_ORBIT_SPEED, SPEED_OF_LIGHT
from canna.lisa.priors import fdot_from_chirp_mass
from canna.lisa.problem import max_f0

style.setup()
T_OBS = 63115200.0
problem = LisaGB(t_obs=T_OBS, f0_range=(1e-4, 12e-3))  # B: the whole band, auto-sized window
f = np.logspace(np.log10(1e-4), np.log10(2e-2), 600)

fig, axes = plt.subplots(1, 3, figsize=(style.FULL, 2.05), constrained_layout=True)

# (a) noise PSDs
ax = axes[0]
psd = np.asarray(problem.noise_psd(jnp.asarray(f)))
ax.loglog(f * 1e3, np.sqrt(psd[:, 0]), color=style.BLUE, label="A, E")
ax.loglog(f * 1e3, np.sqrt(psd[:, 2]), color=style.ORANGE, label="T")
ax.set(xlabel="frequency [mHz]", ylabel=r"$\sqrt{S_c(f)}$  [Hz$^{-1/2}$]")
ax.legend(loc="lower right")
style.panel_label(ax, "a")

# (b) amplitude prior band
ax = axes[1]
lo, hi = (np.asarray(v) for v in problem.a_window(jnp.asarray(f)))
mid = np.sqrt(lo * hi)
ax.fill_between(f * 1e3, lo, hi, color=style.BLUE, alpha=0.12, lw=0)
ax.loglog(f * 1e3, lo, color=style.BLUE, lw=1.0)
ax.loglog(f * 1e3, hi, color=style.BLUE, lw=1.0)
ax.loglog(f * 1e3, mid, color=style.BLUE, lw=0.8, ls=(0, (3, 2)))
i = np.searchsorted(f, 11e-3)
for curve, text in [(hi, r"$\rho=1000$"), (mid, r"$\rho=\sqrt{7000}$ (median)"), (lo, r"$\rho=7$")]:
    ax.text(11, curve[i] * 1.9, text, color=style.INK2, fontsize=6.2, ha="center")
# the faint source of Strub et al. 2022, for scale (SNR 7.93 at 1.40 mHz)
ax.plot([1.40457], [4.55e-23], marker="o", ms=4, color=style.ORANGE, mec="white", mew=0.8, ls="")
ax.annotate("Strub+22\nfaint source", (1.40457, 4.55e-23), xytext=(0.25, 1.2e-23),
            fontsize=6, color=style.INK2, arrowprops=dict(arrowstyle="-", color=style.MUTED, lw=0.5))
ax.set(xlabel="frequency [mHz]", ylabel="amplitude $\\mathcal{A}$", ylim=(2e-24, 2e-18))
style.panel_label(ax, "b")

# (c) response span against each rung's half-window
ax = axes[2]
doppler = f * EARTH_ORBIT_SPEED / SPEED_OF_LIGHT * T_OBS
for mc, color, name in [(1.2188, style.ORANGE, r"$\mathcal{M}=1.22\,M_\odot$ (max)"),
                        (0.45, style.BLUE, r"$\mathcal{M}=0.45\,M_\odot$ (mode)")]:
    drift = np.asarray(fdot_from_chirp_mass(mc, f)) * T_OBS**2
    ax.loglog(f * 1e3, doppler + drift, color=color, label=name)
ax.loglog(f * 1e3, doppler, color=style.MUTED, lw=0.9, ls=(0, (3, 2)), label="Doppler alone")
for rp, name in [(16, "XS"), (64, "S"), (1024, "B")]:
    fmax = max_f0(rp, 1.4 * 2**-0.2, T_OBS) * 1e3  # where the heaviest binary fills the window
    ax.axhline(rp / 2, color=style.AXIS, lw=0.8)
    ax.text(0.105, rp / 2 * 1.15, f"{name}: {rp // 2} bins", fontsize=6, color=style.INK2)
    ax.plot([fmax], [rp / 2], marker="o", ms=3.5, color=style.INK, ls="")
ax.set(xlabel="frequency [mHz]", ylabel="half-span of the response [bins]", ylim=(0.3, 3e3))
ax.legend(loc="lower right", fontsize=6)
style.panel_label(ax, "c")

for ax in axes:
    ax.set_xlim(0.1, 20)
style.save(fig, "band")
