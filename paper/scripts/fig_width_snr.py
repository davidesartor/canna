"""Figure: f0 posterior width against SNR, per source, on the 840 scorecard sources.

The width is 1.4826 x the MAD of the flow's f0 draws for that source, in bins of 1/T_obs.
Points: the final model, every source. Lines: binned medians over found sources (width
< 1 bin) for the uniform-clock baseline, the warped clock and the final model. Black: the
monochromatic Fisher bound sqrt(3) / (pi rho).
"""

import numpy as np

import data
import style
from style import plt

style.setup()
fig, ax = plt.subplots(figsize=(style.COLUMN, 2.35), constrained_layout=True)
edges = np.array([5, 10, 15, 25, 40, 63, 100, 160, 250, 400, 630, 1000, 2000])
centres = np.sqrt(edges[1:] * edges[:-1])

final = data.scorecard("XS-late-768-cool")
ax.scatter(final["snr"], final["width"], s=3.5, color=style.BLUE, alpha=0.35, lw=0, zorder=2)
for run in ["XS-1M", "XS-late", "XS-late-768-cool"]:
    sc = data.scorecard(run)
    found = sc["width"] < 1
    med = [np.median(sc["width"][found & (sc["snr"] >= a) & (sc["snr"] < b)])
           if np.sum(found & (sc["snr"] >= a) & (sc["snr"] < b)) >= 5 else np.nan
           for a, b in zip(edges[:-1], edges[1:])]
    meta = style.MODELS[run]
    ax.plot(centres, med, color=meta["color"], marker="o", ms=3.2, mec="white", mew=0.6,
            lw=1.3, zorder=4, label=meta["label"])
rho = np.logspace(np.log10(5), np.log10(2000), 50)
ax.plot(rho, np.sqrt(3) / (np.pi * rho), color=style.INK, lw=1.0, ls=(0, (4, 2)), zorder=3,
        label=r"ideal, $\sqrt{3}/(\pi\rho)$")
ax.axhline(1.0, color=style.MUTED, lw=0.7)
ax.text(1900, 1.15, "found: width < 1 bin", fontsize=5.8, color=style.INK2, ha="right")
ax.set(xscale="log", yscale="log", xlim=(5, 2000), ylim=(3e-4, 30),
       xlabel=r"source SNR $\rho$", ylabel=r"$f_0$ posterior width [bins of $1/T_{\rm obs}$]")
ax.legend(loc="lower left", fontsize=5.8)
style.save(fig, "width_snr")
