"""Figure: one XS window, from the data to the posterior.

(a) the whitened data power per frequency bin (A + E) with the noiseless signal, and the
    four true frequencies; (b) the network's conditioning input, the arcsinh of the whitened
    WDM coefficients (2 time rows x 64 channels x A/E/T); (c) the flow's f0 posterior for
    each source (relabelled onto it) against the Fisher forecast, around the truth.

Uses eval injection q0.20 (index 1): sources at SNR 191, 5.3, 24 and 32, so it shows a
loud, two moderate and one undetectable source. Final model (XS-late-768-cool).
"""

import numpy as np

import data
import style
from style import plt
from matplotlib.colors import LinearSegmentedColormap

import jax.numpy as jnp
import jax.random as jr

J = 1
style.setup()
problem, _ = data.xs_problem()
inj = data.eval_injections()
f, truth = float(inj["window"][J]), inj["truth"][J]
key = jr.wrap_key_data(jnp.asarray(inj["noise_key"][J]))
t_obs = problem.t_obs

o = problem.sample_observation(key, jnp.asarray(truth), f)
clean = problem.clean_signal(jnp.asarray(truth), f)
power = np.asarray(problem.noise_psd(problem.window_freqs(f)) * t_obs / 2)
white = np.abs(np.asarray(o)) ** 2 / power
white_clean = np.abs(np.asarray(clean)) ** 2 / power
y = np.asarray(problem.preprocess(o, f))  # (2, 64, 3)
start = int(problem.window_start(f))
lo, hi = (float(v) * t_obs - start for v in problem.f0_window(f))
bins = np.arange(problem.window_bins)
f0_bin = truth[:, 0] * t_obs - start
snr = np.asarray(problem.snr(jnp.asarray(truth)[:, None, :], f))

matched = data.matched_draws(problem, data.final_draws(J), truth, f)
fisher = data.fisher_draws(J)

fig = plt.figure(figsize=(style.FULL, 3.9))
outer = fig.add_gridspec(3, 1, height_ratios=[1.3, 0.62, 1.05], hspace=0.78)

# (a) whitened power, summed over A and E
ax = fig.add_subplot(outer[0])
ax.axvspan(-0.5, lo, color=style.GRID, alpha=0.6, lw=0)
ax.axvspan(hi, problem.window_bins - 0.5, color=style.GRID, alpha=0.6, lw=0)
ax.bar(bins, white[:, :2].sum(-1), width=0.8, color=style.MUTED, alpha=0.55, lw=0, label="data")
ax.step(bins, white_clean[:, :2].sum(-1), where="mid", color=style.BLUE, lw=1.0, label="noiseless signal")
ax.set_yscale("log")
top = white[:, :2].sum(-1).max()
ax.set_ylim(0.05, top * 60)
order = np.argsort(f0_bin)
for rank, s in enumerate(order):
    # alternate two label heights so neighbouring sources do not collide
    height = top * (3 if rank % 2 == 0 else 18)
    ax.plot([f0_bin[s], f0_bin[s]], [0.05, height * 0.8], color=style.INK, lw=0.6)
    ax.text(f0_bin[s], height, rf"$\rho={snr[s]:.1f}$", ha="center", va="bottom",
            fontsize=6, color=style.INK)
ax.text(lo / 2, top * 8, "guard", ha="center", fontsize=5.8, color=style.INK2)
ax.text((hi + problem.window_bins) / 2, top * 8, "guard", ha="center", fontsize=5.8, color=style.INK2)
ax.set(xlim=(-0.5, problem.window_bins - 0.5), ylabel="whitened power, A + E")
ax.set_xlabel(f"frequency bin in the window (bin 0 = {start / t_obs * 1e3:.4f} mHz, width 1/T = 15.8 nHz)")
ax.legend(loc="upper center", bbox_to_anchor=(0.6, 1.0), ncol=2)
ax.grid(axis="x", visible=False)
style.panel_label(ax, "a", y=1.0)

# (b) the WDM conditioning image, one strip per TDI channel
inner = outer[1].subgridspec(3, 1, hspace=0.12)
diverging = LinearSegmentedColormap.from_list(
    "bgr", [style.BLUE_RAMP[5], style.BLUE_RAMP[2], "#f0efec", "#f3a58a", "#c4351f"])
vmax = np.abs(y).max()
for c, name in enumerate("AET"):
    ax = fig.add_subplot(inner[c])
    im = ax.imshow(y[:, :, c], cmap=diverging, vmin=-vmax, vmax=vmax, aspect="auto",
                   extent=(-0.5, problem.window_bins - 0.5, 1.5, -0.5), interpolation="nearest")
    ax.set_yticks([])
    ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=6.5, labelpad=4)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if c < 2:
        ax.set_xticks([])
    else:
        ax.set_xlabel("WDM channel (one per frequency bin), 2 time rows each")
    if c == 0:
        style.panel_label(ax, "b", y=1.25)
cax = fig.add_axes([0.915, 0.405, 0.008, 0.12])
cb = fig.colorbar(im, cax=cax)
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=5.5, length=1.5)
cb.set_label("arcsinh", fontsize=5.8)

# (c) per-source f0 posterior, flow against Fisher
inner = outer[2].subgridspec(1, 4, wspace=0.32)
for k, s in enumerate(order):
    ax = fig.add_subplot(inner[k])
    d_flow = matched[:, s, 0] * t_obs - start - f0_bin[s]
    d_fish = fisher[:, s, 0] * t_obs - start - f0_bin[s]
    width = 1.4826 * np.median(np.abs(d_flow - np.median(d_flow)))
    if width < 1.0:
        half = max(5 * np.std(d_fish), 4 * width)
        a, b = -half, half
    else:  # not found: show the whole window
        a, b = -f0_bin[s] - 0.5, problem.window_bins - 0.5 - f0_bin[s]
    edges = np.linspace(a, b, 49)
    # each histogram scaled to a unit peak: the Fisher spike would otherwise flatten the flow
    h_flow = np.histogram(d_flow, edges)[0].astype(float)
    h_fish = np.histogram(d_fish, edges)[0].astype(float)
    ax.stairs(h_flow / h_flow.max(), edges, fill=True, color=style.BLUE, alpha=0.75, lw=0, label="flow")
    ax.stairs(h_fish / h_fish.max(), edges, color=style.MUTED, lw=1.0, label="Fisher")
    ax.set_ylim(0, 1.45)
    ax.axvline(0, color=style.INK, lw=0.7)
    out = np.mean((d_flow < a) | (d_flow > b))
    if out > 0.005:
        ax.text(0.99, 0.99, f"{out:.0%} outside", transform=ax.transAxes,
                ha="right", va="top", fontsize=5.5, color=style.INK2)
    if width >= 1.0:
        ax.text(0.02, 0.99, "not found: draws sit\nnear the loud sources", transform=ax.transAxes,
                ha="left", va="top", fontsize=5.5, color=style.INK2)
    ax.set_yticks([])
    ax.set_xlim(a, b)
    ax.set_title(rf"$\rho={snr[s]:.1f}$", fontsize=6.5, pad=2)
    ax.set_xlabel(r"$f_0-f_0^{\rm true}$ [bins]", fontsize=6.5)
    if k == 0:
        style.panel_label(ax, "c", y=1.05)
        ax.legend(loc="upper left", fontsize=5.8, handlelength=1.0, borderaxespad=0.1)
style.save(fig, "window")
