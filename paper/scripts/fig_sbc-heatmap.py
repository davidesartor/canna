"""figures/sbc-heatmap.pdf: SBC p_min per (f0, SNR) cell, one heatmap per reported model, from data/runs/sbc-heatmap.json
(import_sbc_heatmap.py)."""

import json
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import style

DATA = json.loads((style.PAPER / "data/runs/sbc-heatmap.json").read_text())
ORDER = ["narrow-2", "narrow-4", "narrow-8", "full-4", "full-8", "full-16"]
FLOOR = -3  # gamma_test simulates 1000 nulls: p resolves down to 0.001
CMAP = LinearSegmentedColormap.from_list("p", [style.RED, "#f6c9c8", "#ffffff"])


def mhz(v: float) -> str:
    return f"{v:.2g}"


def pfmt(p: float) -> str:
    """Two significant digits; below the test's resolution, < 0.001."""
    return "<0.001" if p < 1e-3 else f"{p:.2g}"


def snr_labels(edges: list) -> list[str]:
    """The SNR ranges, open at both ends: the lowest and highest edges are just the sample's extremes."""
    inner = [f"{e:.2g}" for e in edges[1:-1]]
    return [f"<{inner[0]}", *[f"{a}–{b}" for a, b in zip(inner, inner[1:])], f">{inner[-1]}"]


style.setup()
fig, axes = plt.subplots(2, 3, figsize=(style.FULL, 4.3), layout="constrained")
for ax, key in zip(axes.flat, ORDER):
    d = DATA[key]
    band, n = key.split("-")
    p = np.array([[np.nan if v is None else v for v in row] for row in d["p_min"]])
    count = np.array(d["count"])
    logp = np.clip(np.log10(np.maximum(p, 1e-6)), FLOOR, 0)
    image = ax.imshow(logp, origin="lower", cmap=CMAP, vmin=FLOOR, vmax=0, aspect="auto")
    for a in range(p.shape[0]):
        for b in range(p.shape[1]):
            if np.isnan(p[a, b]):
                ax.text(b, a, f"–\n({count[a, b]})", ha="center", va="center", fontsize=5, color=style.MUTED)
                continue
            dark = logp[a, b] < -2.5
            ax.text(
                b, a, f"{pfmt(p[a, b])}\n({count[a, b]})", ha="center", va="center", fontsize=5.2,
                color="white" if dark else style.INK,
            )
    f_edges, s_edges = d["f_edges_mhz"], d["snr_edges"]
    ax.set_yticks(np.arange(5), [f"{mhz(f_edges[i])}–{mhz(f_edges[i + 1])}" for i in range(5)], fontsize=5.5)
    ax.set_xticks(np.arange(5), snr_labels(s_edges), fontsize=5.5)
    ax.set_title(rf"{band}, $S = {n}$", fontsize=7)
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
for ax in axes[:, 0]:
    ax.set_ylabel(r"$f_0$ [mHz], draws' median", fontsize=6)
for ax in axes[1]:
    ax.set_xlabel(r"SNR, draws' median", fontsize=6)
bar = fig.colorbar(image, ax=axes, shrink=0.6, pad=0.01)
bar.set_ticks(list(range(FLOOR, 1)), labels=[rf"$10^{{{t}}}$" if t < 0 else "1" for t in range(FLOOR, 1)])
bar.ax.tick_params(labelsize=5.5)
bar.set_label(r"$p_{\min}$ over the eight coordinates", fontsize=6)
bar.outline.set_visible(False)
for ax, letter in zip(axes.flat, "abcdef"):
    style.panel_label(ax, letter)
style.save(fig, "sbc-heatmap")
