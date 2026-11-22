"""The corner figure of Section 6 (figures/corner-narrow-4-alt.pdf): flow and joint MCMC only, no Fisher.

Added 2026-11-22 (user): same model as corner-narrow-4 (narrow band, S=4, the selected model's eval), but the window
is chosen by how well the flow overlaps the chains, not by the closest-pair rule of import_corners.py, and all four
GBs are shown. Over every window whose four sources all have SNR >= 7 and finite chains, the score is pooled over the
twelve plotted coordinates (f0, ln A, right ascension of each source),

    |median_flow - median_mcmc| / width_mcmc + |ln(width_flow / width_mcmc)|,   width = 1.4826 MAD,

and the lowest score wins (injection 105, SNR 18, 7, 10, 14). Draws are relabelled to the injections by f0 order and
folded as in import_corners.py, and stored as offsets from the folded truth in data/corners/narrow-4-alt4.npz
(paper/data stays local), which a rerun reuses. Run from paper/scripts:

    PYTHONPATH=../../src JAX_PLATFORMS=cpu nice ../../.venv/bin/python fig_corner-alt.py
"""

import json

import numpy as np
import yaml
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

import e64
import style
from style import plt

BAND, N = "narrow", 4
CACHE = style.PAPER / "data/corners" / f"{BAND}-{N}-alt4.npz"
COLUMNS = [0, 2, 3]  # of mcmc.COLUMNS: f0 [bins], log A, lon (equatorial)
NAMES = [r"\Delta f_0", r"\Delta\ln\mathcal{A}", r"\Delta\alpha"]
UNITS = [" [bins]", "", " [rad]"]
MASSES = (0.95, 0.68)
BINS = 36
VISIBLE = 7
SETS = [("mcmc", style.BLUE, True), ("flow", style.ORANGE, False)]


def width(x):
    return 1.4826 * np.median(np.abs(x - np.median(x, axis=0)), axis=0)


def build_cache() -> None:
    from canna.lisa.mcmc import fold
    from canna.lisa.problem import LisaGB
    from canna.lisa.samples import load

    runs = {r["run"]: r for r in json.loads((style.PAPER / "data/runs/source-count.json").read_text())["runs"]}
    run = runs[e64.run_name(BAND, N)]
    ev = run["evals"][-1]["eval"]
    root = style.REPO / "outputs/runs" / run["run"]
    problem = LisaGB(**yaml.safe_load((root / "config.yaml").read_text())["problem"])
    T = problem.t_obs
    d = load(root / "eval" / ev / "draws.npz")
    ref = e64.reference_for(d["truth"])
    ok = np.isfinite(ref["mcmc"]).all(axis=(1, 3))
    snr, f0 = d["source_snr"], d["truth"][..., 0] * T

    def window(j):
        out = {}
        for name, arr in (("flow", d["draws"][j]), ("mcmc", ref["mcmc"][j])):
            cols = [fold(arr[:, s], d["truth"][j, s], T) - fold(d["truth"][j, s][None], d["truth"][j, s], T)[0]
                    for s in range(N)]
            x = np.concatenate(cols, axis=1)
            out[name] = x[np.isfinite(x).all(axis=1)].astype(np.float32)
        return out

    dims = [s * 8 + c for s in range(N) for c in COLUMNS]
    best = None
    for j in range(len(snr)):
        if not (ok[j].all() and snr[j].min() >= VISIBLE):
            continue
        sets = window(j)
        fl, mc = sets["flow"][:, dims], sets["mcmc"][:, dims]
        wm, wf = width(mc), width(fl)
        score = float(np.sum(np.abs(np.median(fl, 0) - np.median(mc, 0)) / wm + np.abs(np.log(wf / wm))))
        print(f"  injection {j:3d}: SNR {np.round(snr[j], 1)}, score {score:.2f}")
        if best is None or score < best[0]:
            best = (score, j, sets)
    score, j, sets = best
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **sets, run=run["run"], eval=ev, injection=j, snr=snr[j], f0_bins=f0[j], score=score)
    print(f"picked injection {j}, score {score:.2f} -> {CACHE.name}")


def credible_levels(density, masses):
    ordered = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    return sorted({ordered[min(np.searchsorted(cumulative, m), len(ordered) - 1)] for m in masses})


def contours(ax, x, y, extent, filled, color):
    # 1024 draws per set: coarser 2-D bins and more smoothing than the 1-D histograms, or the 95% contour frays
    hist, xe, ye = np.histogram2d(x, y, bins=22, range=extent)
    density = gaussian_filter(hist.T, 1.4)
    if density.max() <= 0:
        return
    xc, yc = (xe[1:] + xe[:-1]) / 2, (ye[1:] + ye[:-1]) / 2
    levels = credible_levels(density, MASSES)
    if len(levels) < 2:
        return
    if filled:
        ax.contourf(xc, yc, density, levels=[*levels, density.max() * 1.01],
                    colors=[to_rgba(color, 0.3), to_rgba(color, 0.75)])
    else:
        ax.contour(xc, yc, density, levels=levels, colors=color, linewidths=[0.55, 0.95])


def extent_of(sets, c):
    lo = min(np.quantile(s[:, c], 0.02) for s in sets)
    hi = max(np.quantile(s[:, c], 0.98) for s in sets)
    pad = 0.05 * (hi - lo) or 1e-6
    return lo - pad, hi + pad


POOLED = style.PAPER / "data/corners" / f"{BAND}-{N}-pooled.npz"
PHYS_NAMES = [r"$f_0 T_{\rm obs} - k_w$ [bins]", r"$\log_{10}\dot f$", r"$\log_{10}\mathcal{A}$", r"$\lambda$ [rad]",
              r"$\beta$ [rad]", r"$\psi$ [rad]", r"$\iota$ [rad]", r"$\phi_0$ [rad]"]


def build_pooled() -> None:
    """2026-11-22 (user): the corner report's pooled view of the chosen window, all four sources in one 8x8 corner, in
    its physical columns (canna.lisa.plots.physical_columns: f0 from the window start, ecliptic sky, psi/phi0 folded)."""
    from canna.lisa.plots import fold_orientation, physical_columns
    from canna.lisa.problem import LisaGB, window_start
    from canna.lisa.samples import load

    j = int(np.load(CACHE)["injection"])
    runs = {r["run"]: r for r in json.loads((style.PAPER / "data/runs/source-count.json").read_text())["runs"]}
    run = runs[e64.run_name(BAND, N)]
    ev = run["evals"][-1]["eval"]
    root = style.REPO / "outputs/runs" / run["run"]
    problem = LisaGB(**yaml.safe_load((root / "config.yaml").read_text())["problem"])
    d = load(root / "eval" / ev / "draws.npz")
    ref = e64.reference_for(d["truth"])
    truth, f = d["truth"][j], d["f"][j]
    start = float(window_start(problem, f))
    out = {"truth": physical_columns(truth[None], truth, start, problem.t_obs)[0]}
    for name, arr in (("flow", d["draws"][j]), ("mcmc", ref["mcmc"][j])):
        x = physical_columns(fold_orientation(arr, truth), truth, start, problem.t_obs)  # [N, S, 8]
        out[name] = x[np.isfinite(x).all(axis=(1, 2))].astype(np.float32)
    np.savez(POOLED, **out, injection=j, snr=d["source_snr"][j], run=run["run"], eval=ev)


if not CACHE.is_file():
    build_cache()
if not POOLED.is_file():
    build_pooled()
d = np.load(POOLED)
# pooled: every source's draws in one set per method
sets = {name: d[name].reshape(-1, 8) for name, *_ in SETS}
truth = d["truth"]
k = 8


def pooled_extent(c):
    """The central 99% of flow and chains together and every truth, padded 5%."""
    lo = min(min(np.quantile(s[:, c], 0.005) for s in sets.values()), truth[:, c].min())
    hi = max(max(np.quantile(s[:, c], 0.995) for s in sets.values()), truth[:, c].max())
    pad = 0.05 * (hi - lo) or 1e-6
    return lo - pad, hi + pad


def pooled_contours(ax, x, y, extent, filled, color):
    hist, xe, ye = np.histogram2d(x, y, bins=44, range=extent)
    density = gaussian_filter(hist.T, 0.9)
    if density.max() <= 0:
        return
    xc, yc = (xe[1:] + xe[:-1]) / 2, (ye[1:] + ye[:-1]) / 2
    levels = credible_levels(density, MASSES)
    if len(levels) < 2:
        return
    if filled:
        ax.contourf(xc, yc, density, levels=[*levels, density.max() * 1.01],
                    colors=[to_rgba(color, 0.3), to_rgba(color, 0.75)])
    else:
        ax.contour(xc, yc, density, levels=levels, colors=color, linewidths=[0.55, 0.95])


style.setup()
fig, axes = plt.subplots(k, k, figsize=(style.FULL, style.FULL * 0.92), layout="constrained")
ext = [pooled_extent(c) for c in range(k)]
for r in range(k):
    for c in range(k):
        ax = axes[r, c]
        if c > r:
            ax.axis("off")
            continue
        if r == c:
            for name, color, filled in SETS:
                ax.hist(sets[name][:, c], bins=60, range=ext[c], density=True, histtype="step", color=color, lw=0.8)
                if filled:
                    ax.hist(sets[name][:, c], bins=60, range=ext[c], density=True, color=to_rgba(color, 0.2))
            for t in truth[:, c]:
                ax.axvline(t, color=style.INK, lw=0.5, ls=(0, (2, 1.5)), zorder=5)
            ax.set_yticks([])
        else:
            for name, color, filled in SETS:
                pooled_contours(ax, sets[name][:, c], sets[name][:, r], [ext[c], ext[r]], filled, color)
            ax.plot(truth[:, c], truth[:, r], "x", color=style.INK, ms=3, mew=0.7, zorder=5)
            ax.set_ylim(ext[r])
        ax.set_xlim(ext[c])
        ax.tick_params(labelsize=4.5, length=1.5, pad=1)
        if r < k - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(PHYS_NAMES[c], fontsize=6.5)
        if c > 0 or r == 0:
            ax.set_yticklabels([]) if r != c else None
        else:
            ax.set_ylabel(PHYS_NAMES[r], fontsize=6.5)
snr = d["snr"]
fig.legend(
    handles=[*(Line2D([], [], color=color, lw=1.0) for _, color, _ in SETS),
             Line2D([], [], color=style.INK, marker="x", ls="none", ms=3.5, mew=0.8)],
    labels=["joint MCMC", "flow", "truths"], loc="upper right", frameon=False, fontsize=7,
    title=(f"{BAND} band, S = {N}: injection {int(d['injection'])}, four sources pooled\n"
           f"at SNR {', '.join(f'{s:.0f}' for s in snr)}"),
    title_fontsize=7, bbox_to_anchor=(0.98, 0.98),
)
style.save(fig, f"corner-{BAND}-{N}-alt")
