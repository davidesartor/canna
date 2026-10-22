"""Every lisa figure, drawn from saved data: a run's losses and λ, an eval dir's corner and validation pages."""

import sys
from pathlib import Path

import numpy as np
from einops import rearrange

PARAM_LABELS = [
    "f_0",
    "\\mathcal{M}",
    "A",
    "\\alpha",
    "\\delta",
    "\\psi",
    "\\iota",
    "\\phi_0",
]
PARAM_IS_LOG = [True, False, True, False, False, False, False, False]


ROBUST_WIDTHS = 8


def axis_ranges(
    flow: np.ndarray,
    fisher: np.ndarray,
    scale: list[str],
    mcmc: np.ndarray | None = None,
    robust: bool = True,
) -> list[tuple]:
    """Each axis' range: the central 99% of flow and MCMC (else Fisher) together, padded 5%.

    With `robust`, the flow's part is held to its median +- ROBUST_WIDTHS robust widths, so a
    percent of strays a bin away cannot squash a narrow posterior into one histogram bin.
    """
    ranges = []
    for c in range(flow.shape[1]):
        sets = [flow[:, c], mcmc[:, c] if mcmc is not None else fisher[:, c]]
        if scale[c] == "log":
            # a linear-space Fisher can spill below zero on a log axis -- ignore those
            sets = [np.log10(x[x > 0]) for x in sets]
        bounds = [np.quantile(x, [0.005, 0.995]) for x in sets if x.size]
        if robust and sets[0].size:
            med = np.median(sets[0])
            width = 1.4826 * np.median(np.abs(sets[0] - med))
            if width > 0:
                b = bounds[0]
                bounds[0] = np.array(
                    [
                        max(b[0], med - ROBUST_WIDTHS * width),
                        min(b[1], med + ROBUST_WIDTHS * width),
                    ]
                )
        lo, hi = min(b[0] for b in bounds), max(b[1] for b in bounds)
        f = fisher[:, c]
        f = np.log10(f[f > 0]) if scale[c] == "log" else f
        if f.size and not np.any((f >= lo) & (f <= hi)):
            # a Fisher wholly off the page would make its density histogram 0/0: take it in
            fb = np.quantile(f, [0.005, 0.995])
            lo, hi = min(lo, fb[0]), max(hi, fb[1])
        m = 0.05 * (hi - lo) or 0.1
        lo, hi = lo - m, hi + m
        ranges.append((10**lo, 10**hi) if scale[c] == "log" else (lo, hi))
    return ranges


def corner_page(
    path: Path,
    flow: np.ndarray,
    fisher: np.ndarray,
    labels: list[str],
    scale: list[str],
    title: str,
    truths: np.ndarray,
    pooled: bool = False,
    bins: int = 20,
    mcmc: np.ndarray | None = None,
    save_samples: bool = True,
) -> None:
    """One corner page: flow (orange), Fisher (blue, dashed), MCMC (green), truths [k, D] (black); samples beside as .npz."""
    import corner
    import matplotlib.pyplot as plt

    if save_samples:
        extra = {} if mcmc is None else {"mcmc": mcmc}
        np.savez_compressed(
            path.with_suffix(".npz"),
            flow=flow,
            fisher=fisher,
            truths=truths,
            labels=np.array(labels),
            scale=np.array(scale),
            title=np.array(title),
            pooled=pooled,
            bins=bins,
            **extra,
        )
    ranges = axis_ranges(flow, fisher, scale, mcmc, robust=not pooled)
    # draws with any coordinate off the page, so the cut is never silent
    lo_hi = np.array(ranges)
    off = np.mean(np.any((flow < lo_hi[:, 0]) | (flow > lo_hi[:, 1]), axis=1))
    if off > 0:
        title = f"{title}\n{100 * off:.1f}% of the flow draws lie outside the plotted ranges"
    # a log axis over less than a factor of 2 (f0 inside one window) is drawn linear
    scale = [
        "linear" if sc == "log" and hi < 2 * lo else sc
        for sc, (lo, hi) in zip(scale, ranges)
    ]
    style = dict(range=ranges, axes_scale=scale, bins=bins)
    # corner writes its color into hist_kwargs, so each call gets its own dict
    fig = corner.corner(
        flow,
        labels=labels,
        color="C1",
        show_titles=not pooled,
        title_fmt=".2g",
        hist_kwargs={"density": True},
        **style,
    )
    corner.corner(
        fisher,
        fig=fig,
        color="C0",
        plot_datapoints=False,
        contour_kwargs={"linestyles": "dashed"},
        hist_kwargs={"density": True},
        **style,
    )
    if mcmc is not None:
        corner.corner(
            mcmc,
            fig=fig,
            color="C2",
            plot_datapoints=False,
            hist_kwargs={"density": True},
            **style,
        )
    if pooled:
        ndim = flow.shape[1]
        axes = np.array(fig.axes).reshape(ndim, ndim)
        for i in range(ndim):
            for t in truths:
                axes[i, i].axvline(t[i], color="black", lw=0.8)
        corner.overplot_points(fig, truths, marker="s", ms=3, color="black")
    else:
        for t in truths:
            corner.overplot_lines(fig, t, color="black")
    fig.suptitle(title, y=1.0)
    for ax in fig.axes:
        ax.set_rasterized(True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def replot_page(samples: Path, path: Path | None = None) -> Path:
    """Redraw a corner page from its saved samples (default: the .pdf beside the .npz)."""
    d = np.load(samples)
    path = samples.with_suffix(".pdf") if path is None else path
    corner_page(
        path,
        d["flow"],
        d["fisher"],
        list(d["labels"]),
        list(d["scale"]),
        str(d["title"]),
        d["truths"],
        pooled=bool(d["pooled"]),
        bins=int(d["bins"]),
        mcmc=d["mcmc"] if "mcmc" in d else None,
        save_samples=False,
    )
    return path


def gb_picks(source_snr: np.ndarray) -> dict[str, tuple[int, int]]:
    """(injection, source) of the loudest, median and faintest eval source, from SNRs [J, S]."""
    order = sorted(np.ndindex(source_snr.shape), key=lambda js: source_snr[js])
    return {
        "loudest": order[-1],
        "median": order[len(order) // 2],
        "faintest": order[0],
    }


def fold_orientation(p: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Fold draws [..., 8] onto the (psi, phi0) copy nearest the truth: psi within pi/4, phi0 pi.

    By the likelihood's exact symmetries psi -> psi + pi and (psi, phi0) -> (psi + pi/2, phi0 + pi).
    """
    p = np.array(p, np.float64)
    k = np.round((p[..., 5] - truth[..., 5]) / (np.pi / 2))
    p[..., 5] -= k * np.pi / 2
    p[..., 7] = (
        truth[..., 7]
        + (p[..., 7] - k * np.pi - truth[..., 7] + np.pi) % (2 * np.pi)
        - np.pi
    )
    return p


def injection_pages(
    corner_dir: Path,
    tag: str,
    quantile: float,
    snr: float,
    draws: np.ndarray,
    fisher: np.ndarray,
    truth: np.ndarray,
    source_snr: np.ndarray,
    mcmc: np.ndarray | None = None,
) -> None:
    """The pooled page and one page per source of an injection: draws matched onto truth [S, 8] (loudest first), fisher [N, S, 8], mcmc [M, S, 8]."""
    n_sources = truth.shape[0]
    draws, fisher = fold_orientation(draws, truth), fold_orientation(fisher, truth)
    mcmc = None if mcmc is None else fold_orientation(mcmc, truth)
    labels = [f"${label}$" for label in PARAM_LABELS]
    scale = ["log" if is_log else "linear" for is_log in PARAM_IS_LOG]
    with_mcmc = ", MCMC (green)" if mcmc is not None else ""
    # one page per source, by SNR rank
    for s in range(n_sources):
        corner_page(
            corner_dir / f"{tag}-src{s + 1}.pdf",
            draws[:, s],
            fisher[:, s],
            labels,
            scale,
            f"source {s + 1} of {n_sources} by SNR (SNR={source_snr[s]:.1f}),"
            f" flow (orange) vs Fisher (blue){with_mcmc},"
            f" SNR quantile {quantile:.1f}, total SNR={snr:.1f}",
            truth[s][None],
            mcmc=None if mcmc is None else mcmc[:, s],
        )
    # pooled page: every source in the same marginals, labels dropped
    corner_page(
        corner_dir / f"{tag}-pooled.pdf",
        rearrange(draws, "n s p -> (n s) p"),
        rearrange(fisher, "n s p -> (n s) p"),
        labels,
        scale,
        f"all {n_sources} sources pooled, flow (orange) vs Fisher (blue){with_mcmc},"
        f" SNR quantile {quantile:.1f}, SNR={snr:.1f}",
        truth,
        pooled=True,
        bins=50,
        mcmc=None if mcmc is None else rearrange(mcmc, "n s p -> (n s) p"),
    )


def gb_page(
    corner_dir: Path,
    name: str,
    n_eval_sources: int,
    snr: float,
    tag: str,
    s: int,
    flow_s: np.ndarray,
    fisher_s: np.ndarray,
    truth_s: np.ndarray,
    mcmc_s: np.ndarray | None = None,
) -> None:
    """One per-GB page: a single source's draws, relabelled onto it, against its Fisher."""
    with_mcmc = ", MCMC (green)" if mcmc_s is not None else ""
    flow_s, fisher_s = (
        fold_orientation(flow_s, truth_s),
        fold_orientation(fisher_s, truth_s),
    )
    mcmc_s = None if mcmc_s is None else fold_orientation(mcmc_s, truth_s)
    corner_page(
        corner_dir / f"gb-{name}.pdf",
        flow_s,
        fisher_s,
        [f"${label}$" for label in PARAM_LABELS],
        ["log" if is_log else "linear" for is_log in PARAM_IS_LOG],
        f"{name} of the {n_eval_sources} eval sources: SNR {snr:.1f}"
        f" (injection {tag}, source {s + 1}), flow (orange) vs Fisher (blue){with_mcmc}",
        truth_s[None],
        mcmc=mcmc_s,
    )


def ecdf_pages(rows: dict, path) -> None:
    """ECDF - z of every rank set with its simultaneous band: one page per stratum, quantities by slot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    from .validate import ALPHA, SNR_BANDS, gamma_test

    k = int(rows["draws"]) + 1
    z = np.arange(1, k) / k
    n_sources = rows["snr_hat"].shape[1]
    names = list(rows["rank"]) + ["log_likelihood"]
    with PdfPages(path) as pdf:
        for lo, hi in [*SNR_BANDS, (0, np.inf)]:
            band = (rows["snr_hat"] >= lo) & (rows["snr_hat"] < hi)
            joint = (rows["joint_snr_hat"] >= lo) & (rows["joint_snr_hat"] < hi)
            fig, axes = plt.subplots(
                len(names),
                n_sources + 1,
                figsize=(2.2 * (n_sources + 1), 1.4 * len(names)),
                sharex=True,
                squeeze=False,
            )
            for i, name in enumerate(names):
                sets = [
                    rows["rank"][name][band[:, s], s] if name in rows["rank"] else None
                    for s in range(n_sources)
                ]
                sets.append(
                    rows["joint_rank"][name][joint]
                    if name in rows["joint_rank"]
                    else None
                )
                for s, r in enumerate(sets):
                    ax = axes[i, s]
                    ax.tick_params(labelsize=6)
                    if r is None or len(r) < 10:
                        ax.axis("off")
                        continue
                    p, (band_lo, band_hi) = gamma_test(r, k)
                    ecdf = (r[:, None] < np.arange(1, k)).mean(0)
                    ax.fill_between(z, band_lo, band_hi, color="0.85", lw=0)
                    ax.plot(z, ecdf - z, lw=0.8, color="C3" if p < ALPHA else "C0")
                    ax.axhline(0, color="0.5", lw=0.5)
                    ax.set_title(
                        f"{name}, {'joint' if s == n_sources else f'slot {s + 1}'}: p {p:.2f}",
                        fontsize=6,
                    )
            label = (
                f"{lo:g}-{hi:g}" if np.isfinite(hi) else (f">={lo:g}" if lo else "all")
            )
            fig.suptitle(
                f"ECDF - z of the truth's rank, draws' median snr {label}", fontsize=8
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def run_pages(run_dir: Path) -> list[Path]:
    """losses.pdf and lambda.pdf from a run's history.npz."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    h = dict(np.load(run_dir / "history.npz"))
    steps, groups = h["steps"], list(h["groups"])
    lo, median, hi = np.nanpercentile(h["losses"], [10, 50, 90], axis=1)
    has_lambda = np.isfinite(h["log_var"]).any()

    # the training losses (median and 10-90% of each epoch's steps), held-out x, and per group the time-binned mean
    panels = 5 if has_lambda else 3
    fig, axes = plt.subplots(panels, 1, figsize=(7, 2.4 * panels), sharex=True)
    for ax, c, heldout in zip(axes[:3], (0, 1, 2), ("heldout_x", None, None)):
        ax.fill_between(steps, lo[:, c], hi[:, c], color="C0", alpha=0.25, lw=0)
        ax.plot(steps, median[:, c], color="C0", lw=1, label="training")
        if heldout:
            ax.plot(steps, h["heldout"][:, 0], color="C3", lw=1, label=heldout)
            ax.legend(fontsize=7)
        ax.set_ylabel(str(h["loss_names"][c]))
        ax.set_yscale("log" if c < 2 else "linear")
    if has_lambda:
        for g, name in enumerate(groups):
            axes[3].plot(steps, np.nanmean(h["log_var"][:, :, g], axis=1), lw=1, label=name)
            axes[4].plot(
                steps, np.nanmean(h["bin_objective"][:, :, g], axis=1), lw=1, label=name
            )
        axes[3].set_ylabel("λ, mean over time bins")
        axes[4].set_ylabel("objective, mean over time bins")
        axes[3].legend(fontsize=7, ncol=len(groups))
    axes[-1].set_xlabel("step")
    fig.suptitle(f"{run_dir.name}, {int(h['epoch'])} epochs", fontsize=9)
    fig.tight_layout()
    fig.savefig(run_dir / "losses.pdf")
    plt.close(fig)
    pages = [run_dir / "losses.pdf"]

    # per group, step x log10(1 - s) heatmaps of λ, the error and the objective
    quantities = ["log_var", "bin_error", "bin_objective"] if has_lambda else ["bin_error"]
    extent = (steps[0], steps[-1], h["time_edges"][0], h["time_edges"][-1])
    with PdfPages(run_dir / "lambda.pdf") as pdf:
        for name in quantities:
            fig, axes = plt.subplots(1, len(groups), figsize=(3 * len(groups), 3), sharey=True)
            for ax, g in zip(axes, groups):
                values = h[name][:, :, groups.index(g)].T
                image = ax.imshow(
                    np.log10(values) if name == "bin_error" else values,
                    aspect="auto",
                    origin="lower",
                    extent=extent,
                    interpolation="nearest",
                )
                fig.colorbar(image, ax=ax)
                ax.set_title(g, fontsize=8)
                ax.set_xlabel("step", fontsize=7)
                ax.tick_params(labelsize=6)
            axes[0].set_ylabel("log10(1 - s)")
            fig.suptitle(
                f"{run_dir.name}: {'log10 ' if name == 'bin_error' else ''}{name} per time bin",
                fontsize=9,
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    return pages + [run_dir / "lambda.pdf"]


def eval_pages(eval_dir: Path) -> list[Path]:
    """corner/ from draws.npz (with mcmc.npz's chains when `mcmc report` has run), and validation.pdf."""
    pages = []
    if (eval_dir / "draws.npz").exists():
        d = np.load(eval_dir / "draws.npz")
        mcmc = dict(np.load(eval_dir / "mcmc.npz")) if (eval_dir / "mcmc.npz").exists() else {}
        source_snr, quantiles = d["source_snr"], d["quantiles"]
        corner_dir = eval_dir / "corner"
        corner_dir.mkdir(exist_ok=True)
        for j in range(len(d["truth"])):
            n_sources = source_snr.shape[1]
            chains = [mcmc.get(f"{j}_{s}") for s in range(n_sources)]
            injection_pages(
                corner_dir,
                f"q{quantiles[j]:.2f}",
                quantiles[j],
                d["snr"][j],
                d["matched"][j],
                d["fisher"][j],
                d["truth"][j],
                source_snr[j],
                None if any(c is None for c in chains) else np.stack(chains, axis=1),
            )
        for name, (j, s) in gb_picks(source_snr).items():
            gb_page(
                corner_dir,
                name,
                source_snr.size,
                source_snr[j, s],
                f"q{quantiles[j]:.2f}",
                s,
                d["matched"][j][:, s],
                d["fisher"][j][:, s],
                d["truth"][j, s],
                mcmc.get(f"{j}_{s}"),
            )
        pages.append(corner_dir)
    if (eval_dir / "validation_scores.npz").exists():
        scores = dict(np.load(eval_dir / "validation_scores.npz"))
        rows = dict(
            rank={
                k.removeprefix("rank_"): v for k, v in scores.items() if k.startswith("rank_")
            },
            joint_rank={
                k.removeprefix("joint_rank_"): v
                for k, v in scores.items()
                if k.startswith("joint_rank_")
            },
            draws=scores["draws"],
            snr_hat=scores["snr_hat"],
            joint_snr_hat=scores["joint_snr_hat"],
        )
        ecdf_pages(rows, eval_dir / "validation.pdf")
        pages.append(eval_dir / "validation.pdf")
    return pages


def main() -> int:
    """Draw every figure of the given run dirs (their evals too), eval dirs, page dirs or page .npz files."""
    import matplotlib

    matplotlib.use("Agg")
    pages = []
    for target in map(Path, sys.argv[1:]):
        if (target / "config.yaml").exists():
            pages += run_pages(target) if (target / "history.npz").exists() else []
            for eval_dir in sorted((target / "eval").glob("*/eval.yaml")):
                pages += eval_pages(eval_dir.parent)
        elif (target / "eval.yaml").exists():
            pages += eval_pages(target)
        elif target.is_dir():
            pages += [replot_page(npz) for npz in sorted(target.glob("*.npz"))]
        else:
            pages.append(replot_page(target))
    if not pages:
        print("nothing to draw: give a run dir, an eval dir, a corner dir or a page .npz", file=sys.stderr)
        return 1
    for page in pages:
        print(page, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
