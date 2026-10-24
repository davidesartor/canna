"""Every lisa figure, drawn from saved data: a run's losses and λ, an eval dir's corner report and validation pages."""

import base64
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from .samples import robust_width

# the obliquity of the ecliptic at J2000 [rad]: LisaGB's sky is equatorial (lisaorbits v3), papers quote ecliptic
OBLIQUITY = np.deg2rad(23.4392911)
# draws per source the corner report keeps of each reference set; the flow keeps all of its own
REFERENCE_DRAWS = {"fisher": 512, "mcmc": 1024}
# float16 holds offsets from the truth in robust flow widths; past this they are only "off the page"
HALF_LIMIT = 6.0e4
CORNER_TEMPLATE = Path(__file__).with_name("corner.html")
# no creation date, so a redrawn figure with the same content is the same file (they are in git)
PDF_METADATA = {"CreationDate": None}
COORDINATES = {
    "physical": ["f₀", "M<sub>c</sub>", "log₁₀ A", "λ", "β", "ψ", "ι", "φ₀"],
    "flow": ["x<sub>f</sub>", "m", "l / σ<sub>l</sub>", "b / σ<sub>b</sub>", "Re z<sub>+</sub>", "Im z<sub>+</sub>", "Re z<sub>−</sub>", "Im z<sub>−</sub>"],
}
UNITS = {
    "physical": ["bins", "M☉", "", "rad", "rad", "rad", "rad", "rad"],
    "flow": [""] * 8,
}


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


def equatorial_to_ecliptic(ra: np.ndarray, dec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equatorial (RA, Dec) to ecliptic (λ in [0, 2π), β) [rad], a rotation about the equinox by the obliquity."""
    x, y, z = np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)
    c, s = np.cos(OBLIQUITY), np.sin(OBLIQUITY)
    y, z = c * y + s * z, -s * y + c * z
    return np.arctan2(y, x) % (2 * np.pi), np.arcsin(np.clip(z, -1.0, 1.0))


def physical_columns(p: np.ndarray, truth: np.ndarray, start_bin: float, t_obs: float) -> np.ndarray:
    """The report's physical columns of draws [..., S, 8] folded onto truth [S, 8]: f0 in bins from the window's
    first, Mc, log10 A, ecliptic λ (unwrapped around the truth's) and β, ψ, ι, φ0."""
    lam, beta = equatorial_to_ecliptic(p[..., 3], p[..., 4])
    lam_true, _ = equatorial_to_ecliptic(truth[..., 3], truth[..., 4])
    lam = lam_true + (lam - lam_true + np.pi) % (2 * np.pi) - np.pi
    with np.errstate(invalid="ignore", divide="ignore"):
        log_a = np.log10(p[..., 2])
    return np.stack(
        [p[..., 0] * t_obs - start_bin, p[..., 1], log_a, lam, beta, p[..., 5], p[..., 6], p[..., 7]],
        axis=-1,
    )


def corner_data(eval_dir: Path) -> tuple[dict, bytes]:
    """The corner report's index and its draws: every eval injection's flow, Fisher and MCMC draws per source, in
    physical and flow coordinates, as float16 offsets from the truth in units of the flow's robust width."""
    import jax.numpy as jnp
    import yaml

    from .chart import physical_to_flow
    from .problem import LisaGB, window_start
    from .samples import load as load_samples

    run_dir = eval_dir.parents[1]
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    stage = yaml.safe_load((eval_dir / "eval.yaml").read_text()) if (eval_dir / "eval.yaml").exists() else {}
    problem = LisaGB(**config["problem"])
    d = load_samples(eval_dir / "draws.npz")
    mcmc = load_samples(eval_dir / "mcmc.npz") if (eval_dir / "mcmc.npz").exists() else {}
    rng = np.random.default_rng(0)
    n_injections, n_sources = d["source_snr"].shape

    blob, injections = [], []
    offset = 0
    for j in range(n_injections):
        truth, f = d["truth"][j], d["f"][j]
        chains = [mcmc.get(f"{j}_{s}") for s in range(n_sources)]
        sets = {"flow": d["matched"][j], "fisher": d["fisher"][j]}
        if all(c is not None for c in chains):
            sets["mcmc"] = np.stack(chains, axis=1)
        for name, keep in REFERENCE_DRAWS.items():
            if name in sets and len(sets[name]) > keep:
                sets[name] = sets[name][np.sort(rng.choice(len(sets[name]), keep, replace=False))]
        start_bin = float(window_start(problem, f))
        views = {}
        for name, p in [("truth", truth[None]), *sets.items()]:
            p = fold_orientation(p, truth)
            views[name] = {
                "physical": physical_columns(p, truth, start_bin, problem.t_obs),
                # the chart takes the folded draws whole: its z± are the same for every (ψ, φ0) copy
                "flow": np.asarray(physical_to_flow(problem, jnp.asarray(p), f)),
            }
        sources = []
        for s in range(n_sources):
            source = {"snr": float(d["source_snr"][j, s]), "truth": {}, "scale": {}, "sets": {}}
            for view in COORDINATES:
                centre = views["truth"][view][0, s]
                scale = robust_width(views["flow"][view][:, s], 0)
                # a flow collapsed onto one value in a column falls back to a reference's width, then to 1
                for name in ("mcmc", "fisher"):
                    if name in views:
                        fallback = robust_width(views[name][view][:, s], 0)
                        scale = np.where(np.isfinite(scale) & (scale > 0), scale, fallback)
                scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
                source["truth"][view] = centre.tolist()
                source["scale"][view] = scale.tolist()
                for name in sets:
                    v = (views[name][view][:, s] - centre) / scale
                    v = np.where(np.isfinite(v), np.clip(v, -HALF_LIMIT, HALF_LIMIT), np.nan)
                    blob.append(v.astype("<f2").tobytes())
                    source["sets"].setdefault(name, {})[view] = [offset, len(v)]
                    offset += v.size
            sources.append(source)
        injections.append(
            {
                "quantile": float(d["quantiles"][j]),
                "snr": float(d["snr"][j]),
                "f_mhz": float(f * 1e3),
                "start_bin": start_bin,
                "window_bins": int(problem.window_bins),
                "sources": sources,
            }
        )
    meta = {
        "run": run_dir.name,
        "config": config.get("name", ""),
        "commit": str(config.get("commit", ""))[:7],
        "eval": eval_dir.name,
        "epoch": int(d["epoch"]),
        "ode_steps": stage.get("ode_steps", config.get("ode_steps")),
        "weights": stage.get("weights", config.get("weights")),
        "t_obs": float(problem.t_obs),
        "draws": {name: int(len(p)) for name, p in sets.items()},
        "coordinates": COORDINATES,
        "units": UNITS,
        "picks": {k: list(map(int, v)) for k, v in gb_picks(d["source_snr"]).items()},
    }
    return {"meta": meta, "injections": injections}, b"".join(blob)


def corner_report(eval_dir: Path) -> Path:
    """eval_dir/corner.html: one interactive page of every eval injection's corners, physical and flow coordinates."""
    index, blob = corner_data(eval_dir)
    page = CORNER_TEMPLATE.read_text()
    # the layout's version on the page, so a stale build is told from the current template at a glance
    index["meta"]["layout"] = hashlib.sha1(page.encode()).hexdigest()[:8]
    page = page.replace("__TITLE__", f"{index['meta']['run']} corners")
    page = page.replace("__DATA__", json.dumps(index, separators=(",", ":")))
    page = page.replace("__BLOB__", base64.b64encode(blob).decode())
    path = eval_dir / "corner.html"
    path.write_text(page)
    return path


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
    with PdfPages(path, metadata=PDF_METADATA) as pdf:
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

    # the training losses (median and 10-90% of each epoch's steps), held-out x, and per group the time-binned mean;
    # log-log, each curve over its own minimum (the training band over the median's), so a plateau sits on 1
    panels = 5 if has_lambda else 3
    fig, axes = plt.subplots(panels, 1, figsize=(7, 2.4 * panels), sharex=True)
    for ax, c, heldout in zip(axes[:3], (0, 1, 2), ("heldout_x", None, None)):
        curves = [median[:, c]] + ([h["heldout"][:, 0]] if heldout else [])
        floors = [np.nanmin(v) for v in curves]
        # a non-positive minimum (the objective can go below zero) cannot scale a log axis
        scaled = all(m > 0 for m in floors)
        ref = floors[0] if scaled else 1.0
        ax.fill_between(steps, lo[:, c] / ref, hi[:, c] / ref, color="C0", alpha=0.25, lw=0)
        ax.plot(steps, median[:, c] / ref, color="C0", lw=1, label=f"training (min {floors[0]:.4g})")
        if heldout:
            ax.plot(
                steps,
                h["heldout"][:, 0] / (floors[1] if scaled else 1.0),
                color="C3",
                lw=1,
                label=f"{heldout} (min {floors[1]:.4g})",
            )
        ax.legend(fontsize=7)
        name = str(h["loss_names"][c])
        ax.set_ylabel(f"{name} / min" if scaled else name)
        ax.set_yscale("log" if scaled else "linear")
    axes[0].set_xscale("log")
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
    fig.savefig(run_dir / "losses.pdf", metadata=PDF_METADATA)
    plt.close(fig)
    pages = [run_dir / "losses.pdf"]

    # per group, step x log10(1 - s) heatmaps of λ, the error and the objective
    quantities = ["log_var", "bin_error", "bin_objective"] if has_lambda else ["bin_error"]
    extent = (steps[0], steps[-1], h["time_edges"][0], h["time_edges"][-1])
    with PdfPages(run_dir / "lambda.pdf", metadata=PDF_METADATA) as pdf:
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
    """corner.html from draws.npz (with mcmc.npz's chains when `mcmc report` has run), and validation.pdf."""
    pages = []
    if (eval_dir / "draws.npz").exists():
        pages.append(corner_report(eval_dir))
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
    """Draw every figure of the given run dirs (their evals too) or eval dirs, on the CPU."""
    import jax
    import matplotlib

    # the only JAX here is the corner report's chart conversion: no accelerator needed
    jax.config.update("jax_platforms", "cpu")

    matplotlib.use("Agg")
    pages = []
    for target in map(Path, sys.argv[1:]):
        if (target / "config.yaml").exists():
            pages += run_pages(target) if (target / "history.npz").exists() else []
            for eval_dir in sorted((target / "eval").glob("*/eval.yaml")):
                pages += eval_pages(eval_dir.parent)
        elif (target / "eval.yaml").exists():
            pages += eval_pages(target)
    if not pages:
        print("nothing to draw: give a run dir or an eval dir", file=sys.stderr)
        return 1
    for page in pages:
        print(page, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
