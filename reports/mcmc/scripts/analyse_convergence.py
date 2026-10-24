"""Burn-in, kept length and thinning of the joint narrowband MCMC, from the saved long chains of long_chain.py."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from einops import rearrange
from jaxtyping import Float

REPORT = Path(__file__).parents[1]
from joint_mcmc import WINDOWS, chain_dir

from canna.lisa import config as run_config
from canna.lisa.mcmc import autocorr_time, gelman_rubin
from canna.lisa.problem import LisaGB

# the criterion: burn >= BURN_TAUS tau_max with every later window's marginals settled; then keep
# until ESS >= ESS_MIN per seed, R-hat < RHAT_MAX and every marginal within tolerance of the other seeds'
BURN_TAUS = 10
ESS_MIN = 2000
RHAT_MAX = 1.01
MEDIAN_TOL = 0.1  # median shift, in the reference's half 68% widths
WIDTH_TOL = 0.1  # |68% width / the reference's - 1|
# a long enough chain for tau's own estimate (emcee's rule)
TAU_RELIABLE = 50
# (burn, keep, thin) in use: the paper appendix's and the fml script's
SETTINGS = {"paper": (100_000, 300_000, 100), "fml": (20_000, 60_000, 20)}
KEEP_FRACTIONS = (1 / 40, 1 / 20, 1 / 10, 3 / 20, 1 / 4, 3 / 8, 1 / 2, 3 / 4)


def load(directory: Path, j: int) -> dict:
    """Window j over seeds: draws [K, W, T, P] (S x 8 coordinates, then log L) and the run's metadata."""
    runs = sorted(directory.glob(f"window{j}_seed*.npz"))
    assert (
        len(runs) >= 2
    ), f"R-hat across seeds needs two seeds of window {j} in {directory}"
    zs = [np.load(r) for r in runs]
    for z in zs[1:]:
        assert z["trace"].shape == zs[0]["trace"].shape and int(z["thin"]) == int(
            zs[0]["thin"]
        )
    trace = rearrange(np.stack([z["trace"] for z in zs]), "k w t s c -> k w t (s c)")
    loglik = np.stack([z["loglik"] for z in zs])[..., None]
    columns = [str(c) for c in zs[0]["columns"]]
    n_sources = zs[0]["trace"].shape[2]
    return dict(
        x=np.concatenate([trace.astype(np.float64), loglik], axis=-1),
        names=[f"s{s} {c}" for s in range(n_sources) for c in columns] + ["log L"],
        columns=columns,
        n_sources=n_sources,
        thin=int(zs[0]["thin"]),
        chunk=int(zs[0]["chunk"]),
        nwalker=int(zs[0]["nwalker"]),
        seeds=[int(z["chain_seed"]) for z in zs],
        snr=zs[0]["snr"].tolist(),
        window_snr=float(zs[0]["window_snr"]),
        config=str(zs[0]["config"]),
        problem_id=str(zs[0]["problem_id"]),
    )


def tau(x: Float[np.ndarray, "K W T P"], thin: int) -> Float[np.ndarray, "K P"]:
    """Integrated autocorrelation time per seed and parameter, in iterations."""
    return thin * np.array(
        [
            [autocorr_time(x[k, ..., p]) for p in range(x.shape[-1])]
            for k in range(len(x))
        ]
    )


def rhat(x: Float[np.ndarray, "K W T P"]) -> Float[np.ndarray, "P"]:
    """Split R-hat across seeds: each half of each seed's ensemble is one group."""
    t = x.shape[2] // 2 * 2
    return np.array(
        [
            gelman_rubin(
                rearrange(x[:, :, :t, p], "k w (h t) -> (k h w) t", h=2),
                groups=2 * len(x),
            )
            for p in range(x.shape[-1])
        ]
    )


def quantiles(x: Float[np.ndarray, "... P"]) -> tuple[np.ndarray, np.ndarray]:
    """Median and central 68% width of each parameter, over every leading axis."""
    q16, q50, q84 = np.percentile(
        rearrange(x, "... p -> (...) p"), [16, 50, 84], axis=0
    )
    return q50, q84 - q16


def references(x: Float[np.ndarray, "K W T P"]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per seed, the other seeds' second halves pooled: a full-chain value independent of that seed."""
    half = x.shape[2] // 2
    return [quantiles(np.delete(x, k, axis=0)[:, :, half:]) for k in range(len(x))]


def drift(x: Float[np.ndarray, "K W T P"], refs) -> tuple[np.ndarray, np.ndarray]:
    """The worst seed's median shift (in reference half-widths) and |68% width ratio - 1|, per parameter."""
    shifts, ratios = [], []
    for k, (ref_median, ref_width) in enumerate(refs):
        median, width = quantiles(x[k])
        shifts.append(np.abs(median - ref_median) / (ref_width / 2))
        ratios.append(np.abs(width / ref_width - 1))
    return np.max(shifts, axis=0), np.max(ratios, axis=0)


def segment(x: np.ndarray, thin: int, start: int, stop: int) -> np.ndarray:
    """Draws of iterations [start, stop)."""
    return x[:, :, start // thin : stop // thin]


def burn_curves(x, thin: int, nwalker: int, step: int, refs) -> dict:
    """tau, ESS and R-hat on [b, end) for b up to half the chain; drift of each step-long window [b, b + step)."""
    n_iter = x.shape[2] * thin
    windows = np.arange(0, n_iter - step + 1, step)
    burns = windows[windows <= n_iter // 2]
    taus, ess, rhats = [], [], []
    for b in burns:
        t = tau(segment(x, thin, b, n_iter), thin)
        taus.append(t.max(axis=0))
        ess.append((nwalker * (n_iter - b) / t).min(axis=0))
        rhats.append(rhat(segment(x, thin, b, n_iter)))
    medians, widths = [], []
    for b in windows:
        m, w = drift(segment(x, thin, b, b + step), refs)
        medians.append(m)
        widths.append(w)
    # the drift test's own noise: the ensemble's ESS in one window
    window_ess = float((nwalker * step / np.array(taus)).min())
    return dict(
        burn=burns,
        tau=np.array(taus),
        ess=np.array(ess),
        rhat=np.array(rhats),
        window=windows,
        window_median=np.array(medians),
        window_width=np.array(widths),
        window_ess=window_ess,
    )


def keep_curves(x, thin: int, nwalker: int, burn: int, keeps, refs) -> dict:
    """tau, ESS, R-hat and drift of the kept draws [burn, burn + keep), per kept length."""
    taus, ess, rhats, medians, widths = [], [], [], [], []
    for keep in keeps:
        kept = segment(x, thin, burn, burn + keep)
        t = tau(kept, thin)
        taus.append(t.max(axis=0))
        ess.append((nwalker * keep / t).min(axis=0))
        rhats.append(rhat(kept))
        m, w = drift(kept, refs)
        medians.append(m)
        widths.append(w)
    return dict(
        keep=np.asarray(keeps),
        tau=np.array(taus),
        ess=np.array(ess),
        rhat=np.array(rhats),
        median=np.array(medians),
        width=np.array(widths),
    )


def pick_burn(curves: dict) -> int | None:
    """The smallest burn past BURN_TAUS tau_max after which every window's marginals are within tolerance."""
    ok = (curves["window_median"].max(axis=1) <= MEDIAN_TOL) & (
        curves["window_width"].max(axis=1) <= WIDTH_TOL
    )
    settled = np.logical_and.accumulate(ok[::-1])[::-1]
    for i, b in enumerate(curves["burn"]):
        if b >= BURN_TAUS * curves["tau"][i].max() and settled[i]:
            return int(b)
    return None


def passes_keep(ess, rhats, medians, widths) -> np.ndarray:
    return (
        (ess.min(axis=-1) >= ESS_MIN)
        & (rhats.max(axis=-1) < RHAT_MAX)
        & (medians.max(axis=-1) <= MEDIAN_TOL)
        & (widths.max(axis=-1) <= WIDTH_TOL)
    )


def pick_thin(tau_max: float, chunk: int) -> int:
    """The largest divisor of the epoch length at most tau_max / 2: the slowest parameter's kept draws near independent."""
    return max(
        d for d in range(1, chunk + 1) if chunk % d == 0 and d <= max(tau_max / 2, 1)
    )


def evaluate(
    x, thin: int, nwalker: int, refs, burn: int, keep: int, keep_thin: int, burn_star
) -> dict | None:
    """One (burn, keep, thin) setting against the criterion; None if the chain is shorter than burn + keep."""
    if burn + keep > x.shape[2] * thin:
        return None
    kept = segment(x, thin, burn, burn + keep)
    t = tau(kept, thin)
    ess = (nwalker * keep / t).min(axis=0)
    rhats = rhat(kept)
    medians, widths = drift(kept, refs)
    return dict(
        burn=burn,
        keep=keep,
        thin=keep_thin,
        tau_max=float(t.max()),
        ess_min=float(ess.min()),
        # a draw kept every thin iterations carries at most one draw's worth
        ess_thinned_min=float((nwalker * keep / np.maximum(t, keep_thin)).min()),
        stored_per_seed=nwalker * keep // keep_thin,
        rhat_max=float(rhats.max()),
        median_drift_max=float(medians.max()),
        width_drift_max=float(widths.max()),
        passes=bool(
            burn_star is not None
            and burn >= burn_star
            and passes_keep(ess, rhats, medians, widths)
        ),
    )


def round_up(n: float, chunk: int) -> int:
    return int(-(-n // chunk) * chunk)


def main() -> int:
    parser = argparse.ArgumentParser(prog="analyse_convergence", description=__doc__)
    parser.add_argument("--windows", type=int, nargs="+", default=WINDOWS)
    parser.add_argument(
        "--root", type=Path, default=REPORT.parents[1] / "outputs" / "mcmc" / "convergence" / "traces"
    )
    parser.add_argument("--out", type=Path, default=REPORT / "data" / "convergence")
    parser.add_argument(
        "--step",
        type=int,
        help="burn grid and drift window [iterations] (default: 1/20 of the chain)",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="the smoke test's chains: outputs/mcmc/convergence/bench/ in and out",
    )
    args = parser.parse_args()
    if args.bench:
        bench = REPORT.parents[1] / "outputs" / "mcmc" / "convergence" / "bench"
        args.root, args.out = bench, bench

    config = run_config.load()
    directory = chain_dir(LisaGB(**config.problem), config.seed, args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    windows, records = {}, {}
    for j in args.windows:
        data = load(directory, j)
        x, thin, chunk, nwalker = (
            data["x"],
            data["thin"],
            data["chunk"],
            data["nwalker"],
        )
        n_iter = x.shape[2] * thin
        step = args.step or round_up(n_iter / 20, chunk)
        refs = references(x)
        print(
            f"window {j}: {len(x)} seeds x {nwalker} walkers x {n_iter} iterations (thin {thin})",
            flush=True,
        )

        burn_c = burn_curves(x, thin, nwalker, step, refs)
        burn_star = pick_burn(burn_c)
        # with no burn-in found, the kept-length curves still go on the page, after half the chain
        burn_used = burn_star if burn_star is not None else int(burn_c["burn"][-1])
        keeps = sorted({round_up(n_iter * f, chunk) for f in KEEP_FRACTIONS})
        keeps = [k for k in keeps if burn_used + k <= n_iter]
        keep_c = keep_curves(x, thin, nwalker, burn_used, keeps, refs)
        ok = passes_keep(
            keep_c["ess"], keep_c["rhat"], keep_c["median"], keep_c["width"]
        )
        keep_star = (
            int(keep_c["keep"][np.argmax(ok)])
            if burn_star is not None and ok.any()
            else None
        )

        record = dict(
            window_snr=data["window_snr"],
            snr=data["snr"],
            seeds=data["seeds"],
            burn=burn_star,
            keep=keep_star,
            thin=None,
            step=step,
            window_ess_min=burn_c["window_ess"],
            burn_test_noise_limited=burn_c["window_ess"] < ESS_MIN,
        )
        if keep_star is not None:
            i = int(np.argmax(ok))
            tau_max = float(keep_c["tau"][i].max())
            record.update(
                thin=pick_thin(tau_max, chunk),
                tau_max=tau_max,
                tau_max_parameter=data["names"][int(keep_c["tau"][i].argmax())],
                ess_min=float(keep_c["ess"][i].min()),
                ess_min_parameter=data["names"][int(keep_c["ess"][i].argmin())],
                rhat_max=float(keep_c["rhat"][i].max()),
                # tau at or below twice the trace's thinning is not resolved: thin finer
                tau_resolved=tau_max > 2 * thin,
                tau_reliable=keep_star >= TAU_RELIABLE * tau_max,
            )
        record["settings"] = {
            name: evaluate(x, thin, nwalker, refs, *setting, burn_star)
            for name, setting in SETTINGS.items()
        }
        windows[str(j)] = record
        print(json.dumps(record, indent=1), flush=True)

        records.update({f"w{j}_burn_{k}": v for k, v in burn_c.items()})
        records.update({f"w{j}_keep_{k}": v for k, v in keep_c.items()})
        records[f"w{j}_names"] = np.array(data["names"])

    found = all(r["keep"] is not None for r in windows.values())
    summary = dict(
        criterion=dict(
            text=(
                f"per window: burn = the smallest grid burn-in >= {BURN_TAUS} tau_max (tau on the post-burn"
                f" chain) after which every window's median is within {MEDIAN_TOL} reference half-widths and"
                f" its 68% width within {WIDTH_TOL:.0%} of the other seeds' second half; keep = the smallest"
                f" grid length with ESS >= {ESS_MIN} per seed in every parameter, split R-hat across seeds"
                f" < {RHAT_MAX} and the same median and width tolerance; thin = the largest divisor of the"
                f" epoch length <= tau_max / 2. The recommendation is the worst window: max burn, max keep,"
                " min thin."
            ),
            burn_taus=BURN_TAUS,
            ess_min=ESS_MIN,
            rhat_max=RHAT_MAX,
            median_tol=MEDIAN_TOL,
            width_tol=WIDTH_TOL,
        ),
        config=data["config"],
        problem_id=data["problem_id"],
        chain=dict(iterations=n_iter, thin=thin, nwalker=nwalker, chunk=chunk),
        windows=windows,
        recommended=(
            dict(
                burn=max(r["burn"] for r in windows.values()),
                keep=max(r["keep"] for r in windows.values()),
                thin=min(r["thin"] for r in windows.values()),
            )
            if found
            else None
        ),
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    np.savez(args.out / "curves.npz", **records)
    print(
        f"\nrecommended {summary['recommended']}; saved {args.out / 'summary.json'}, curves.npz"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
