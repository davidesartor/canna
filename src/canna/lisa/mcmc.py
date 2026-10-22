"""Narrowband MCMC of the eval sources: one jexplore ensemble per source, the others at their truth."""

import argparse
import hashlib
import json
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import yaml
from jaxtyping import Array

from . import config as run_config
from .injections import (
    N_QUANTILES,
    eval_injections,
    scaled_inverse,
)
from .physics import (
    chirp_mass_log_density,
    chirp_mass_support,
    sample_chirp_mass,
    sky_log_density,
)
from .plots import fold_orientation, gb_picks
from .problem import LisaGB, f0_window, window_power
from .simulator import (
    clean_signal,
    sample_observation,
    snr,
    source_segments,
    source_snrs,
)

N_PARAMS = 8
NAMES = ["log f0", "log Mc", "log A", "lon", "lat", "psi", "sin iota", "phi0"]
ANGLES = [3, 5, 7]
MCMC_ROOT = Path("outputs/mcmc")
# draws per source kept for the corner pages
N_PLOT = 20000
COLUMNS = ["f0 [bins]", "log Mc", "log A", "lon", "lat", "psi", "sin iota", "phi0"]


def problem_id(problem: LisaGB, seed: int) -> str:
    """A short hash of everything the eval injections depend on: the problem and the seed."""
    fields = {
        k: getattr(problem, k)
        for k in (
            "n_sources",
            "t_obs",
            "response_points",
            "oms_noise",
            "acceleration_noise",
            "mass_range",
            "f0_range",
            "snr_range",
        )
    }
    # fields the problem has since dropped, at their only values: keeps existing chain caches' ids
    fields["sampling_step"], fields["order_sources"], fields["orbit"] = (
        0.25,
        True,
        "EqualArmlengthOrbits",
    )
    fields["amplitude_prior"] = "snr"
    fields["sky_prior"] = "galactic"
    fields["seed"] = seed
    text = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha1(text.encode()).hexdigest()[:10]


def cache_dir(problem: LisaGB, seed: int, root: Path = MCMC_ROOT) -> Path:
    return root / f"eval-{problem_id(problem, seed)}"


def targets(source_snr: np.ndarray) -> dict[tuple[int, int], list[str]]:
    """(injection, source) -> the page names it appears on."""
    out = {}
    for name, js in gb_picks(source_snr).items():
        out.setdefault(js, []).append(f"gb-{name}")
    # every source of every eval injection, so every corner page has its MCMC
    for j in range(source_snr.shape[0]):
        for s in range(source_snr.shape[1]):
            out.setdefault((j, s), []).append(f"q{(j + 1) / N_QUANTILES:.2f}")
    return out


def chain_file(directory: Path, j: int, s: int, seed: int) -> Path:
    return directory / f"inj{j}_src{s}_seed{seed}.npz"


def load_chains(
    problem: LisaGB, seed: int, root: Path = MCMC_ROOT, n: int = N_PLOT
) -> dict[tuple[int, int], np.ndarray]:
    """The cached chains, (injection, source) -> physical draws [n, 8] over every seed; {} if none."""
    directory = cache_dir(problem, seed, root)
    out = {}
    for path in sorted(directory.glob("inj*_src*_seed*.npz")):
        j, s = (
            int(v)
            for v in path.stem.replace("inj", "").replace("src", "").split("_")[:2]
        )
        try:
            physical = np.load(path)["physical"]
        except (
            OSError,
            ValueError,
            KeyError,
        ) as error:  # a damaged file: skip it, loudly
            print(f"[mcmc] skipping {path}: {error}", file=sys.stderr)
            continue
        out.setdefault((j, s), []).append(physical)
    rng = np.random.default_rng(0)
    for js, parts in out.items():
        pooled = np.concatenate(parts)
        # the chains' angle boxes are centred on the truth, the flow's angles in [0, 2 pi)
        pooled[:, ANGLES] = np.mod(pooled[:, ANGLES], 2 * np.pi)
        out[js] = pooled[rng.choice(len(pooled), min(n, len(pooled)), replace=False)]
    return out


def to_physical(theta: jax.Array) -> jax.Array:
    """Sampler coordinates (log f0, log Mc, log A, lon, lat, psi, sin iota, phi0) -> physical."""
    t = theta
    return jnp.stack(
        [
            jnp.exp(t[..., 0]),
            jnp.exp(t[..., 1]),
            jnp.exp(t[..., 2]),
            t[..., 3],
            t[..., 4],
            t[..., 5],
            jnp.arcsin(t[..., 6]),
            t[..., 7],
        ],
        axis=-1,
    )


def to_sampler(p: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """(..., 8) physical -> (..., 8) sampler coordinates, angles wrapped onto their boxes."""
    p = np.asarray(p, np.float64)
    t = np.stack(
        [
            np.log(p[..., 0]),
            np.log(p[..., 1]),
            np.log(np.abs(p[..., 2])),
            p[..., 3],
            p[..., 4],
            p[..., 5],
            np.sin(p[..., 6]),
            p[..., 7],
        ],
        axis=-1,
    )
    for i in ANGLES:
        t[..., i] = centres[i] + (t[..., i] - centres[i] + np.pi) % (2 * np.pi) - np.pi
    return t


class Target(NamedTuple):
    """Source s of eval injection j, the others subtracted at their truth: its data and prior box."""

    j: int
    s: int
    f: Array
    truth: np.ndarray
    snr: float
    window_snr: float
    residual: Array
    power: Array
    centres: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    mc_grid: Array
    mc_table: Array


def make_target(problem: LisaGB, injections, j: int, s: int) -> Target:
    latents, windows, snrs, _, noise_keys, _ = injections
    f = windows[j]
    truth_all = np.asarray(latents[j], np.float64)
    truth = truth_all[s]
    obs = sample_observation(problem, noise_keys[j], latents[j], f)
    others = np.delete(truth_all, s, axis=0)
    residual = (
        obs - clean_signal(problem, jnp.asarray(others), f) if len(others) else obs
    )

    centres = np.zeros(N_PARAMS)
    centres[ANGLES] = truth[ANGLES]
    f_lo, f_hi = (float(v) for v in f0_window(problem, f))
    # the snr cut in log_likelihood is the amplitude's support; the box only has to hold it near the truth
    snr_lo, snr_hi = problem.snr_range
    a_lo, a_hi = truth[2] * snr_lo / snr_hi, truth[2] * snr_hi / snr_lo
    mc_lo, mc_hi = chirp_mass_support(problem.mass_range)
    lower = np.array([np.log(f_lo), np.log(mc_lo), np.log(a_lo), 0, -np.pi / 2, 0, -1, 0])
    upper = np.array([np.log(f_hi), np.log(mc_hi), np.log(a_hi), 0, np.pi / 2, 0, 1, 0])
    lower[ANGLES], upper[ANGLES] = centres[ANGLES] - np.pi, centres[ANGLES] + np.pi

    grid, table = chirp_mass_log_density(problem.mass_range)
    finite = np.isfinite(table)
    return Target(
        j=j,
        s=s,
        f=f,
        truth=truth,
        snr=float(snr(problem, jnp.asarray(truth)[None], f)),
        window_snr=float(snrs[j]),
        residual=residual,
        power=window_power(problem, f),
        centres=centres,
        lower=lower,
        upper=upper,
        mc_grid=jnp.asarray(grid[finite]),
        mc_table=jnp.asarray(table[finite]),
    )


def log_prior(target: Target, theta: jax.Array) -> jax.Array:
    inside = jnp.all((theta >= target.lower) & (theta <= target.upper))
    lp = jnp.interp(
        theta[1], target.mc_grid, target.mc_table, left=-jnp.inf, right=-jnp.inf
    )
    return jnp.where(inside, lp + sky_log_density(theta[3], theta[4]), -jnp.inf)


def log_likelihood(problem: LisaGB, target: Target, theta: jax.Array) -> jax.Array:
    """log L + const: -sum (|r - h|^2 - |r|^2) / P over the source's own band of the residual."""
    idx, seg = source_segments(problem, to_physical(theta)[None], target.f)
    idx, seg = idx[0], seg[0]
    # clean_signal drops the bins outside the window
    inside = (idx >= 0) & (idx < problem.window_bins)
    k = jnp.clip(idx, 0, problem.window_bins - 1)
    r, power = target.residual[k], target.power[k]
    d = (jnp.abs(r - seg) ** 2 - jnp.abs(r) ** 2) / power
    # the amplitude prior's support, the source's own snr in snr_range, off the same response
    rho = jnp.sqrt(2.0 * jnp.sum(jnp.where(inside[:, None], jnp.abs(seg) ** 2 / power, 0.0)))
    in_range = (rho >= problem.snr_range[0]) & (rho <= problem.snr_range[1])
    return jnp.where(in_range, -jnp.sum(jnp.where(inside[:, None], d, 0.0)), -jnp.inf)


def fisher_cov(problem: LisaGB, target: Target) -> np.ndarray:
    """Fisher covariance in sampler coordinates, plus the box as a weak Gaussian to keep it finite."""
    t0 = jnp.asarray(to_sampler(target.truth, target.centres))

    def signal(theta):
        h = clean_signal(problem, to_physical(theta)[None], target.f)
        h = h / jnp.sqrt(target.power)
        return jnp.concat([h.real.ravel(), h.imag.ravel()])

    jac = jax.jacfwd(signal)(t0)
    fisher = 2.0 * np.asarray(jac.T @ jac)
    prior = 12.0 / (target.upper - target.lower) ** 2
    return np.asarray(scaled_inverse(jnp.asarray(fisher + np.diag(prior))))


def initial_points(
    problem: LisaGB, target: Target, n: int, key: jax.Array
) -> np.ndarray:
    """Walkers on the Fisher ellipse, chirp masses from the prior, clipped into the box and snr_range."""
    t0 = to_sampler(target.truth, target.centres)
    cov = fisher_cov(problem, target)
    theta = np.array(
        jr.multivariate_normal(
            key, jnp.asarray(t0), jnp.asarray(cov), (n,), method="eigh"
        )
    )
    mc = jax.vmap(partial(sample_chirp_mass, mass_range=problem.mass_range))(
        jr.split(jr.fold_in(key, 1), n)
    )
    theta[:, 1] = np.log(np.asarray(mc)).reshape(n)
    pad = 1e-9 * (target.upper - target.lower)
    theta = np.clip(theta, target.lower + pad, target.upper - pad)
    rho = np.asarray(
        jax.vmap(lambda t: snr(problem, to_physical(t)[None], target.f))(jnp.asarray(theta))
    )
    margin = 1 + 1e-6
    lo, hi = problem.snr_range[0] * margin, problem.snr_range[1] / margin
    theta[:, 2] += np.log(np.clip(rho, lo, hi) / rho)
    return theta


def autocorr_time(x, c: float = 5.0):
    """Integrated autocorrelation time of chains [W, T]: emcee's estimator, Sokal's window."""
    y = x - x.mean(axis=1, keepdims=True)
    n = 1 << (2 * y.shape[1] - 1).bit_length()
    acf = np.fft.irfft(np.abs(np.fft.rfft(y, n, axis=1)) ** 2, axis=1)[:, : y.shape[1]]
    acf = (acf / acf[:, :1]).mean(axis=0)
    taus = 2.0 * np.cumsum(acf) - 1.0
    m = np.arange(len(taus)) < c * taus
    window = np.argmin(m) if not m.all() else len(taus) - 1
    return taus[window]


def gelman_rubin(chains, groups: int = 4):
    """Split R-hat over walker groups of one parameter's chains [W, T]."""
    g = np.array_split(chains, groups, axis=0)
    means = np.array([c.mean() for c in g])
    within = np.mean([c.var(ddof=1) for c in g])
    n = np.mean([c.size for c in g])
    between = n * means.var(ddof=1)
    return np.sqrt(((n - 1) / n * within + between / n) / within)


class NullBackend:
    """jexplore backend that keeps nothing: the cold chains are thinned here."""

    burn = 0

    def reset(self):
        pass

    def ingest(self, epoch, eind=-1):
        return self


def sample_chains(problem: LisaGB, target: Target, args, seed: int) -> dict:
    """Run one source's chains; returns the arrays to save."""
    from jexplore.sampling import SamplingMH  # jexplore needs Python 3.13
    from jexplore.steps import Stretch, TSwap
    from jexplore.steps.de import DEStep

    temps = jnp.asarray(args.temps)
    n_temps = len(args.temps)
    p0 = initial_points(
        problem, target, args.nwalker * n_temps, jr.key(1000 * seed + 17)
    )
    assert bool(
        jnp.all(jnp.isfinite(jax.vmap(partial(log_prior, target))(jnp.asarray(p0))))
    ), "start outside the prior"
    sampling = SamplingMH(
        dim=N_PARAMS,
        nwalker=args.nwalker,
        temps=temps,
        loglik=partial(log_likelihood, problem, target),
        logprior=partial(log_prior, target),
    )
    steps = [
        {TSwap(permute=True).builder: 1.0},
        {DEStep(permute=True).builder: 0.5, Stretch(permute=True).builder: 0.5},
    ]
    sampler = sampling.get_sampler(steps=steps, backend=NullBackend())
    epoch, kept, kept_ll, acc, done = sampling.get_epoch(jnp.asarray(p0)), [], [], [], 0
    total, start = args.burn + args.keep, time.perf_counter()
    while done < total:
        epoch = sampler.run(
            epoch, niters=args.chunk, nepoch=1, seed=seed * 1_000_003 + done
        )
        p = np.asarray(epoch.samples.p)[::n_temps]  # cold chains: (nwalker, dim, iters)
        ll = np.asarray(epoch.samples.ll)[::n_temps, 0]
        acc.append(
            np.asarray(epoch.stats.stats).mean(axis=-1) / np.asarray(epoch.stats.counts)
        )
        if done >= args.burn:
            kept.append(p[:, :, :: args.thin])
            kept_ll.append(ll[:, :: args.thin])
        done += args.chunk
    chains = np.concatenate(kept, axis=-1)  # (nwalker, dim, n)
    rhat = np.array([gelman_rubin(chains[:, d]) for d in range(N_PARAMS)])
    tau = np.array([autocorr_time(chains[:, d]) for d in range(N_PARAMS)]) * args.thin
    samples = chains.transpose(0, 2, 1).reshape(-1, N_PARAMS)
    return dict(
        samples=samples,
        physical=np.asarray(to_physical(jnp.asarray(samples))),
        loglik=np.concatenate(kept_ll, axis=-1).reshape(-1),
        rhat=rhat,
        tau=tau,
        ess=args.nwalker * args.keep / tau,
        acceptance=np.array(acc),
        seconds=time.perf_counter() - start,
    )


def run(config: run_config.Config, problem: LisaGB, args) -> None:
    injections = eval_injections(problem, config.seed)
    latents, windows = injections[0], injections[1]
    source_snr = np.asarray(jax.vmap(partial(source_snrs, problem))(latents, windows))
    directory = cache_dir(problem, config.seed, args.root)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "problem.yaml").write_text(
        yaml.safe_dump(
            {
                "config": config.name,
                "seed": config.seed,
                "problem_id": problem_id(problem, config.seed),
            }
        )
    )
    todo = targets(source_snr)
    # (source, seed) jobs, split over --shard i/n (a Slurm array: lisa-mcmc.sbatch)
    jobs = [(js, k) for js in todo for k in args.seeds]
    i, n = (int(v) for v in args.shard.split("/"))
    jobs = jobs[i::n]
    print(
        f"{config.name}: {len(todo)} sources, shard {args.shard}: {len(jobs)} chains -> {directory},"
        f" backend {jax.default_backend()}",
        flush=True,
    )
    for (j, s), k in jobs:
        pages = todo[j, s]
        path = chain_file(directory, j, s, k)
        if path.exists() and not args.force:
            print(f"  {path.name}: cached", flush=True)
            continue
        target = make_target(problem, injections, j, s)
        out = sample_chains(problem, target, args, k)
        # written under a hidden name, renamed when complete: no truncated chain looks cached
        pending = path.with_name(f".partial-{path.name}")
        np.savez(
            pending,
            **out,
            truth=target.truth,
            snr=target.snr,
            window_snr=target.window_snr,
            injection=j,
            source=s,
            pages=np.array(pages),
            f=float(target.f),
            centres=target.centres,
            temps=np.asarray(args.temps),
            nwalker=args.nwalker,
            burn=args.burn,
            keep=args.keep,
            thin=args.thin,
            chain_seed=k,
            config=config.name,
        )
        pending.replace(path)
        print(
            f"  {path.name} ({', '.join(pages)}, SNR {target.snr:.1f}): {out['seconds'] / 60:.1f} min,"
            f" R-hat max {out['rhat'].max():.4f}, ESS min {out['ess'].min():.0f},"
            f" acceptance {np.round(out['acceptance'].mean(0), 2)}",
            flush=True,
        )


def status(config: run_config.Config, problem: LisaGB, args) -> int:
    latents, windows = eval_injections(problem, config.seed)[:2]
    source_snr = np.asarray(jax.vmap(partial(source_snrs, problem))(latents, windows))
    directory = cache_dir(problem, config.seed, args.root)
    missing = [
        chain_file(directory, j, s, k).name
        for (j, s) in targets(source_snr)
        for k in args.seeds
        if not chain_file(directory, j, s, k).exists()
    ]
    if args.count:  # the chains in all, for the array size
        print(len(targets(source_snr)) * len(args.seeds))
    elif args.dir:
        print(directory)
    else:
        print(
            f"{config.name}: {directory}, {'complete' if not missing else 'missing ' + ' '.join(missing)}"
        )
    return 1 if missing else 0


def fold(p: np.ndarray, truth: np.ndarray, t_obs: float) -> np.ndarray:
    """Draws [N, 8] -> COLUMNS coordinates, psi/phi0 folded and lon wrapped onto the truth."""
    p = fold_orientation(p, truth)
    lon = truth[3] + (p[..., 3] - truth[3] + np.pi) % (2 * np.pi) - np.pi
    return np.stack(
        [
            p[..., 0] * t_obs,
            np.log(p[..., 1]),
            np.log(np.abs(p[..., 2])),
            lon,
            p[..., 4],
            p[..., 5],
            np.sin(p[..., 6]),
            p[..., 7],
        ],
        axis=-1,
    )


def report(config: run_config.Config, problem: LisaGB, args) -> None:
    """Flow against MCMC for one eval dir's draws: a table per source, and the chains the corner pages overlay."""
    from scipy.stats import wasserstein_distance

    seed, eval_dir = config.seed, args.eval or newest_eval(config)
    chains = load_chains(problem, seed, args.root, n=10**9)
    assert (
        chains
    ), f"no MCMC for {config.name} in {cache_dir(problem, seed, args.root)}: run it first"
    d = np.load(eval_dir / "draws.npz")
    matched, truth = d["matched"], d["truth"]
    source_snr, quantiles = d["source_snr"], d["quantiles"]

    lines, rows = [], {}
    say = lambda text="": lines.append(text)
    say(
        f"{eval_dir / 'draws.npz'} (epoch {int(d['epoch'])}) against the narrowband MCMC of {config.name}"
        f" ({cache_dir(problem, seed, args.root)}); robust widths (1.4826 MAD), angles folded"
    )
    for (j, s), mcmc in sorted(chains.items(), key=lambda kv: -source_snr[kv[0]]):
        flow = matched[j][:, s]
        a, b = (
            fold(flow, truth[j, s], problem.t_obs),
            fold(mcmc, truth[j, s], problem.t_obs),
        )
        robust_width = lambda x: (
            1.4826 * np.median(np.abs(x - np.median(x, axis=0)), axis=0)
        )
        wa, wb = robust_width(a), robust_width(b)
        lo68, hi68 = np.percentile(a, [16, 84], axis=0)
        lo95, hi95 = np.percentile(a, [2.5, 97.5], axis=0)
        med = np.median(b, axis=0)
        w1 = np.array(
            [wasserstein_distance(a[:, c], b[:, c]) for c in range(N_PARAMS)]
        ) / b.std(0)
        names = ", ".join(targets(source_snr).get((j, s), []))
        say(
            f"\n injection {j} (q{quantiles[j]:.2f}), source {s}, SNR {source_snr[j, s]:.1f}  [{names}]"
        )
        say("                          " + "".join(f"{c:>11s}" for c in COLUMNS))
        say("  MCMC width              " + "".join(f"{v:11.4g}" for v in wb))
        say("  flow / MCMC width       " + "".join(f"{v:11.3g}" for v in wa / wb))
        say("  flow W1 / MCMC sd       " + "".join(f"{v:11.3g}" for v in w1))
        say(
            "  MCMC median in flow 68  "
            + "".join(f"{str(v):>11s}" for v in (med >= lo68) & (med <= hi68))
        )
        say(
            "  MCMC median in flow 95  "
            + "".join(f"{str(v):>11s}" for v in (med >= lo95) & (med <= hi95))
        )
        say(
            f"  flow draws > 1 bin from the MCMC median in f0: {np.mean(np.abs(a[:, 0] - med[0]) > 1):.3f}"
        )
        rows[f"{j}_{s}"] = np.stack([wb, wa / wb, w1])
    text = "\n".join(lines)
    print(text)
    (eval_dir / "mcmc_compare.txt").write_text(text + "\n")
    np.savez(eval_dir / "mcmc_compare.npz", **rows)
    # the chains `plots` overlays on the corner report, N_PLOT per source, in float64: float32 f0 in Hz
    # steps by ~1e-3 bins at 1 mHz, wider than a loud source's whole posterior
    sub = load_chains(problem, seed, args.root)
    np.savez(eval_dir / "mcmc.npz", **{f"{j}_{s}": v for (j, s), v in sub.items()})
    print(f"saved mcmc_compare.txt, mcmc_compare.npz and mcmc.npz -> {eval_dir}")


def newest_eval(config: run_config.Config) -> Path:
    """The run's latest-epoch eval dir with draws.npz at the config's weights and ode_steps."""
    found = sorted(
        (
            path.parent
            for path in config.run_dir.glob(
                f"eval/*-{config.weights}-{config.ode_steps}/draws.npz"
            )
        ),
        key=lambda path: int(path.name.split("-")[0]),
    )
    assert found, f"no eval draws in {config.run_dir / 'eval'}: run canna.lisa.eval first"
    return found[-1]


def main() -> int:
    parser = argparse.ArgumentParser(prog="mcmc", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "status", "report"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, default=MCMC_ROOT)
        p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    r = sub.choices["run"]
    r.add_argument("--nwalker", type=int, default=32)
    r.add_argument(
        "--temps", type=float, nargs="+", default=[1.0, 1.58, 2.51, 3.98, 6.31, 10.0]
    )
    r.add_argument(
        "--chunk", type=int, default=2000, help="iterations per jexplore epoch"
    )
    r.add_argument("--burn", type=int, default=10000)
    r.add_argument("--keep", type=int, default=30000)
    r.add_argument("--thin", type=int, default=10)
    r.add_argument("--force", action="store_true", help="rerun cached chains")
    r.add_argument(
        "--shard", default="0/1", help="i/n: run every n-th chain from the i-th"
    )
    st = sub.choices["status"]
    st.add_argument("--count", action="store_true", help="print the number of chains")
    st.add_argument("--dir", action="store_true", help="print the cache folder")
    rep = sub.choices["report"]
    rep.add_argument(
        "--eval",
        type=Path,
        help="an eval dir with draws.npz (default: the run's latest at its weights and ode_steps)",
    )
    args = parser.parse_args()

    # a run's config.yaml when LISA_RUN names one; report needs one, run and status take a bare config too
    config = (
        run_config.load_run()
        if os.environ.get("LISA_RUN") or args.command == "report"
        else run_config.load()
    )
    problem = LisaGB(**config.problem)
    if args.command == "run":
        run(config, problem, args)
    elif args.command == "status":
        return status(config, problem, args)
    else:
        report(config, problem, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
