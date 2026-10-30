"""Joint MCMC of the eval injections: one jexplore ensemble per injection over all its sources, on the window's likelihood."""

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
from einops import rearrange
from jaxtyping import Array, Float

from . import config as run_config
from .injections import (
    N_PER_WINDOW,
    N_WINDOWS,
    eval_injections,
    fisher_samples,
    match_sources,
)
from .physics import (
    FDOT_STD,
    SOURCE_COVARIANCE,
    SOURCE_INTERCEPT,
    SOURCE_MEAN,
    SOURCE_RESIDUAL,
    SOURCE_SLOPE,
    log10_fdot_mean,
    sky_log_density,
    source_log_density,
)
from .plots import fold_orientation
from .samples import load as load_samples
from .samples import pack as pack_samples
from .problem import LisaGB, f0_window, window_power
from .simulator import sample_observation, snr, source_segments

N_PARAMS = 8
NAMES = ["log f0", "log fdot", "log A", "lon", "lat", "psi", "sin iota", "phi0"]
ANGLES = [3, 5, 7]
MCMC_ROOT = Path("outputs/mcmc")
# the box's log fdot and log A reach this many conditional stds past the prior's mean across the window
BOX_SIGMAS = 8.0
# the chains keep the prior's old 5-sigma fdot cut, and with it their cache ids, until the forward model is final
FDOT_SIGMAS = 5.0
# joint draws per injection in the eval catalog's reference, reports/mcmc/data/eval-<problem id>.npz
N_KEEP = 1024
REFERENCE_ROOT = Path("reports/mcmc/data")
COLUMNS = ["f0 [bins]", "log fdot", "log A", "lon", "lat", "psi", "sin iota", "phi0"]


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
            "f0_range",
        )
    }
    # fields the problem has since dropped, at their only values: keeps existing chain caches' ids
    fields["sampling_step"], fields["order_sources"], fields["orbit"] = (
        0.25,
        True,
        "EqualArmlengthOrbits",
    )
    fields["source_prior"] = [SOURCE_MEAN.tolist(), SOURCE_COVARIANCE.tolist(), FDOT_SIGMAS]
    fields["sky_prior"] = "galactic"
    fields["seed"] = seed
    fields["eval_grid"] = [N_WINDOWS, N_PER_WINDOW]
    if problem.visible_snr:
        fields["visible_snr"] = problem.visible_snr
    text = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha1(text.encode()).hexdigest()[:10]


def cache_dir(problem: LisaGB, seed: int, root: Path = MCMC_ROOT) -> Path:
    return root / f"joint-{problem_id(problem, seed)}"


def reference_file(problem: LisaGB, seed: int) -> Path:
    return REFERENCE_ROOT / f"eval-{problem_id(problem, seed)}.npz"


def load_reference(problem: LisaGB, seed: int) -> dict[str, np.ndarray] | None:
    """The eval catalog's packed MCMC reference: mcmc and fisher [J, N, S, 8] (NaN where no chain), truth, scores; None if not packed."""
    path = reference_file(problem, seed)
    return load_samples(path) if path.exists() else None


def chain_file(directory: Path, j: int, seed: int) -> Path:
    return directory / f"inj{j}_seed{seed}.npz"


def load_chains(
    problem: LisaGB, seed: int, root: Path = MCMC_ROOT, n: int = N_KEEP
) -> dict[int, np.ndarray]:
    """The cached chains, injection -> joint physical draws [n, S, 8] over every seed; {} if none."""
    out = {}
    for path in sorted(cache_dir(problem, seed, root).glob("inj*_seed*.npz")):
        j = int(path.stem.removeprefix("inj").split("_")[0])
        try:
            physical = np.load(path)["physical"]
        except (OSError, ValueError, KeyError) as error:  # a damaged file: skip it, loudly
            print(f"[mcmc] skipping {path}: {error}", file=sys.stderr)
            continue
        out.setdefault(j, []).append(physical)
    rng = np.random.default_rng(0)
    for j, parts in out.items():
        pooled = np.concatenate(parts)
        out[j] = pooled[rng.choice(len(pooled), min(n, len(pooled)), replace=False)]
    return out


def to_physical(theta: jax.Array) -> jax.Array:
    """Sampler coordinates (log f0, log fdot, log A, lon, lat, psi, sin iota, phi0) -> physical."""
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


class Window(NamedTuple):
    """Eval injection j with every source free: its data, the sources' prior boxes [S, 8] and Fisher draws."""

    j: int
    f: Array
    truth: np.ndarray
    snr: np.ndarray
    window_snr: float
    obs: Array
    power: Array
    centres: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    fisher: np.ndarray


def make_window(problem: LisaGB, injections, j: int) -> Window:
    latents, windows, snrs, _, noise_keys, prior_prec = injections
    f = windows[j]
    truth = np.asarray(latents[j], np.float64)

    centres = np.zeros_like(truth)
    centres[:, ANGLES] = truth[:, ANGLES]
    f_lo, f_hi = (float(v) for v in f0_window(problem, f))
    # log10 fdot and log10 A's conditional means at the window's ends; the top of fdot is the chains' cut
    mean_lo, mean_hi = (SOURCE_INTERCEPT + SOURCE_SLOPE * np.log10(v) for v in (f_lo, f_hi))
    reach = BOX_SIGMAS * np.sqrt(np.diag(SOURCE_RESIDUAL))
    log10_lo = mean_lo - reach
    log10_hi = mean_hi + np.array([FDOT_SIGMAS * FDOT_STD, reach[1]])
    box = lambda log_f0, log10, lat, sin_iota: np.array([log_f0, *np.log(10) * log10, 0, lat, 0, sin_iota, 0])
    lower = np.tile(box(np.log(f_lo), log10_lo, -np.pi / 2, -1), (truth.shape[0], 1))
    upper = np.tile(box(np.log(f_hi), log10_hi, np.pi / 2, 1), (truth.shape[0], 1))
    lower[:, ANGLES], upper[:, ANGLES] = centres[:, ANGLES] - np.pi, centres[:, ANGLES] + np.pi

    return Window(
        j=j,
        f=f,
        truth=truth,
        snr=np.asarray(snr(problem, latents[j][:, None, :], f)),
        window_snr=float(snrs[j]),
        obs=sample_observation(problem, noise_keys[j], latents[j], f),
        power=window_power(problem, f),
        centres=centres,
        lower=lower,
        upper=upper,
        fisher=fisher_samples(problem, latents[j], f, noise_keys[j], prior_prec),
    )


def log_prior(window: Window, theta: Float[Array, "D"]) -> Float[Array, ""]:
    """The sources' independent priors, each in its box and under the fdot cut."""
    t = rearrange(theta, "(s k) -> s k", k=N_PARAMS)
    inside = jnp.all((t >= window.lower) & (t <= window.upper))
    log10_source = t[:, :3] / jnp.log(10.0)
    ceiling = log10_fdot_mean(10.0 ** log10_source[:, 0]) + FDOT_SIGMAS * FDOT_STD
    lp = jnp.where(
        log10_source[:, 1] <= ceiling, jax.vmap(source_log_density)(log10_source), -jnp.inf
    )
    lp = lp + jax.vmap(sky_log_density)(t[:, 3], t[:, 4])
    return jnp.where(inside, jnp.sum(lp), -jnp.inf)


def log_likelihood(
    problem: LisaGB, window: Window, theta: Float[Array, "D"]
) -> Float[Array, ""]:
    """-sum |o - h|^2 / P over the window, one jaxgb call."""
    p = to_physical(rearrange(theta, "(s k) -> s k", k=N_PARAMS))
    idx, seg = source_segments(problem, p, window.f)
    # the bins outside the window are dropped, as in clean_signal
    inside = (idx >= 0) & (idx < problem.window_bins)
    k = jnp.clip(idx, 0, problem.window_bins - 1)
    seg = jnp.where(inside[..., None], seg, 0.0)
    signal = jnp.zeros((problem.window_bins, 3), seg.dtype).at[k].add(seg)
    return -jnp.sum(jnp.abs(window.obs - signal) ** 2 / window.power)


def initial_points(window: Window, n: int, key: jax.Array) -> Float[np.ndarray, "N D"]:
    """Walkers from the Fisher draws, fdot from the prior at each f0, clipped into the boxes and under the fdot cut."""
    n_sources, n_fisher = window.truth.shape[0], window.fisher.shape[0]
    pick = np.asarray(jr.choice(key, n_fisher, (n,), replace=n_fisher < n))
    p = window.fisher[pick].copy()
    residual_fdot = FDOT_STD * np.asarray(jr.normal(jr.fold_in(key, 1), (n, n_sources)))
    p[..., 1] = 10.0 ** (np.asarray(log10_fdot_mean(p[..., 0])) + residual_fdot)

    theta = np.stack([to_sampler(p[:, s], window.centres[s]) for s in range(n_sources)], axis=1)
    pad = 1e-9 * (window.upper - window.lower)
    theta = np.clip(theta, window.lower + pad, window.upper - pad)
    ceiling = log10_fdot_mean(np.exp(theta[..., 0])) + FDOT_SIGMAS * FDOT_STD
    theta[..., 1] = np.minimum(theta[..., 1], np.log(10) * ceiling - pad[:, 1])
    return rearrange(theta, "n s k -> n (s k)")


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


def sample_chains(problem: LisaGB, window: Window, args, seed: int) -> dict:
    """Run one injection's chains; returns the arrays to save."""
    from jexplore.sampling import SamplingMH  # jexplore needs Python 3.13
    from jexplore.steps import Stretch, TSwap
    from jexplore.steps.de import DEStep

    n_temps = len(args.temps)
    p0 = initial_points(window, args.nwalker * n_temps, jr.key(1000 * seed + 17))
    dim = p0.shape[1]
    assert bool(
        jnp.all(jnp.isfinite(jax.vmap(partial(log_prior, window))(jnp.asarray(p0))))
    ), "start outside the prior"
    sampling = SamplingMH(
        dim=dim,
        nwalker=args.nwalker,
        temps=jnp.asarray(args.temps),
        loglik=partial(log_likelihood, problem, window),
        logprior=partial(log_prior, window),
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
            kept.append(p[:, :, :: args.thin].copy())
            kept_ll.append(ll[:, :: args.thin].copy())
        done += args.chunk
        print(
            f"    [{done}/{total}] {time.perf_counter() - start:.0f} s, cold log L median"
            f" {np.median(ll[:, -1]):.2f}, acceptance (swap, move) {np.round(acc[-1], 3)}",
            flush=True,
        )
    chains = np.concatenate(kept, axis=-1)  # (nwalker, dim, n)
    rhat = np.array([gelman_rubin(chains[:, d]) for d in range(dim)])
    tau = np.array([autocorr_time(chains[:, d]) for d in range(dim)]) * args.thin
    samples = rearrange(chains, "w d n -> (w n) d")
    per_source = lambda a: rearrange(a, "... (s k) -> ... s k", k=N_PARAMS)
    return dict(
        samples=samples,
        physical=np.asarray(to_physical(jnp.asarray(per_source(samples)))),
        loglik=np.concatenate(kept_ll, axis=-1).reshape(-1),
        rhat=per_source(rhat),
        tau=per_source(tau),
        ess=per_source(args.nwalker * args.keep / tau),
        acceptance=np.array(acc),
        seconds=time.perf_counter() - start,
    )


def run(config: run_config.Config, problem: LisaGB, args) -> None:
    injections = eval_injections(problem, config.seed)
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
    # (injection, seed) jobs, split over --shard i/n (a Slurm array: lisa-mcmc.sbatch)
    n_injections = len(injections[1])
    jobs = [(j, k) for j in range(n_injections) for k in args.seeds]
    i, n = (int(v) for v in args.shard.split("/"))
    jobs = jobs[i::n]
    print(
        f"{config.name}: {n_injections} injections, shard {args.shard}: {len(jobs)} chains -> {directory},"
        f" backend {jax.default_backend()}",
        flush=True,
    )
    for j, k in jobs:
        path = chain_file(directory, j, k)
        if path.exists() and not args.force:
            print(f"  {path.name}: cached", flush=True)
            continue
        window = make_window(problem, injections, j)
        print(
            f"  {path.name}: window SNR {window.window_snr:.1f}, sources SNR {np.round(window.snr, 1)}",
            flush=True,
        )
        out = sample_chains(problem, window, args, k)
        # written under a hidden name, renamed when complete: no truncated chain looks cached
        pending = path.with_name(f".partial-{path.name}")
        np.savez(
            pending,
            **out,
            truth=window.truth,
            snr=window.snr,
            window_snr=window.window_snr,
            injection=j,
            f=float(window.f),
            centres=window.centres,
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
            f"  {path.name}: {out['seconds'] / 3600:.1f} h, R-hat max {out['rhat'].max():.4f},"
            f" ESS min {out['ess'].min():.0f}, acceptance {np.round(out['acceptance'].mean(0), 2)}",
            flush=True,
        )


def status(config: run_config.Config, problem: LisaGB, args) -> int:
    n_injections = N_WINDOWS * N_PER_WINDOW
    directory = cache_dir(problem, config.seed, args.root)
    missing = [
        chain_file(directory, j, k).name
        for j in range(n_injections)
        for k in args.seeds
        if not chain_file(directory, j, k).exists()
    ]
    if args.count:  # the chains in all, for the array size
        print(n_injections * len(args.seeds))
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


def pack(config: run_config.Config, problem: LisaGB, args) -> None:
    """The eval catalog's reference: N_KEEP joint MCMC draws per injection, Fisher draws and the MCMC's SBC scores, into
    reports/mcmc/data/eval-<problem id>.npz; then its corner and validation pages beside it."""
    from . import plots, validate
    from .chart import sample_flow

    seed = config.seed
    latents, windows, snrs, grid, noise_keys, prior_prec_p = eval_injections(problem, seed)
    truth = np.asarray(latents, np.float64)
    n_injections, n_sources = truth.shape[:2]
    directory = cache_dir(problem, seed, args.root)
    chains = load_chains(problem, seed, args.root, n=N_KEEP)
    assert chains, f"no MCMC for {config.name} in {directory}: run it first"

    # each draw's sources relabelled onto the truth, angles back on the truth-centred box; NaN for an injection without chains
    mcmc = np.full((n_injections, N_KEEP, n_sources, N_PARAMS), np.nan)
    for j, draws in chains.items():
        if len(draws) < N_KEEP:
            print(f"[mcmc] injection {j}: {len(draws)} draws, skipped", file=sys.stderr)
            continue
        draws = match_sources(problem, draws, truth[j], windows[j])
        t = truth[j][:, ANGLES]
        draws[..., ANGLES] = t + (draws[..., ANGLES] - t + np.pi) % (2 * np.pi) - np.pi
        mcmc[j] = draws

    # per source: the worst R-hat over its injection's chains, their summed least ESS; and how many seeds
    rhat = np.full((n_injections, n_sources), np.nan)
    ess = np.zeros((n_injections, n_sources))
    n_seeds = np.zeros(n_injections, int)
    for path in directory.glob("inj*_seed*.npz"):
        try:
            with np.load(path) as c:
                j = int(c["injection"])
                rhat[j] = np.fmax(rhat[j], c["rhat"].max(-1))
                ess[j] += c["ess"].min(-1)
                n_seeds[j] += 1
        except (OSError, ValueError, KeyError):
            continue

    fisher, source_snr = [], []
    for j in range(n_injections):
        draws = fisher_samples(problem, latents[j], windows[j], noise_keys[j], prior_prec_p)
        fisher.append(draws[: plots.PAGE_DRAWS["fisher"]])
        source_snr.append(np.asarray(snr(problem, latents[j][:, None, :], windows[j])))

    # SBC of the MCMC on the injections with every source run: the truth, then validate's draw count of draws
    validated = np.flatnonzero(np.isfinite(mcmc).all(axis=(1, 2, 3)))
    tarp_keys = jr.split(jr.fold_in(jr.key(seed), 5), n_injections)
    rows = []
    for j in validated:
        f = windows[j]
        o = sample_observation(problem, noise_keys[j], latents[j], f)
        theta = jnp.concat([latents[j][None], jnp.asarray(mcmc[j, : validate.N_DRAWS])])
        rows.append(
            jax.tree.map(
                np.asarray,
                dict(
                    f=f,
                    tarp_ref=sample_flow(problem, tarp_keys[j], f),
                    **validate.quantities(problem, theta, o, f),
                ),
            )
        )
    scores = {}
    if len(validated) >= 10:
        scores = validate.summarise(
            {k: np.stack([r[k] for r in rows]) for k in rows[0]},
            seed,
            f"joint MCMC on {len(validated)} eval injections",
        )

    path = reference_file(problem, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **pack_samples("mcmc", mcmc, truth, axis=1),
        **pack_samples("fisher", np.stack(fisher), truth, axis=1),
        truth=truth,
        f=np.asarray(windows),
        snr=snrs,
        source_snr=np.stack(source_snr),
        grid=grid,
        rhat=rhat,
        ess=ess,
        n_seeds=n_seeds,
        validated=validated,
        problem_id=problem_id(problem, seed),
        config=config.name,
        seed=seed,
        fdot_sigmas=FDOT_SIGMAS,
        **scores,
    )
    print(f"saved {path}: {len(validated)} of {n_injections} injections", flush=True)
    for page in plots.mcmc_pages(path, problem):
        print(page, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(prog="mcmc", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "status", "pack"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, default=MCMC_ROOT)
        p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    r = sub.choices["run"]
    r.add_argument("--nwalker", type=int, default=64)
    r.add_argument(
        "--temps", type=float, nargs="+", default=[1.0, 1.58, 2.51, 3.98, 6.31, 10.0]
    )
    r.add_argument(
        "--chunk", type=int, default=2000, help="iterations per jexplore epoch"
    )
    r.add_argument("--burn", type=int, default=100000)
    r.add_argument("--keep", type=int, default=300000)
    r.add_argument("--thin", type=int, default=100)
    r.add_argument("--force", action="store_true", help="rerun cached chains")
    r.add_argument(
        "--shard", default="0/1", help="i/n: run every n-th chain from the i-th"
    )
    st = sub.choices["status"]
    st.add_argument("--count", action="store_true", help="print the number of chains")
    st.add_argument("--dir", action="store_true", help="print the cache folder")
    args = parser.parse_args()

    # a run's config.yaml when LISA_RUN names one, else a bare config: the chains depend on the problem alone
    config = run_config.load_run() if os.environ.get("LISA_RUN") else run_config.load()
    problem = LisaGB(**config.problem)
    if args.command == "run":
        run(config, problem, args)
    elif args.command == "status":
        return status(config, problem, args)
    else:
        pack(config, problem, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
