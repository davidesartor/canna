"""Narrowband MCMC of the eval sources: the reference posteriors of the corner pages.

One jexplore parallel-tempered ensemble per source, for the sources the corner pages show: the
loudest, median and faintest of the 40 eval sources (gb-*.pdf) and the four of the lowest
window, q0.10. Each source is sampled alone, with the others held at their true values: the
data are the eval's own noise realisation (eval.eval_injections) minus the other sources'
signals. That is the posterior of the source if the others were known exactly, which is the
marginal posterior whenever the sources do not overlap (the independence assumption).

A source's response is zero outside its own `response_points` bins (LisaGB.source_segments),
so its likelihood needs only that piece of the frequency stream:

    log L(theta) = -sum_band (|r - h_theta|^2 - |r|^2) / P   (+ a constant)

with r the residual, exact for the whole window, wherever theta puts the band. On XS a band is
16 bins of a 64-bin window, on B 1024 of 4096; and 8 dimensions converge much faster than the
32 of a joint run (lisa_checks/mcmc_xs.py).

The chains depend only on the problem and the eval seed, not on the network, so they are cached
per problem under outputs/mcmc/eval-<problem id>/ and every run of that problem reuses them
(XS-fd and XS-fd-rope share one set). eval.py overlays them on its corner pages when present;
`report` writes the flow-vs-MCMC table and the pages for a finished run from its eval draws.

Each source is sampled as (log f0, log Mc, log A, sky longitude, sin latitude, psi, sin iota,
phi0) under the training prior, the angles on 2 pi boxes centred on their true values, from
walkers started on the source's Fisher ellipse (as mcmc_xs.py).

    .venv313/bin/python -m canna.lisa.mcmc run --config XS-fd       # jexplore: Python 3.13
    .venv/bin/python -m canna.lisa.mcmc status --config XS-fd
    .venv/bin/python -m canna.lisa.mcmc report --config XS-fd --run outputs/15-lisa-XS-fixed
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import yaml

from .eval import N_QUANTILES, eval_injections, gb_picks, match_sources, scaled_inverse
from .problem import LisaGB

N_PARAMS = 8
NAMES = ["log f0", "log Mc", "log A", "lon", "sin lat", "psi", "sin iota", "phi0"]
ANGLES = [3, 5, 7]
MCMC_ROOT = Path("outputs/mcmc")
# the corner pages' sources: gb-loudest, gb-median, gb-faintest, and every source of these
# injections (q0.10)
PAGE_INJECTIONS = (0,)
# draws per source kept for the corner pages
N_PLOT = 20000


# --- the problem, its eval sources and their cache --------------------------------------------


def load_problem(config: str) -> tuple[LisaGB, int]:
    cfg = yaml.safe_load(open(Path(__file__).parent / "configs" / f"{config}.yaml"))
    return LisaGB(**cfg["problem"]), int(cfg.get("seed", 0))


def problem_id(problem: LisaGB, seed: int) -> str:
    """A short hash of everything the eval injections depend on: the problem and the seed."""
    fields = {
        k: getattr(problem, k)
        for k in (
            "n_sources", "t_obs", "sampling_step", "response_points", "oms_noise",
            "acceleration_noise", "mass_range", "f0_range", "snr_range", "order_sources",
        )
    }
    fields["orbit"], fields["seed"] = type(problem.orbit).__name__, seed
    text = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha1(text.encode()).hexdigest()[:10]


def cache_dir(problem: LisaGB, seed: int, root: Path = MCMC_ROOT) -> Path:
    return root / f"eval-{problem_id(problem, seed)}"


def targets(source_snr: np.ndarray) -> dict[tuple[int, int], list[str]]:
    """(injection, source) -> the page names it appears on."""
    out = {}
    for name, js in gb_picks(source_snr).items():
        out.setdefault(js, []).append(f"gb-{name}")
    for j in PAGE_INJECTIONS:
        for s in range(source_snr.shape[1]):
            out.setdefault((j, s), []).append(f"q{(j + 1) / N_QUANTILES:.2f}")
    return out


def chain_file(directory: Path, j: int, s: int, seed: int) -> Path:
    return directory / f"inj{j}_src{s}_seed{seed}.npz"


def load_chains(
    problem: LisaGB, seed: int, root: Path = MCMC_ROOT, n: int = N_PLOT
) -> dict[tuple[int, int], np.ndarray]:
    """The cached chains of this problem, (injection, source) -> physical draws [n, 8], every
    seed pooled; {} when none have been run."""
    directory = cache_dir(problem, seed, root)
    out = {}
    for path in sorted(directory.glob("inj*_src*_seed*.npz")):
        j, s = (int(v) for v in path.stem.replace("inj", "").replace("src", "").split("_")[:2])
        out.setdefault((j, s), []).append(np.load(path)["physical"])
    rng = np.random.default_rng(0)
    for js, parts in out.items():
        pooled = np.concatenate(parts)
        # the chains' angle boxes are centred on the truth; the flow's angles live in [0, 2 pi)
        pooled[:, ANGLES] = np.mod(pooled[:, ANGLES], 2 * np.pi)
        out[js] = pooled[rng.choice(len(pooled), min(n, len(pooled)), replace=False)]
    return out


# --- one source: coordinates, prior, narrowband likelihood ------------------------------------


def to_physical(theta: jax.Array) -> jax.Array:
    """(..., 8) sampler coordinates -> (..., 8) physical parameters."""
    t = theta
    return jnp.stack(
        [
            jnp.exp(t[..., 0]), jnp.exp(t[..., 1]), jnp.exp(t[..., 2]), t[..., 3],
            jnp.arcsin(t[..., 4]), t[..., 5], jnp.arcsin(t[..., 6]), t[..., 7],
        ],
        axis=-1,
    )


def to_sampler(p: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """(..., 8) physical -> (..., 8) sampler coordinates, angles wrapped onto their boxes."""
    p = np.asarray(p, np.float64)
    t = np.stack(
        [
            np.log(p[..., 0]), np.log(p[..., 1]), np.log(np.abs(p[..., 2])), p[..., 3],
            np.sin(p[..., 4]), p[..., 5], np.sin(p[..., 6]), p[..., 7],
        ],
        axis=-1,
    )
    for i in ANGLES:
        t[..., i] = centres[i] + (t[..., i] - centres[i] + np.pi) % (2 * np.pi) - np.pi
    return t


def chirp_mass_log_density(prior, n_grid: int = 1024, n_m1: int = 8000):
    """log p(log Mc) on a grid, by quadrature over the primary mass.

    ChirpMass draws m1 from a truncated Gaussian mixture and m2 uniform on [m_min, m1]. For a
    given m1, Mc = (m1 m2)^0.6 / (m1 + m2)^0.2 rises monotonically with m2, so
    p(Mc) = int dm1 p(m1) / (m1 - m_min) / (dMc/dm2) at the m2 that gives Mc.
    """
    from scipy.special import ndtr

    m_min, m_max = prior.m_min, prior.m_max
    w, mu, sd = (np.asarray(a, np.float64) for a in (prior.weights, prior.means, prior.stds))
    norm = lambda z: np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    kept = ndtr((m_max - mu) / sd) - ndtr((m_min - mu) / sd)
    m1 = np.linspace(m_min, m_max, n_m1 + 1)[1:]
    p1 = (w * norm((m1[:, None] - mu) / sd) / sd).sum(-1) / (w * kept).sum()
    chirp = lambda a, b: (a * b) ** 0.6 / (a + b) ** 0.2

    lo, hi = prior.support
    log_mc = np.linspace(np.log(lo), np.log(hi), n_grid)
    mc = np.exp(log_mc)[:, None]
    low, high = chirp(m1, m_min)[None], chirp(m1, m1)[None]
    valid = (mc >= low) & (mc <= high)
    a, b = np.full_like(mc * m1, m_min), np.broadcast_to(m1, (n_grid, n_m1)).copy()
    for _ in range(60):  # bisection for m2 on [m_min, m1]
        mid = 0.5 * (a + b)
        below = chirp(m1[None], mid) < mc
        a, b = np.where(below, mid, a), np.where(below, b, mid)
    m2 = 0.5 * (a + b)
    slope = chirp(m1[None], m2) * (0.6 / m2 - 0.2 / (m1[None] + m2))
    integrand = np.where(valid, p1[None] / (m1[None] - m_min) / slope, 0.0)
    density = np.trapezoid(integrand, m1, axis=1)  # p(Mc)
    with np.errstate(divide="ignore"):
        return log_mc, np.log(density * np.exp(log_mc))  # p(log Mc) = p(Mc) Mc


class Source:
    """Source s of eval injection j: its residual data, prior box and narrowband likelihood."""

    def __init__(self, problem: LisaGB, injections, j: int, s: int):
        latents, windows, snrs, _, noise_keys, _ = injections
        self.problem, self.j, self.s = problem, j, s
        self.f = f = windows[j]
        truth_all = np.asarray(latents[j], np.float64)
        self.truth = truth_all[s]
        self.snr = float(problem.snr(jnp.asarray(self.truth)[None], f))
        self.window_snr = float(snrs[j])
        obs = problem.sample_observation(noise_keys[j], latents[j], f)
        others = np.delete(truth_all, s, axis=0)
        self.residual = obs - problem.clean_signal(jnp.asarray(others), f) if len(others) else obs
        self.power = problem.noise_psd(problem.window_freqs(f)) * problem.t_obs / 2.0

        self.centres = np.zeros(N_PARAMS)
        self.centres[ANGLES] = self.truth[ANGLES]
        f_lo, f_hi = (float(v) for v in problem.f0_window(f))
        a_lo, a_hi = (float(v) for v in problem.a_window(f))
        mc_lo, mc_hi = problem.chirp_mass_range
        self.lower = np.array([np.log(f_lo), np.log(mc_lo), np.log(a_lo), 0, -1, 0, -1, 0])
        self.upper = np.array([np.log(f_hi), np.log(mc_hi), np.log(a_hi), 0, 1, 0, 1, 0])
        self.lower[ANGLES], self.upper[ANGLES] = self.centres[ANGLES] - np.pi, self.centres[ANGLES] + np.pi

        grid, table = chirp_mass_log_density(problem.chirp_mass_prior)
        finite = np.isfinite(table)
        self.mc_grid, self.mc_table = jnp.asarray(grid[finite]), jnp.asarray(table[finite])

    def logprior(self, theta: jax.Array) -> jax.Array:
        inside = jnp.all((theta >= self.lower) & (theta <= self.upper))
        lp = jnp.interp(theta[1], self.mc_grid, self.mc_table, left=-jnp.inf, right=-jnp.inf)
        return jnp.where(inside, lp, -jnp.inf)

    def loglik(self, theta: jax.Array) -> jax.Array:
        """log L up to a constant, from the source's own band of the residual."""
        idx, seg = self.problem.source_segments(to_physical(theta)[None], self.f)
        idx, seg = idx[0], seg[0]
        inside = (idx >= 0) & (idx < self.problem.window_bins)  # clean_signal drops the rest
        k = jnp.clip(idx, 0, self.problem.window_bins - 1)
        r, power = self.residual[k], self.power[k]
        d = (jnp.abs(r - seg) ** 2 - jnp.abs(r) ** 2) / power
        return -jnp.sum(jnp.where(inside[:, None], d, 0.0))

    def fisher_cov(self) -> np.ndarray:
        """Covariance of the Fisher ellipse in sampler coordinates, with the prior box as a
        weak Gaussian so unconstrained directions stay finite."""
        t0 = jnp.asarray(to_sampler(self.truth, self.centres))
        power = self.power

        def signal(theta):
            h = self.problem.clean_signal(to_physical(theta)[None], self.f) / jnp.sqrt(power)
            return jnp.concat([h.real.ravel(), h.imag.ravel()])

        jac = jax.jacfwd(signal)(t0)
        fisher = 2.0 * np.asarray(jac.T @ jac)
        width = self.upper - self.lower
        return np.asarray(scaled_inverse(jnp.asarray(fisher + np.diag(12.0 / width**2))))

    def initial_points(self, n: int, key: jax.Array) -> np.ndarray:
        """Walkers on the Fisher ellipse, chirp masses from the prior, clipped into the box."""
        t0 = to_sampler(self.truth, self.centres)
        cov = self.fisher_cov()
        theta = np.array(jr.multivariate_normal(key, jnp.asarray(t0), jnp.asarray(cov), (n,), method="eigh"))
        mc = jax.vmap(self.problem.chirp_mass_prior)(jr.split(jr.fold_in(key, 1), n))
        theta[:, 1] = np.log(np.asarray(mc)).reshape(n)
        pad = 1e-9 * (self.upper - self.lower)
        return np.clip(theta, self.lower + pad, self.upper - pad)


# --- the sampler -------------------------------------------------------------------------------


def autocorr_time(x, c: float = 5.0):
    """Integrated autocorrelation time of chains (nwalker, iters): emcee's estimator, with the
    autocorrelation function averaged over walkers and Sokal's window."""
    y = x - x.mean(axis=1, keepdims=True)
    n = 1 << (2 * y.shape[1] - 1).bit_length()
    acf = np.fft.irfft(np.abs(np.fft.rfft(y, n, axis=1)) ** 2, axis=1)[:, : y.shape[1]]
    acf = (acf / acf[:, :1]).mean(axis=0)
    taus = 2.0 * np.cumsum(acf) - 1.0
    m = np.arange(len(taus)) < c * taus
    window = np.argmin(m) if not m.all() else len(taus) - 1
    return taus[window]


def gelman_rubin(chains, groups: int = 4):
    """Split R-hat over walker groups: chains (nwalker, iters) for one parameter."""
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


def sample(src: Source, args, seed: int) -> dict:
    """Run one source's chains; returns the arrays to save."""
    from jexplore.sampling import SamplingMH  # jexplore needs Python 3.13
    from jexplore.steps import Stretch, TSwap
    from jexplore.steps.de import DEStep

    temps = jnp.asarray(args.temps)
    n_temps = len(args.temps)
    p0 = src.initial_points(args.nwalker * n_temps, jr.key(1000 * seed + 17))
    assert bool(jnp.all(jnp.isfinite(jax.vmap(src.logprior)(jnp.asarray(p0))))), "start outside the prior"
    sampling = SamplingMH(
        dim=N_PARAMS, nwalker=args.nwalker, temps=temps, loglik=src.loglik, logprior=src.logprior
    )
    steps = [
        {TSwap(permute=True).builder: 1.0},
        {DEStep(permute=True).builder: 0.5, Stretch(permute=True).builder: 0.5},
    ]
    sampler = sampling.get_sampler(steps=steps, backend=NullBackend())
    epoch, kept, kept_ll, acc, done = sampling.get_epoch(jnp.asarray(p0)), [], [], [], 0
    total, start = args.burn + args.keep, time.perf_counter()
    while done < total:
        epoch = sampler.run(epoch, niters=args.chunk, nepoch=1, seed=seed * 1_000_003 + done)
        p = np.asarray(epoch.samples.p)[::n_temps]  # cold chains: (nwalker, dim, iters)
        ll = np.asarray(epoch.samples.ll)[::n_temps, 0]
        acc.append(np.asarray(epoch.stats.stats).mean(axis=-1) / np.asarray(epoch.stats.counts))
        if done >= args.burn:
            kept.append(p[:, :, :: args.thin])
            kept_ll.append(ll[:, :: args.thin])
        done += args.chunk
    chains = np.concatenate(kept, axis=-1)  # (nwalker, dim, n)
    rhat = np.array([gelman_rubin(chains[:, d]) for d in range(N_PARAMS)])
    tau = np.array([autocorr_time(chains[:, d]) for d in range(N_PARAMS)]) * args.thin
    samples = chains.transpose(0, 2, 1).reshape(-1, N_PARAMS)
    return dict(
        samples=samples, physical=np.asarray(to_physical(jnp.asarray(samples))),
        loglik=np.concatenate(kept_ll, axis=-1).reshape(-1), rhat=rhat, tau=tau,
        ess=args.nwalker * args.keep / tau, acceptance=np.array(acc),
        seconds=time.perf_counter() - start,
    )


def run(args) -> None:
    problem, seed = load_problem(args.config)
    injections = eval_injections(problem, seed)
    latents, windows = injections[0], injections[1]
    source_snr = np.asarray(jax.vmap(problem.source_snrs)(latents, windows))
    directory = cache_dir(problem, seed, args.root)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "problem.yaml").write_text(
        yaml.safe_dump({"config": args.config, "seed": seed, "problem_id": problem_id(problem, seed)})
    )
    todo = targets(source_snr)
    # (source, seed) jobs, split over --shard i/n (a Slurm array: lisa-mcmc.sbatch)
    jobs = [(js, k) for js in todo for k in args.seeds]
    i, n = (int(v) for v in args.shard.split("/"))
    jobs = jobs[i::n]
    print(f"{args.config}: {len(todo)} sources, shard {args.shard}: {len(jobs)} chains -> {directory},"
          f" backend {jax.default_backend()}", flush=True)
    for (j, s), k in jobs:
        pages = todo[j, s]
        path = chain_file(directory, j, s, k)
        if path.exists() and not args.force:
            print(f"  {path.name}: cached", flush=True)
            continue
        src = Source(problem, injections, j, s)
        out = sample(src, args, k)
        np.savez(
            path, **out, truth=src.truth, snr=src.snr, window_snr=src.window_snr,
            injection=j, source=s, pages=np.array(pages), f=float(src.f), centres=src.centres,
            temps=np.asarray(args.temps), nwalker=args.nwalker, burn=args.burn,
            keep=args.keep, thin=args.thin, chain_seed=k, config=args.config,
        )
        print(f"  {path.name} ({', '.join(pages)}, SNR {src.snr:.1f}): {out['seconds'] / 60:.1f} min,"
              f" R-hat max {out['rhat'].max():.4f}, ESS min {out['ess'].min():.0f},"
              f" acceptance {np.round(out['acceptance'].mean(0), 2)}", flush=True)


def status(args) -> int:
    problem, seed = load_problem(args.config)
    latents, windows = eval_injections(problem, seed)[:2]
    source_snr = np.asarray(jax.vmap(problem.source_snrs)(latents, windows))
    directory = cache_dir(problem, seed, args.root)
    missing = [
        chain_file(directory, j, s, k).name
        for (j, s) in targets(source_snr) for k in args.seeds
        if not chain_file(directory, j, s, k).exists()
    ]
    if args.count:  # the chains in all, for the array size
        print(len(targets(source_snr)) * len(args.seeds))
    elif args.dir:
        print(directory)
    else:
        print(f"{args.config}: {directory}, {'complete' if not missing else 'missing ' + ' '.join(missing)}")
    return 1 if missing else 0


# --- a finished run against the MCMC ------------------------------------------------------------


def fold(p: np.ndarray, truth: np.ndarray, t_obs: float) -> np.ndarray:
    """(N, 8) physical -> comparison coordinates, folded by the likelihood's exact symmetries
    psi -> psi + pi and (psi, phi0) -> (psi + pi/2, phi0 + pi) around the truth (as
    lisa_checks/compare_mcmc.py): f0 in bins, log Mc, log A, lon, sin lat, psi, sin iota, phi0."""
    p = np.array(p, np.float64)
    wrap = lambda a, c: c + (a - c + np.pi) % (2 * np.pi) - np.pi
    k = np.round((p[..., 5] - truth[5]) / (np.pi / 2))
    psi, phi0 = p[..., 5] - k * np.pi / 2, p[..., 7] - k * np.pi
    return np.stack(
        [
            p[..., 0] * t_obs, np.log(p[..., 1]), np.log(np.abs(p[..., 2])),
            wrap(p[..., 3], truth[3]), np.sin(p[..., 4]), psi, np.sin(p[..., 6]),
            wrap(phi0, truth[7]),
        ],
        axis=-1,
    )


COLUMNS = ["f0 [bins]", "log Mc", "log A", "lon", "sin lat", "psi", "sin iota", "phi0"]


def robust_width(x: np.ndarray) -> np.ndarray:
    return 1.4826 * np.median(np.abs(x - np.median(x, axis=0)), axis=0)


def report(args) -> None:
    """Flow against MCMC for one run's eval draws: a table per source, and the corner pages of
    the MCMC'd sources (gb-*, q0.10) remade with the MCMC on them."""
    import matplotlib

    matplotlib.use("Agg")
    from scipy.stats import wasserstein_distance

    from .eval import fisher_samples, gb_page, injection_pages

    problem, seed = load_problem(args.config)
    injections = eval_injections(problem, seed)
    latents, windows, snrs, quantiles, noise_keys, prior_prec = injections
    chains = load_chains(problem, seed, args.root, n=10**9)
    assert chains, f"no MCMC for {args.config} in {cache_dir(problem, seed, args.root)}: run it first"
    d = np.load(args.run / args.draws)
    draws, truth = d["draws"], d["truth"]
    assert np.allclose(truth, np.asarray(latents)), "the eval draws are of other injections"
    source_snr = np.asarray(jax.vmap(problem.source_snrs)(latents, windows))

    lines, rows = [], {}
    say = lambda text="": lines.append(text)
    say(f"{args.run / args.draws} (epoch {int(d['epoch'])}) against the narrowband MCMC of {args.config}"
        f" ({cache_dir(problem, seed, args.root)}); robust widths (1.4826 MAD), angles folded")
    for (j, s), mcmc in sorted(chains.items(), key=lambda kv: -source_snr[kv[0]]):
        flow = match_sources(problem, draws[j], truth[j], windows[j])[:, s]
        a, b = fold(flow, truth[j, s], problem.t_obs), fold(mcmc, truth[j, s], problem.t_obs)
        wa, wb = robust_width(a), robust_width(b)
        lo68, hi68 = np.percentile(a, [16, 84], axis=0)
        lo95, hi95 = np.percentile(a, [2.5, 97.5], axis=0)
        med = np.median(b, axis=0)
        w1 = np.array([wasserstein_distance(a[:, c], b[:, c]) for c in range(N_PARAMS)]) / b.std(0)
        names = ", ".join(targets(source_snr).get((j, s), []))
        say(f"\n injection {j} (q{quantiles[j]:.2f}), source {s}, SNR {source_snr[j, s]:.1f}  [{names}]")
        say("                          " + "".join(f"{c:>11s}" for c in COLUMNS))
        say("  MCMC width              " + "".join(f"{v:11.4g}" for v in wb))
        say("  flow / MCMC width       " + "".join(f"{v:11.3g}" for v in wa / wb))
        say("  flow W1 / MCMC sd       " + "".join(f"{v:11.3g}" for v in w1))
        say("  MCMC median in flow 68  " + "".join(f"{str(v):>11s}" for v in (med >= lo68) & (med <= hi68)))
        say("  MCMC median in flow 95  " + "".join(f"{str(v):>11s}" for v in (med >= lo95) & (med <= hi95)))
        say(f"  flow draws > 1 bin from the MCMC median in f0: {np.mean(np.abs(a[:, 0] - med[0]) > 1):.3f}")
        rows[f"{j}_{s}"] = np.stack([wb, wa / wb, w1])
    text = "\n".join(lines)
    print(text)
    tag = Path(args.draws).stem.replace("eval_draws", "")
    (args.run / f"mcmc_compare{tag}.txt").write_text(text + "\n")
    np.savez(args.run / f"mcmc_compare{tag}.npz", **rows)

    if args.no_pages:
        return
    corner_dir = args.run / f"corner{tag}"
    corner_dir.mkdir(exist_ok=True)
    fisher = {}
    for j in sorted({j for j, _ in chains}):
        fisher[j] = fisher_samples(problem, latents[j], windows[j], noise_keys[j], prior_prec)
    sub = load_chains(problem, seed, args.root)
    for j in PAGE_INJECTIONS:
        if all((j, s) in sub for s in range(problem.n_sources)):
            mcmc = np.stack([sub[j, s] for s in range(problem.n_sources)], axis=1)
            injection_pages(corner_dir, f"q{quantiles[j]:.2f}", quantiles[j], snrs[j], draws[j],
                            fisher[j], truth[j], mcmc)
    for name, (j, s) in gb_picks(source_snr).items():
        if (j, s) in sub:
            flow = match_sources(problem, draws[j], truth[j], windows[j])[:, s]
            gb_page(corner_dir, name, source_snr.size, source_snr[j, s], f"q{quantiles[j]:.2f}", s,
                    flow, fisher[j][:, s], truth[j, s], sub[j, s])
    print(f"corner pages with the MCMC -> {corner_dir}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "status", "report"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--root", type=Path, default=MCMC_ROOT)
        p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    r = sub.choices["run"]
    r.add_argument("--nwalker", type=int, default=32)
    r.add_argument("--temps", type=float, nargs="+", default=[1.0, 1.58, 2.51, 3.98, 6.31, 10.0])
    r.add_argument("--chunk", type=int, default=2000, help="iterations per jexplore epoch")
    r.add_argument("--burn", type=int, default=10000)
    r.add_argument("--keep", type=int, default=30000)
    r.add_argument("--thin", type=int, default=10)
    r.add_argument("--force", action="store_true", help="rerun cached chains")
    r.add_argument("--shard", default="0/1", help="i/n: run every n-th chain from the i-th")
    st = sub.choices["status"]
    st.add_argument("--count", action="store_true", help="print the number of chains")
    st.add_argument("--dir", action="store_true", help="print the cache folder")
    rep = sub.choices["report"]
    rep.add_argument("--run", type=Path, required=True, help="a run folder with eval_draws.npz")
    rep.add_argument("--draws", default="eval_draws.npz")
    rep.add_argument("--no_pages", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args)
    elif args.command == "status":
        return status(args)
    else:
        report(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
