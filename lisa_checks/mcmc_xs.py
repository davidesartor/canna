"""Flow against MCMC on XS eval windows (paper appendix "Comparison with MCMC").

A joint jexplore MCMC over the four sources of one XS eval window: 32 parameters, the
training prior, and LisaGB.log_likelihood on the same noise realisation the flow saw (eval.py's
keys). The cold chains, thinned, go to outputs/mcmc/window<j>.npz with their diagnostics.

Each source is sampled as (log f0, log Mc, log A, sky longitude, sin latitude, psi, sin iota,
phi0). Under the prior the first and third are uniform on their window's boxes, the chirp mass
follows ChirpMass (its density by quadrature, `chirp_mass_log_density`), the two sines are
uniform on [-1, 1], and the three angles uniform on a circle. Each angle is sampled on a 2 pi
box centred on its true value: a full period, so the prior is unchanged, and no narrow mode is
split by the box edge. The walkers start from the window's Fisher draws (eval-tools), with the
chirp mass drawn from its prior.

jexplore needs Python 3.13, so run it from the side venv, from the repo root:

    .venv313/bin/python lisa_checks/mcmc_xs.py --window 9
"""

import argparse
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=r"Using `field\(init=False\)`")

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import yaml

import canna.lisa as lisa  # also switches on float64
from canna.lisa import LisaGB
from canna.lisa.eval import N_CANDIDATES, N_QUANTILES

N_PARAMS = 8
NAMES = ["log f0", "log Mc", "log A", "lon", "sin lat", "psi", "sin iota", "phi0"]
ANGLES = [3, 5, 7]
FISHER = Path("outputs/lisa-XS/eval-tools/fisher_eval_injections.npz")


def eval_injections(problem: LisaGB, seed: int):
    """The ten eval injections and their noise keys, exactly as eval.py draws them."""
    key_pick, key_window, key_noise = jr.split(jr.key(seed), 3)
    windows = jax.vmap(problem.sample_f)(jr.split(key_window, N_CANDIDATES))
    latents = jax.vmap(problem.sample_physical)(jr.split(key_pick, N_CANDIDATES), windows)
    snrs = np.asarray(jax.lax.map(lambda pc: problem.snr(*pc), (latents, windows)))
    quantiles = np.linspace(1.0 / N_QUANTILES, 1.0, N_QUANTILES)
    chosen = np.argsort(snrs)[np.round(quantiles * (N_CANDIDATES - 1)).astype(int)]
    return latents[chosen], windows[chosen], snrs[chosen], jr.split(key_noise, N_QUANTILES)


def chirp_mass_log_density(prior, n_grid: int = 1024, n_m1: int = 8000):
    """log p(log Mc) on a grid, by quadrature over the primary mass.

    ChirpMass draws m1 from a truncated Gaussian mixture and m2 uniform on [m_min, m1]. For a
    given m1, Mc = (m1 m2)^0.6 / (m1 + m2)^0.2 rises monotonically with m2, so
    p(Mc) = int dm1 p(m1) / (m1 - m_min) / (dMc/dm2) at the m2 that gives Mc.
    """
    m_min, m_max = prior.m_min, prior.m_max
    w, mu, sd = (np.asarray(a, np.float64) for a in (prior.weights, prior.means, prior.stds))
    norm = lambda z: np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    from scipy.special import ndtr

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


def to_physical(theta):
    """(4 * 8,) sampler coordinates -> (4, 8) physical parameters."""
    t = theta.reshape(-1, N_PARAMS)
    return jnp.stack(
        [
            jnp.exp(t[:, 0]),
            jnp.exp(t[:, 1]),
            jnp.exp(t[:, 2]),
            t[:, 3],
            jnp.arcsin(t[:, 4]),
            t[:, 5],
            jnp.arcsin(t[:, 6]),
            t[:, 7],
        ],
        axis=-1,
    )


def to_sampler(p, centres):
    """(..., 4, 8) physical -> (..., 32) sampler coordinates, angles wrapped onto their boxes."""
    p = np.asarray(p, np.float64)
    t = np.stack(
        [
            np.log(p[..., 0]),
            np.log(p[..., 1]),
            np.log(np.abs(p[..., 2])),
            p[..., 3],
            np.sin(p[..., 4]),
            p[..., 5],
            np.sin(p[..., 6]),
            p[..., 7],
        ],
        axis=-1,
    )
    for i in ANGLES:
        t[..., i] = centres[..., i] + (t[..., i] - centres[..., i] + np.pi) % (2 * np.pi) - np.pi
    return t.reshape(*t.shape[:-2], -1)


def setup(window: int, config: str = "XS-late-768-cool"):
    cfg = yaml.safe_load(open(Path(lisa.__file__).parent / "configs" / f"{config}.yaml"))
    problem = LisaGB(**cfg["problem"])
    latents, windows, snrs, noise_keys = eval_injections(problem, cfg["seed"])
    truth, f = np.asarray(latents[window], np.float64), windows[window]
    obs = problem.sample_observation(noise_keys[window], latents[window], f)
    source_snr = np.asarray(problem.snr(jnp.asarray(truth)[:, None, :], f))

    n_sources = problem.n_sources
    centres = np.tile(np.zeros(N_PARAMS), (n_sources, 1))
    centres[:, ANGLES] = truth[:, ANGLES]
    f_lo, f_hi = (float(v) for v in problem.f0_window(f))
    a_lo, a_hi = (float(v) for v in problem.a_window(f))
    mc_lo, mc_hi = problem.chirp_mass_range
    lower = np.array([np.log(f_lo), np.log(mc_lo), np.log(a_lo), 0, -1, 0, -1, 0], np.float64)
    upper = np.array([np.log(f_hi), np.log(mc_hi), np.log(a_hi), 0, 1, 0, 1, 0], np.float64)
    lower, upper = np.tile(lower, (n_sources, 1)), np.tile(upper, (n_sources, 1))
    lower[:, ANGLES], upper[:, ANGLES] = centres[:, ANGLES] - np.pi, centres[:, ANGLES] + np.pi

    grid, table = chirp_mass_log_density(problem.chirp_mass_prior)
    finite = np.isfinite(table)
    grid_j, table_j = jnp.asarray(grid[finite]), jnp.asarray(table[finite])
    lower_j, upper_j = jnp.asarray(lower.reshape(-1)), jnp.asarray(upper.reshape(-1))

    def logprior(theta):
        inside = jnp.all((theta >= lower_j) & (theta <= upper_j))
        log_mc = theta.reshape(-1, N_PARAMS)[:, 1]
        lp = jnp.interp(log_mc, grid_j, table_j, left=-jnp.inf, right=-jnp.inf).sum()
        return jnp.where(inside, lp, -jnp.inf)

    def loglik(theta):
        return problem.log_likelihood(to_physical(theta), obs, f)

    return dict(
        problem=problem, truth=truth, f=f, obs=obs, snr=source_snr, window_snr=snrs[window],
        centres=centres, lower=lower, upper=upper, logprior=logprior, loglik=loglik,
        mc_grid=grid, mc_table=table,
    )


def initial_points(s, window: int, n_chains: int, key):
    """Fisher draws of the window, chirp masses from the prior, clipped into the boxes."""
    fisher = np.load(FISHER)
    assert np.allclose(fisher["truth"][window], s["truth"]), "cached Fisher is for other injections"
    draws = fisher["fisher"][window]
    pick = np.asarray(jr.choice(key, draws.shape[0], (n_chains,), replace=draws.shape[0] < n_chains))
    p = draws[pick].copy()
    p[..., 1] = np.asarray(
        jax.vmap(s["problem"].chirp_mass_prior)(jr.split(jr.fold_in(key, 1), n_chains * p.shape[1]))
    ).reshape(p.shape[:2])
    theta = to_sampler(p, s["centres"]).reshape(n_chains, -1, N_PARAMS)
    lo, hi = s["lower"], s["upper"]
    pad = 1e-9 * (hi - lo)
    return np.clip(theta, lo + pad, hi - pad).reshape(n_chains, -1)


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
    """jexplore backend that keeps nothing: this script thins the cold chains itself."""

    burn = 0

    def reset(self):
        pass

    def ingest(self, epoch, eind=-1):
        return self


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, required=True, help="eval injection index, 0-9")
    parser.add_argument("--nwalker", type=int, default=64)
    parser.add_argument("--temps", type=float, nargs="+", default=[1.0, 1.58, 2.51, 3.98, 6.31, 10.0])
    parser.add_argument("--chunk", type=int, default=2000, help="iterations per jexplore epoch")
    parser.add_argument("--burn", type=int, default=20000)
    parser.add_argument("--keep", type=int, default=60000)
    parser.add_argument("--thin", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/mcmc"))
    parser.add_argument("--bench", action="store_true", help="time the likelihood and stop")
    args = parser.parse_args()

    from jexplore.sampling import SamplingMH  # jexplore needs Python 3.13
    from jexplore.steps import Stretch, TSwap
    from jexplore.steps.de import DEStep

    s = setup(args.window)
    temps = jnp.asarray(args.temps)
    n_temps, dim = len(args.temps), s["truth"].size
    n_chains = args.nwalker * n_temps
    print(f"window {args.window}: window SNR {s['window_snr']:.1f}, sources SNR "
          f"{np.round(s['snr'], 1)}, f {float(s['f']):.6e} Hz", flush=True)
    print(f"{dim} dims, {args.nwalker} walkers x {n_temps} temps = {n_chains} chains, "
          f"backend {jax.default_backend()}", flush=True)

    p0 = initial_points(s, args.window, n_chains, jr.key(args.seed + 100))
    lp0 = jax.vmap(s["logprior"])(jnp.asarray(p0))
    assert bool(jnp.all(jnp.isfinite(lp0))), "an initial point is outside the prior"
    truth_theta = to_sampler(s["truth"], s["centres"])
    print(f"log L at truth {float(s['loglik'](jnp.asarray(truth_theta))):.2f}, "
          f"log prior at truth {float(s['logprior'](jnp.asarray(truth_theta))):.2f}", flush=True)

    if args.bench:
        batch = jax.jit(jax.vmap(s["loglik"]))
        batch(jnp.asarray(p0)).block_until_ready()
        start = time.perf_counter()
        for _ in range(20):
            batch(jnp.asarray(p0)).block_until_ready()
        print(f"log L over {n_chains} chains: {(time.perf_counter() - start) / 20 * 1e3:.2f} ms")
        return

    sampling = SamplingMH(
        dim=dim, nwalker=args.nwalker, temps=temps, loglik=s["loglik"], logprior=s["logprior"]
    )
    steps = [
        {TSwap(permute=True).builder: 1.0},
        {DEStep(permute=True).builder: 0.5, Stretch(permute=True).builder: 0.5},
    ]
    sampler = sampling.get_sampler(steps=steps, backend=NullBackend())

    epoch, kept, kept_ll, acc, done = sampling.get_epoch(jnp.asarray(p0)), [], [], [], 0
    total = args.burn + args.keep
    start = time.perf_counter()
    while done < total:
        epoch = sampler.run(epoch, niters=args.chunk, nepoch=1, seed=args.seed + done)
        p = np.asarray(epoch.samples.p)[::n_temps]  # cold chains: (nwalker, dim, iters)
        ll = np.asarray(epoch.samples.ll)[::n_temps, 0]
        acc.append(np.asarray(epoch.stats.stats).mean(axis=-1) / np.asarray(epoch.stats.counts))
        if done >= args.burn:
            kept.append(p[:, :, :: args.thin])
            kept_ll.append(ll[:, :: args.thin])
        done += args.chunk
        print(f"[{done}/{total}] {time.perf_counter() - start:.0f} s, cold log L median "
              f"{np.median(ll[:, -1]):.2f}, acceptance (swap, move) {np.round(acc[-1], 3)}",
              flush=True)

    chains = np.concatenate(kept, axis=-1)  # (nwalker, dim, n)
    lls = np.concatenate(kept_ll, axis=-1)
    elapsed = time.perf_counter() - start
    rhat = np.array([gelman_rubin(chains[:, d]) for d in range(dim)])
    tau = np.array([autocorr_time(chains[:, d]) for d in range(dim)]) * args.thin
    ess = args.nwalker * args.keep / tau
    print(f"\n{elapsed / 60:.1f} min; R-hat max {rhat.max():.4f}; autocorrelation time "
          f"{tau.min():.0f}-{tau.max():.0f} iterations; ESS min {ess.min():.0f}")
    for src in range(s["truth"].shape[0]):
        sl = slice(src * N_PARAMS, (src + 1) * N_PARAMS)
        print(f"source {src} (SNR {s['snr'][src]:.1f}): R-hat {np.round(rhat[sl], 4)}, "
              f"ESS {np.round(ess[sl]).astype(int)}")

    args.out.mkdir(parents=True, exist_ok=True)
    samples = chains.transpose(0, 2, 1).reshape(-1, dim)
    physical = np.asarray(jax.vmap(to_physical)(jnp.asarray(samples)))
    np.savez(
        args.out / f"window{args.window}_s{args.seed}.npz",
        samples=samples, physical=physical, loglik=lls.reshape(-1), truth=s["truth"],
        snr=s["snr"], window_snr=s["window_snr"], f=float(s["f"]), centres=s["centres"],
        rhat=rhat, tau=tau, ess=ess, acceptance=np.array(acc), temps=np.asarray(temps),
        nwalker=args.nwalker, burn=args.burn, keep=args.keep, thin=args.thin, seconds=elapsed,
    )
    print(f"saved {args.out / f'window{args.window}_s{args.seed}.npz'}")


if __name__ == "__main__":
    main()
