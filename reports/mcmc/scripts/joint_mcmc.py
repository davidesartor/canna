"""Joint jexplore MCMC over every source of one narrowband eval window, on the flow's own noise realisation."""

import argparse
import sys
import time
import warnings
from functools import partial
from pathlib import Path
from typing import NamedTuple

warnings.filterwarnings("ignore", message=r"Using `field\(init=False\)`")

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import yaml
from einops import rearrange
from jaxtyping import Array, Float

from canna.lisa import config as run_config
from canna.lisa.injections import eval_injections, fisher_samples
from canna.lisa.mcmc import (
    ANGLES,
    MCMC_ROOT,
    N_PARAMS,
    NullBackend,
    autocorr_time,
    gelman_rubin,
    problem_id,
    to_physical,
    to_sampler,
)
from canna.lisa.physics import (
    chirp_mass_log_density,
    chirp_mass_support,
    sample_chirp_mass,
    sky_log_density,
)
from canna.lisa.problem import LisaGB, f0_window, window_power
from canna.lisa.simulator import sample_observation, source_segments, source_snrs

# the paper's four windows: q1.00, q0.60, q0.30, q0.80
WINDOWS = [9, 5, 2, 7]


def chain_dir(problem: LisaGB, seed: int, root: Path = MCMC_ROOT) -> Path:
    return root / f"joint-{problem_id(problem, seed)}"


def chain_file(directory: Path, j: int, seed: int) -> Path:
    return directory / f"window{j}_seed{seed}.npz"


class Window(NamedTuple):
    """Eval injection j with every source free: its data, the per-source prior boxes [S, 8] and Fisher draws."""

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
    mc_grid: Array
    mc_table: Array
    fisher: np.ndarray


def make_window(problem: LisaGB, injections, j: int) -> Window:
    latents, windows, snrs, _, noise_keys, prior_prec = injections
    f = windows[j]
    truth = np.asarray(latents[j], np.float64)
    n_sources = truth.shape[0]

    centres = np.zeros_like(truth)
    centres[:, ANGLES] = truth[:, ANGLES]
    f_lo, f_hi = (float(v) for v in f0_window(problem, f))
    mc_lo, mc_hi = chirp_mass_support(problem.mass_range)
    lower = np.tile(
        [np.log(f_lo), np.log(mc_lo), 0, 0, -np.pi / 2, 0, -1, 0], (n_sources, 1)
    )
    upper = np.tile(
        [np.log(f_hi), np.log(mc_hi), 0, 0, np.pi / 2, 0, 1, 0], (n_sources, 1)
    )
    # the snr cut in log_likelihood is the amplitude's support; the box only has to hold it near the truth
    snr_lo, snr_hi = problem.snr_range
    lower[:, 2], upper[:, 2] = (
        np.log(truth[:, 2] * snr_lo / snr_hi),
        np.log(truth[:, 2] * snr_hi / snr_lo),
    )
    lower[:, ANGLES], upper[:, ANGLES] = (
        centres[:, ANGLES] - np.pi,
        centres[:, ANGLES] + np.pi,
    )

    grid, table = chirp_mass_log_density(problem.mass_range)
    finite = np.isfinite(table)
    return Window(
        j=j,
        f=f,
        truth=truth,
        snr=np.asarray(source_snrs(problem, latents[j], f)),
        window_snr=float(snrs[j]),
        obs=sample_observation(problem, noise_keys[j], latents[j], f),
        power=window_power(problem, f),
        centres=centres,
        lower=lower,
        upper=upper,
        mc_grid=jnp.asarray(grid[finite]),
        mc_table=jnp.asarray(table[finite]),
        fisher=fisher_samples(problem, latents[j], f, noise_keys[j], prior_prec),
    )


def log_prior(window: Window, theta: Float[Array, "D"]) -> Float[Array, ""]:
    t = rearrange(theta, "(s k) -> s k", k=N_PARAMS)
    inside = jnp.all((t >= window.lower) & (t <= window.upper))
    lp = jnp.interp(
        t[:, 1], window.mc_grid, window.mc_table, left=-jnp.inf, right=-jnp.inf
    ).sum()
    sky = sky_log_density(t[:, 3], t[:, 4]).sum()
    return jnp.where(inside, lp + sky, -jnp.inf)


def log_likelihood(
    problem: LisaGB, window: Window, theta: Float[Array, "D"]
) -> Float[Array, ""]:
    """-sum |o - h|^2 / P over the window, one jaxgb call; -inf unless every source's own snr is in snr_range."""
    p = to_physical(rearrange(theta, "(s k) -> s k", k=N_PARAMS))
    idx, seg = source_segments(problem, p, window.f)
    # the bins outside the window are dropped, as in clean_signal
    inside = (idx >= 0) & (idx < problem.window_bins)
    k = jnp.clip(idx, 0, problem.window_bins - 1)
    seg = jnp.where(inside[..., None], seg, 0.0)
    signal = jnp.zeros((problem.window_bins, 3), seg.dtype).at[k].add(seg)
    log_l = -jnp.sum(jnp.abs(window.obs - signal) ** 2 / window.power)
    rho = jnp.sqrt(2.0 * jnp.sum(jnp.abs(seg) ** 2 / window.power[k], axis=(-2, -1)))
    in_range = jnp.all((rho >= problem.snr_range[0]) & (rho <= problem.snr_range[1]))
    return jnp.where(in_range, log_l, -jnp.inf)


def initial_points(
    problem: LisaGB, window: Window, n: int, key: jax.Array
) -> Float[np.ndarray, "N D"]:
    """Walkers from the window's Fisher draws, chirp masses from the prior, clipped into the boxes and snr_range."""
    n_sources = window.truth.shape[0]
    n_fisher = window.fisher.shape[0]
    pick = np.asarray(jr.choice(key, n_fisher, (n,), replace=n_fisher < n))
    p = window.fisher[pick].copy()
    mc = jax.vmap(partial(sample_chirp_mass, mass_range=problem.mass_range))(
        jr.split(jr.fold_in(key, 1), n * n_sources)
    )
    p[..., 1] = rearrange(np.asarray(mc), "(n s) 1 -> n s", s=n_sources)

    theta = np.stack(
        [to_sampler(p[:, s], window.centres[s]) for s in range(n_sources)], axis=1
    )
    pad = 1e-9 * (window.upper - window.lower)
    theta = np.clip(theta, window.lower + pad, window.upper - pad)
    rho = np.asarray(source_snrs(problem, to_physical(jnp.asarray(theta)), window.f))
    margin = 1 + 1e-6
    lo, hi = problem.snr_range[0] * margin, problem.snr_range[1] / margin
    theta[..., 2] += np.log(np.clip(rho, lo, hi) / rho)
    return rearrange(theta, "n s k -> n (s k)")


def sample_chains(problem: LisaGB, window: Window, p0: np.ndarray, args) -> dict:
    """Run the window's chains from p0; returns the arrays to save."""
    from jexplore.sampling import SamplingMH  # jexplore needs Python 3.13
    from jexplore.steps import Stretch, TSwap
    from jexplore.steps.de import DEStep

    n_temps, dim = len(args.temps), p0.shape[1]
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
        epoch = sampler.run(epoch, niters=args.chunk, nepoch=1, seed=args.seed + done)
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
            f"[{done}/{total}] {time.perf_counter() - start:.0f} s, cold log L median"
            f" {np.median(ll[:, -1]):.2f}, acceptance (swap, move) {np.round(acc[-1], 3)}",
            flush=True,
        )
    chains = np.concatenate(kept, axis=-1)  # (nwalker, dim, n)
    rhat = np.array([gelman_rubin(chains[:, d]) for d in range(dim)])
    tau = np.array([autocorr_time(chains[:, d]) for d in range(dim)]) * args.thin
    samples = rearrange(chains, "w d n -> (w n) d")
    return dict(
        samples=samples,
        physical=np.asarray(
            to_physical(jnp.asarray(rearrange(samples, "n (s k) -> n s k", k=N_PARAMS)))
        ),
        loglik=np.concatenate(kept_ll, axis=-1).reshape(-1),
        rhat=rhat,
        tau=tau,
        ess=args.nwalker * args.keep / tau,
        acceptance=np.array(acc),
        seconds=time.perf_counter() - start,
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="joint_mcmc", description=__doc__)
    parser.add_argument("--window", type=int, required=True, help="eval injection, 0-9")
    parser.add_argument("--seed", type=int, default=0, help="chain seed")
    parser.add_argument("--nwalker", type=int, default=64)
    parser.add_argument(
        "--temps", type=float, nargs="+", default=[1.0, 1.58, 2.51, 3.98, 6.31, 10.0]
    )
    parser.add_argument(
        "--chunk", type=int, default=2000, help="iterations per jexplore epoch"
    )
    parser.add_argument("--burn", type=int, default=100000)
    parser.add_argument("--keep", type=int, default=300000)
    parser.add_argument("--thin", type=int, default=100)
    parser.add_argument("--root", type=Path, default=MCMC_ROOT)
    parser.add_argument("--force", action="store_true", help="rerun a cached chain")
    parser.add_argument(
        "--bench", action="store_true", help="time the likelihood and stop"
    )
    args = parser.parse_args()

    config = run_config.load()
    problem = LisaGB(**config.problem)
    directory = chain_dir(problem, config.seed, args.root)
    path = chain_file(directory, args.window, args.seed)
    if path.exists() and not (args.force or args.bench):
        print(f"{path}: cached", flush=True)
        return 0

    window = make_window(problem, eval_injections(problem, config.seed), args.window)
    n_sources = window.truth.shape[0]
    n_chains = args.nwalker * len(args.temps)
    print(
        f"{config.name} window {args.window}: window SNR {window.window_snr:.1f}, sources SNR"
        f" {np.round(window.snr, 1)}, f {float(window.f):.6e} Hz -> {path}",
        flush=True,
    )
    print(
        f"{n_sources * N_PARAMS} dims, {args.nwalker} walkers x {len(args.temps)} temps ="
        f" {n_chains} chains, backend {jax.default_backend()}",
        flush=True,
    )

    prior = partial(log_prior, window)
    likelihood = partial(log_likelihood, problem, window)
    p0 = initial_points(problem, window, n_chains, jr.key(args.seed + 100))
    assert bool(
        jnp.all(jnp.isfinite(jax.vmap(prior)(jnp.asarray(p0))))
    ), "an initial point is outside the prior"
    truth_theta = rearrange(
        np.stack(
            [to_sampler(window.truth[s], window.centres[s]) for s in range(n_sources)]
        ),
        "s k -> (s k)",
    )
    print(
        f"log L at truth {float(likelihood(jnp.asarray(truth_theta))):.2f},"
        f" log prior at truth {float(prior(jnp.asarray(truth_theta))):.2f}",
        flush=True,
    )

    if args.bench:
        batch = jax.jit(jax.vmap(likelihood))
        batch(jnp.asarray(p0)).block_until_ready()
        start = time.perf_counter()
        for _ in range(20):
            batch(jnp.asarray(p0)).block_until_ready()
        ms = (time.perf_counter() - start) / 20 * 1e3
        print(f"log L over {n_chains} chains: {ms:.2f} ms", flush=True)
        return 0

    out = sample_chains(problem, window, p0, args)
    rhat, ess = out["rhat"], out["ess"]
    print(
        f"\n{out['seconds'] / 60:.1f} min; R-hat max {rhat.max():.4f}; autocorrelation time"
        f" {out['tau'].min():.0f}-{out['tau'].max():.0f} iterations; ESS min {ess.min():.0f}"
    )
    for s in range(n_sources):
        cut = slice(s * N_PARAMS, (s + 1) * N_PARAMS)
        print(
            f"source {s} (SNR {window.snr[s]:.1f}): R-hat {np.round(rhat[cut], 4)},"
            f" ESS {np.round(ess[cut]).astype(int)}"
        )

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
    # written under a hidden name, renamed when complete: no truncated chain looks cached
    pending = path.with_name(f".partial-{path.name}")
    np.savez(
        pending,
        **out,
        truth=window.truth,
        snr=window.snr,
        window_snr=window.window_snr,
        window=args.window,
        f=float(window.f),
        centres=window.centres,
        temps=np.asarray(args.temps),
        nwalker=args.nwalker,
        burn=args.burn,
        keep=args.keep,
        thin=args.thin,
        chain_seed=args.seed,
        config=config.name,
    )
    pending.replace(path)
    print(f"saved {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
