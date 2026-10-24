"""One long cold chain of the joint narrowband MCMC per window and seed, kept from iteration 0 at a fixed thinning."""

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from einops import rearrange
from jaxtyping import Float

REPORT = Path(__file__).parents[1]
from joint_mcmc import (
    Window,
    chain_dir,
    chain_file,
    initial_points,
    log_likelihood,
    log_prior,
    make_window,
)

from canna.lisa import config as run_config
from canna.lisa.injections import eval_injections, match_sources
from canna.lisa.mcmc import (
    COLUMNS,
    N_PARAMS,
    NullBackend,
    fold,
    problem_id,
    to_physical,
)
from canna.lisa.problem import LisaGB

CONVERGENCE = REPORT.parents[1] / "outputs" / "mcmc" / "convergence"
TRACES = CONVERGENCE / "traces"
BENCH = CONVERGENCE / "bench"
TEMPS = [1.0, 1.58, 2.51, 3.98, 6.31, 10.0]


def long_chain(problem: LisaGB, window: Window, p0: np.ndarray, args) -> dict:
    """joint_mcmc.sample_chains with no burn-in: cold draws [W, T, D] and log L [W, T], every thin-th iteration."""
    from jexplore.sampling import SamplingMH  # jexplore needs Python 3.13
    from jexplore.steps import Stretch, TSwap
    from jexplore.steps.de import DEStep

    n_temps = len(args.temps)
    sampling = SamplingMH(
        dim=p0.shape[1],
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
    start = time.perf_counter()
    while done < args.iterations:
        epoch = sampler.run(epoch, niters=args.chunk, nepoch=1, seed=args.seed + done)
        # copies: a view would keep every chunk's draws at every temperature alive
        kept.append(np.array(np.asarray(epoch.samples.p)[::n_temps, :, :: args.thin]))
        kept_ll.append(
            np.array(np.asarray(epoch.samples.ll)[::n_temps, 0, :: args.thin])
        )
        acc.append(
            np.asarray(epoch.stats.stats).mean(axis=-1) / np.asarray(epoch.stats.counts)
        )
        done += args.chunk
        elapsed = time.perf_counter() - start
        print(
            f"[{done}/{args.iterations}] {elapsed:.0f} s, {1e3 * elapsed / done:.1f} ms/iteration,"
            f" {elapsed / done * (args.iterations - done) / 60:.0f} min to go, cold log L median"
            f" {np.median(kept_ll[-1][:, -1]):.2f}, acceptance (swap, move) {np.round(acc[-1], 3)}",
            flush=True,
        )
    return dict(
        theta=rearrange(np.concatenate(kept, axis=-1), "w d t -> w t d"),
        loglik=np.concatenate(kept_ll, axis=-1),
        acceptance=np.array(acc),
        seconds=time.perf_counter() - start,
    )


def fold_trace(
    problem: LisaGB, window: Window, theta: Float[np.ndarray, "W T D"]
) -> tuple[Float[np.ndarray, "W T S 8"], Float[np.ndarray, "S 8"]]:
    """Sampler draws -> compare_joint's coordinates (matched onto the truth, folded), as float32 offsets from the truth."""
    truth, n_sources = window.truth, window.truth.shape[0]
    true = np.stack(
        [fold(truth[s][None], truth[s], problem.t_obs)[0] for s in range(n_sources)]
    )
    out = np.empty((*theta.shape[:2], n_sources, N_PARAMS), np.float32)
    # one walker at a time: match_sources holds every permutation of every draw
    for w in range(theta.shape[0]):
        physical = np.asarray(
            to_physical(
                jnp.asarray(rearrange(theta[w], "t (s k) -> t s k", k=N_PARAMS))
            )
        )
        matched = match_sources(problem, physical, truth, window.f)
        folded = np.stack(
            [fold(matched[:, s], truth[s], problem.t_obs) for s in range(n_sources)],
            axis=1,
        )
        # f0 in bins is ~1e4-1e5 with sub-1e-3-bin widths: only the offset survives float32
        out[w] = folded - true
    return out, true


def main() -> int:
    parser = argparse.ArgumentParser(prog="long_chain", description=__doc__)
    parser.add_argument("--window", type=int, required=True, help="eval injection, 0-9")
    parser.add_argument("--seed", type=int, default=0, help="chain seed")
    parser.add_argument("--iterations", type=int, default=400_000)
    parser.add_argument("--thin", type=int, default=20, help="must divide --chunk")
    parser.add_argument("--nwalker", type=int, default=64)
    parser.add_argument("--temps", type=float, nargs="+", default=TEMPS)
    parser.add_argument(
        "--chunk", type=int, default=2000, help="iterations per jexplore epoch"
    )
    parser.add_argument("--root", type=Path, default=TRACES)
    parser.add_argument("--force", action="store_true", help="rerun a saved chain")
    parser.add_argument(
        "--bench",
        action="store_true",
        help="smoke test: 10 000 iterations into outputs/mcmc/convergence/bench/",
    )
    args = parser.parse_args()
    if args.bench:
        args.iterations, args.root = 10_000, BENCH
    assert args.iterations % args.chunk == 0, "--chunk must divide --iterations"
    assert args.chunk % args.thin == 0, "--thin must divide --chunk"

    config = run_config.load()
    problem = LisaGB(**config.problem)
    directory = chain_dir(problem, config.seed, args.root)
    path = chain_file(directory, args.window, args.seed)
    if path.exists() and not args.force:
        print(f"{path}: saved already", flush=True)
        return 0

    window = make_window(problem, eval_injections(problem, config.seed), args.window)
    n_chains = args.nwalker * len(args.temps)
    print(
        f"{config.name} window {args.window} seed {args.seed}: window SNR {window.window_snr:.1f},"
        f" sources SNR {np.round(window.snr, 1)}; {args.iterations} iterations, thin {args.thin},"
        f" {args.nwalker} walkers x {len(args.temps)} temps -> {path}",
        flush=True,
    )
    # the same start and per-epoch seeds as joint_mcmc.py: this is its chain, unburnt
    p0 = initial_points(problem, window, n_chains, jr.key(args.seed + 100))
    assert bool(
        jnp.all(jnp.isfinite(jax.vmap(partial(log_prior, window))(jnp.asarray(p0))))
    ), "an initial point is outside the prior"

    out = long_chain(problem, window, p0, args)
    trace, true = fold_trace(problem, window, out.pop("theta"))

    directory.mkdir(parents=True, exist_ok=True)
    # written under a hidden name, renamed when complete: no truncated chain looks saved
    pending = path.with_name(f".partial-{path.name}")
    np.savez(
        pending,
        trace=trace,
        **out,
        truth_folded=true,
        truth=window.truth,
        snr=window.snr,
        window_snr=window.window_snr,
        window=args.window,
        f=float(window.f),
        columns=np.array(COLUMNS),
        iterations=args.iterations,
        thin=args.thin,
        chunk=args.chunk,
        nwalker=args.nwalker,
        temps=np.asarray(args.temps),
        chain_seed=args.seed,
        config=config.name,
        problem_id=problem_id(problem, config.seed),
    )
    pending.replace(path)
    print(
        f"saved {path}: trace {trace.shape}, {out['seconds'] / 60:.1f} min", flush=True
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
