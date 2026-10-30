"""One long cold chain of the joint reference MCMC per injection and seed, kept from iteration 0 at a fixed thinning."""

import argparse
import sys
from pathlib import Path

import numpy as np
from einops import rearrange
from jaxtyping import Float

REPORT = Path(__file__).parents[1]
from joint_mcmc import TEMPS

from canna.lisa import config as run_config
from canna.lisa.injections import eval_injections, match_sources
from canna.lisa.mcmc import (
    COLUMNS,
    Window,
    cache_dir,
    chain_file,
    fold,
    make_window,
    problem_id,
    sample_chains,
)
from canna.lisa.problem import LisaGB

CONVERGENCE = REPORT.parents[1] / "outputs" / "mcmc" / "convergence"
TRACES = CONVERGENCE / "traces"
BENCH = CONVERGENCE / "bench"


def fold_trace(
    problem: LisaGB, window: Window, physical: Float[np.ndarray, "W T S 8"]
) -> tuple[Float[np.ndarray, "W T S 8"], Float[np.ndarray, "S 8"]]:
    """Physical draws -> COLUMNS coordinates (matched onto the truth, folded), as float32 offsets from the truth."""
    truth, n_sources = window.truth, window.truth.shape[0]
    true = np.stack(
        [fold(truth[s][None], truth[s], problem.t_obs)[0] for s in range(n_sources)]
    )
    out = np.empty(physical.shape, np.float32)
    # one walker at a time: match_sources holds every permutation of every draw
    for w in range(physical.shape[0]):
        matched = match_sources(problem, physical[w], truth, window.f)
        folded = np.stack(
            [fold(matched[:, s], truth[s], problem.t_obs) for s in range(n_sources)],
            axis=1,
        )
        # f0 in bins is ~1e4-1e5 with sub-1e-3-bin widths: only the offset survives float32
        out[w] = folded - true
    return out, true


def main() -> int:
    parser = argparse.ArgumentParser(prog="long_chain", description=__doc__)
    parser.add_argument("--injection", type=int, required=True, help="eval injection")
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
    directory = cache_dir(problem, config.seed, args.root)
    path = chain_file(directory, args.injection, args.seed)
    if path.exists() and not args.force:
        print(f"{path}: saved already", flush=True)
        return 0

    window = make_window(problem, eval_injections(problem, config.seed), args.injection)
    print(
        f"{config.name} injection {args.injection} seed {args.seed}: window SNR {window.window_snr:.1f},"
        f" sources SNR {np.round(window.snr, 1)}; {args.iterations} iterations, thin {args.thin},"
        f" {args.nwalker} walkers x {len(args.temps)} temps -> {path}",
        flush=True,
    )
    # mcmc.sample_chains with no burn-in: the same start and per-epoch seeds, so this is the reference chain unburnt
    out = sample_chains(
        problem,
        window,
        argparse.Namespace(**vars(args), burn=0, keep=args.iterations),
        args.seed,
    )
    per_walker = lambda a: rearrange(a, "(w t) ... -> w t ...", w=args.nwalker)
    trace, true = fold_trace(problem, window, per_walker(out["physical"]))

    directory.mkdir(parents=True, exist_ok=True)
    # written under a hidden name, renamed when complete: no truncated chain looks saved
    pending = path.with_name(f".partial-{path.name}")
    np.savez(
        pending,
        trace=trace,
        loglik=per_walker(out["loglik"]),
        acceptance=out["acceptance"],
        seconds=out["seconds"],
        truth_folded=true,
        truth=window.truth,
        snr=window.snr,
        window_snr=window.window_snr,
        injection=args.injection,
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
