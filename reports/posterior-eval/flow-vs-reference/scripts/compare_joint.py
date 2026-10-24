"""Flow against the joint MCMC on narrowband eval windows: per source, the widths, coverage and draws the page and the paper's MCMC appendix read."""

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPORT = Path(__file__).parents[1]
# joint_mcmc.py (the windows and the chain cache) lives with the mcmc report
sys.path.insert(0, str(REPORT.parents[1] / "mcmc" / "scripts"))
import jax.numpy as jnp
import numpy as np
from scipy.stats import wasserstein_distance

from joint_mcmc import WINDOWS, chain_dir, make_window

from canna.lisa import config as run_config
from canna.lisa.chart import physical_to_flow
from canna.lisa.injections import eval_injections, match_sources
from canna.lisa.mcmc import COLUMNS, MCMC_ROOT, fold, newest_eval
from canna.lisa.problem import LisaGB
from canna.lisa.simulator import log_likelihood

# draws per set and source kept for the page and the paper
N_PLOT = 2000
ROWS = [
    "mcmc width",
    "flow/mcmc width",
    "flow W1/sd",
    "fisher/mcmc width",
    "seed1/seed0 width",
    "seed W1/sd",
]


def fold_sources(p: np.ndarray, truth: np.ndarray, t_obs: float) -> np.ndarray:
    """Draws [N, S, 8] -> COLUMNS coordinates [N, S, 8], each source folded onto its truth."""
    return np.stack(
        [fold(p[:, s], truth[s], t_obs) for s in range(truth.shape[0])], axis=1
    )


def width(a: np.ndarray) -> np.ndarray:
    return 1.4826 * np.median(np.abs(a - np.median(a, axis=0)), axis=0)


def compare(a: np.ndarray, ref: np.ndarray):
    """a, ref [N, 8] of one source -> width ratio, W1 / width(ref), ref median in a's 68 / 95."""
    ratio = width(a) / width(ref)
    w1 = np.array([wasserstein_distance(a[:, c], ref[:, c]) for c in range(a.shape[1])])
    w1 /= width(ref)
    q = np.percentile(a, [2.5, 16, 84, 97.5], axis=0)
    med = np.median(ref, axis=0)
    return ratio, w1, (q[1] <= med) & (med <= q[2]), (q[0] <= med) & (med <= q[3])


def chart_phases(problem: LisaGB, p: np.ndarray, f) -> np.ndarray:
    """Physical draws [..., 8] -> (phi+, phi-) [..., 2], the phases of z± at the chart's phase reference."""
    x = np.asarray(physical_to_flow(problem, jnp.asarray(p), f))
    return np.stack([np.arctan2(x[..., 5], x[..., 4]), np.arctan2(x[..., 7], x[..., 6])], -1)


def check_symmetry(problem: LisaGB, window) -> None:
    """The folding is only right if the likelihood is invariant under it: check at the truth."""
    p = jnp.asarray(window.truth)
    base = float(log_likelihood(problem, p, window.obs, window.f))
    for dpsi, dphi in [(np.pi, 0.0), (np.pi / 2, np.pi)]:
        q = p.at[:, 5].add(dpsi).at[:, 7].add(dphi)
        moved = float(log_likelihood(problem, q, window.obs, window.f))
        assert abs(moved - base) < 1e-6 * abs(base), (dpsi, dphi, moved, base)


def main() -> int:
    parser = argparse.ArgumentParser(prog="compare_joint", description=__doc__)
    parser.add_argument("--windows", type=int, nargs="+", default=WINDOWS)
    parser.add_argument(
        "--root", type=Path, default=MCMC_ROOT, help="the chains' cache root"
    )
    parser.add_argument(
        "--n_mcmc", type=int, default=40000, help="MCMC draws compared, per seed"
    )
    parser.add_argument(
        "--flow",
        type=Path,
        help="an eval dir's draws.npz to compare (default: LISA_RUN's newest eval)",
    )
    parser.add_argument(
        "--tag", help="suffix of the outputs (default: _<run> with LISA_RUN, else none)"
    )
    args = parser.parse_args()

    config = run_config.load_run() if os.environ.get("LISA_RUN") else run_config.load()
    if args.tag is None:
        args.tag = f"_{config.run}" if config.run else ""
    problem = LisaGB(**config.problem)
    t_obs = problem.t_obs
    injections = eval_injections(problem, config.seed)
    flow_path = args.flow or newest_eval(config) / "draws.npz"
    saved = np.load(flow_path)
    assert np.allclose(
        saved["truth"], np.asarray(injections[0])
    ), f"{flow_path} holds other injections than {config.name}'s eval"
    flow_all = saved["draws"]
    chains = chain_dir(problem, config.seed, args.root)
    out_dir = chain_dir(problem, config.seed, REPORT / "data")
    out_dir.mkdir(parents=True, exist_ok=True)

    for j in args.windows:
        window = make_window(problem, injections, j)
        check_symmetry(problem, window)
        truth, f = window.truth, window.f
        runs = sorted(chains.glob(f"window{j}_seed*.npz"))
        assert (
            runs
        ), f"no joint MCMC for window {j} in {chains}: run joint_mcmc.py first"

        lines = []
        say = lambda text="": (lines.append(text), print(text, flush=True))
        rng = np.random.default_rng(0)
        mcmc, matched = [], []
        for r in runs:
            z = np.load(r)
            say(
                f"{r.name}: {z['seconds'] / 60:.1f} min, R-hat max {z['rhat'].max():.4f},"
                f" tau max {z['tau'].max():.0f} iterations, ESS min {z['ess'].min():.0f}"
            )
            physical = z["physical"]
            pick = rng.choice(
                len(physical), min(args.n_mcmc, len(physical)), replace=False
            )
            matched.append(match_sources(problem, physical[pick], truth, f))
            mcmc.append(fold_sources(matched[-1], truth, t_obs))
        flow_matched = match_sources(problem, flow_all[j], truth, f)
        fisher_matched = match_sources(problem, window.fisher, truth, f)
        flow = fold_sources(flow_matched, truth, t_obs)
        fisher = fold_sources(fisher_matched, truth, t_obs)
        ref = np.concatenate(mcmc)
        # phi± offsets from the truth's, wrapped: the phases the data pin, invariant under the folds
        true_phases = chart_phases(problem, truth, f)
        phases = {
            name: (chart_phases(problem, p, f) - true_phases + np.pi) % (2 * np.pi) - np.pi
            for name, p in (
                ("mcmc", np.concatenate(matched)),
                ("flow", flow_matched),
                ("fisher", fisher_matched),
            )
        }
        true = fold_sources(truth[None], truth, t_obs)[0]

        say(
            f"\nwindow {j} ({config.name}, {flow_path}): window SNR {window.window_snr:.1f};"
            f" {len(flow)} flow draws, {len(ref)} MCMC draws over {len(runs)} runs"
        )
        out = {}
        for s in np.argsort(-window.snr):
            r_flow = compare(flow[:, s], ref[:, s])
            r_fish = compare(fisher[:, s], ref[:, s])
            r_seed = (
                compare(mcmc[1][:, s], mcmc[0][:, s]) if len(mcmc) > 1 else None
            )
            say(f"\n source {s}, SNR {window.snr[s]:.1f}")
            say(f"  {'':22s}" + "".join(f"{c:>11s}" for c in COLUMNS))
            rows = [
                ("MCMC width", width(ref[:, s])),
                ("flow / MCMC width", r_flow[0]),
                ("flow W1 / MCMC sd", r_flow[1]),
                ("Fisher / MCMC width", r_fish[0]),
            ]
            if r_seed is not None:
                rows += [
                    ("seed 1 / seed 0 width", r_seed[0]),
                    ("seed W1 / sd", r_seed[1]),
                ]
            for name, v in rows:
                say(f"  {name:22s}" + "".join(f"{x:11.4g}" for x in v))
            for level, inside in (("68", r_flow[2]), ("95", r_flow[3])):
                say(
                    f"  {'MCMC median in flow ' + level:22s}"
                    + "".join(f"{str(bool(x)):>11s}" for x in inside)
                )
            seed_rows = r_seed[:2] if r_seed else [np.nan * r_flow[0]] * 2
            out[f"s{s}"] = np.stack(
                [width(ref[:, s]), *r_flow[:2], r_fish[0], *seed_rows]
            )
            out[f"s{s}_in68"], out[f"s{s}_in95"] = r_flow[2], r_flow[3]
            far = np.mean(np.abs(flow[:, s, 0] - np.median(ref[:, s, 0])) > 1.0)
            out[f"s{s}_far"] = far
            say(f"  flow draws > 1 bin from the MCMC median in f0: {far:.3f}")
            for name, draws in (("mcmc", ref[:, s]), ("flow", flow[:, s]), ("fisher", fisher[:, s])):
                pick = rng.choice(len(draws), min(N_PLOT, len(draws)), replace=False)
                # offsets from the truth: f0 in bins is ~1e4 while its widths are below 1e-3
                out[f"s{s}_{name}_offsets"] = (draws[pick] - true[s]).astype(np.float32)
                out[f"s{s}_{name}_phases"] = phases[name][pick, s].astype(np.float32)

        loudest = int(np.argmax(window.snr))
        draws_loudest = np.concatenate(matched)[:, loudest]
        draws_loudest = draws_loudest[
            rng.choice(len(draws_loudest), 4000, replace=False)
        ]
        summary = out_dir / f"compare_window{j}{args.tag}.npz"
        np.savez(
            summary,
            snr=window.snr,
            columns=COLUMNS,
            truth=truth,
            truth_folded=true,
            window_snr=float(window.window_snr),
            window=float(f),
            loudest=loudest,
            mcmc_loudest=draws_loudest,
            rows=ROWS,
            run=config.run,
            flow_draws=str(flow_path),
            epoch=int(saved["epoch"]),
            **out,
        )
        summary.with_suffix(".txt").write_text("\n".join(lines) + "\n")
        print(f"\nsaved {summary} (and .txt)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
