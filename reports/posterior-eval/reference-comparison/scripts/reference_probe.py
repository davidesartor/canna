"""Flow draws against the joint MCMC reference (every source's marginal, and whole windows), one record per setting in data/<setting>.json.

Settings are `name=eval_dir` pairs (an eval dir holds draws.npz); the base config is --run's config or --config's.
Run from the repo root as reference.sbatch does:
    uv run --frozen python reports/posterior-eval/reference-comparison/scripts/reference_probe.py \
        --run <run> <run>-32=outputs/runs/<run>/eval/<epoch>-average-32
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPORT = Path(__file__).parents[1]
import jax.numpy as jnp
import numpy as np
import yaml
from scipy.stats import wasserstein_distance

from record import window_record

from canna.lisa import config as run_config
from canna.lisa.chart import physical_to_flow
from canna.lisa.injections import (
    N_PER_WINDOW,
    N_WINDOWS,
    eval_injections,
    match_sources,
)
from canna.lisa.mcmc import cache_dir, fold, load_chains, make_window
from canna.lisa.problem import LisaGB
from canna.lisa.simulator import log_likelihood, source_snrs

PARAMETERS = [
    "f0 [bins]",
    "log fdot",
    "log A",
    "RA",
    "Dec",
    "ψ",
    "sin ι",
    "φ0",
    "φ+",
    "φ−",
]
PHASES = slice(8, 10)
N_MCMC = 40000
N_PLOT = 2000
SMOKE_STEPS = 20_000
# the joint card's windows: a mid-SNR injection in every other window of the band
WINDOWS = [w * N_PER_WINDOW + N_PER_WINDOW // 2 for w in range(0, N_WINDOWS, 2)]
ROWS = [
    "mcmc width",
    "flow/mcmc width",
    "flow W1/sd",
    "fisher/mcmc width",
    "seed1/seed0 width",
    "seed W1/sd",
]


def page_coordinates(
    problem: LisaGB, p: np.ndarray, truth: np.ndarray, f
) -> np.ndarray:
    """Physical draws [N, 8] of one source -> PARAMETERS [N, 10], folded onto its truth."""
    x = np.asarray(physical_to_flow(problem, jnp.asarray(p), f))
    phases = np.stack([np.arctan2(x[:, 5], x[:, 4]), np.arctan2(x[:, 7], x[:, 6])], 1)
    return np.concatenate([fold(p, truth, problem.t_obs), phases], 1)


def widths(a: np.ndarray) -> list[float]:
    """PARAMETERS [N, 10] -> robust widths, circular (sqrt(-2 log R)) for phi±."""
    linear = 1.4826 * np.median(np.abs(a - np.median(a, 0)), 0)
    resultant = np.abs(np.mean(np.exp(1j * a[:, PHASES]), 0))
    linear[PHASES] = np.sqrt(-2 * np.log(np.maximum(resultant, 1e-12)))
    return [round(float(w), 6) for w in linear]


def truth_ranks(a: np.ndarray, true: np.ndarray) -> list[float]:
    """Fraction of draws below the truth per parameter; phi± are cut opposite the draws' circular mean."""
    a, true = a.copy(), true.copy()
    centre = np.angle(np.mean(np.exp(1j * a[:, PHASES]), 0))
    a[:, PHASES] = (a[:, PHASES] - centre + np.pi) % (2 * np.pi)
    true[PHASES] = (true[PHASES] - centre + np.pi) % (2 * np.pi)
    return [round(float(r), 4) for r in np.mean(a < true, 0)]


def width(a: np.ndarray) -> np.ndarray:
    return 1.4826 * np.median(np.abs(a - np.median(a, axis=0)), axis=0)


def compare(a: np.ndarray, ref: np.ndarray):
    """a, ref [N, 8] of one source -> width ratio, W1 / width(ref), ref median in a's 68 / 95."""
    ratio = width(a) / width(ref)
    w1 = np.array(
        [wasserstein_distance(a[:, c], ref[:, c]) for c in range(a.shape[1])]
    ) / width(ref)
    q = np.percentile(a, [2.5, 16, 84, 97.5], axis=0)
    med = np.median(ref, axis=0)
    return ratio, w1, (q[1] <= med) & (med <= q[2]), (q[0] <= med) & (med <= q[3])


def fold_sources(p: np.ndarray, truth: np.ndarray, t_obs: float) -> np.ndarray:
    """Draws [N, S, 8] -> COLUMNS coordinates [N, S, 8], each source folded onto its truth."""
    return np.stack(
        [fold(p[:, s], truth[s], t_obs) for s in range(truth.shape[0])], axis=1
    )


def chart_phases(problem: LisaGB, p: np.ndarray, f) -> np.ndarray:
    """Physical draws [..., 8] -> (phi+, phi-) [..., 2], the phases of z± at the chart's phase reference."""
    x = np.asarray(physical_to_flow(problem, jnp.asarray(p), f))
    return np.stack(
        [np.arctan2(x[..., 5], x[..., 4]), np.arctan2(x[..., 7], x[..., 6])], -1
    )


def check_symmetry(problem: LisaGB, window) -> None:
    """The folding is only right if the likelihood is invariant under it: check at the truth."""
    p = jnp.asarray(window.truth)
    base = float(log_likelihood(problem, p, window.obs, window.f))
    for dpsi, dphi in [(np.pi, 0.0), (np.pi / 2, np.pi)]:
        moved = float(
            log_likelihood(
                problem, p.at[:, 5].add(dpsi).at[:, 7].add(dphi), window.obs, window.f
            )
        )
        assert abs(moved - base) < 1e-6 * abs(base), (dpsi, dphi, moved, base)


def per_source_records(problem: LisaGB, saved: dict, chains: dict) -> list[dict]:
    """Every eval source: SNR, sin iota, flow widths, truth ranks and, where chains exist, the MCMC marginal's widths."""
    sources = []
    for j, (matched, truth) in enumerate(zip(saved["matched"], saved["truth"])):
        f = saved["f"][j]
        mcmc = match_sources(problem, chains[j], truth, f) if j in chains else None
        for s in range(truth.shape[0]):
            flow = page_coordinates(problem, matched[:, s], truth[s], f)
            true = page_coordinates(problem, truth[s][None], truth[s], f)[0]
            record = dict(
                injection=j,
                source=s,
                snr=round(float(saved["source_snr"][j, s]), 3),
                sin_iota=round(float(np.sin(truth[s, 6])), 4),
                flow=widths(flow),
                rank=truth_ranks(flow, true),
            )
            if mcmc is not None:
                record["mcmc"] = widths(
                    page_coordinates(problem, mcmc[:, s], truth[s], f)
                )
            sources.append(record)
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(prog="reference_probe", description=__doc__)
    parser.add_argument(
        "settings", nargs="+", help="name=eval_dir, an eval dir holding draws.npz"
    )
    parser.add_argument(
        "--run", help="the run whose config and chains the draws belong to"
    )
    parser.add_argument(
        "--config",
        default="narrow-B",
        help="base config when there is no --run (mocks)",
    )
    parser.add_argument("--windows", type=int, nargs="+", default=WINDOWS)
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="name=text, a setting's button label",
    )
    args = parser.parse_args()

    if args.run:
        os.environ["LISA_RUN"] = args.run
    config = run_config.load_run() if args.run else run_config.load(args.config)
    problem = LisaGB(**config.problem)
    injections = eval_injections(problem, config.seed)
    truth_all, windows_f = np.asarray(injections[0]), np.asarray(injections[1])
    source_snr = np.stack(
        [
            np.asarray(source_snrs(problem, truth_all[j], windows_f[j]))
            for j in range(len(truth_all))
        ]
    )
    labels = dict(a.split("=", 1) for a in args.label)

    # every setting's draws must be on the eval injections the chains were run for
    records = {}
    for pair in args.settings:
        name, eval_dir = pair.split("=", 1)
        eval_dir = Path(eval_dir)
        saved = dict(np.load(eval_dir / "draws.npz"))
        meta = yaml.safe_load((eval_dir / "eval.yaml").read_text())
        mock = meta["weights"] == "mock"
        error = float(np.abs(saved["truth"] - truth_all).max())
        assert error == 0.0 and np.allclose(
            saved["f"], windows_f
        ), f"{eval_dir}: other injections than {config.name}'s eval (truth err {error})"
        assert np.allclose(
            saved["source_snr"], source_snr, rtol=1e-6
        ), f"{eval_dir}: other source SNRs"
        print(
            f"{name}: {eval_dir}, truth, f and source SNR identical to the eval injections",
            flush=True,
        )
        records[name] = dict(
            setting=name,
            label=labels.get(name, name),
            kind="mock" if mock else "run",
            run=config.run or "",
            config=config.name,
            band=config.name.split("-")[0],
            total_steps=None if mock else int(config.total_steps),
            smoke=False if mock else int(config.total_steps) < SMOKE_STEPS,
            ode_steps=None if mock else int(meta["ode_steps"]),
            eval=eval_dir.name,
            epoch=int(saved["epoch"]),
            weights=meta["weights"],
            draws=int(saved["matched"].shape[1]),
            parameters=PARAMETERS,
            saved=saved,
        )

    chains = load_chains(problem, config.seed, n=N_MCMC)
    print(f"{len(chains)} MCMC'd injections", flush=True)
    for r in records.values():
        r["widths"] = dict(
            eval=r["eval"],
            epoch=r["epoch"],
            phase_reference=problem.phase_reference,
            sources=per_source_records(problem, r["saved"], chains),
        )
        r["joint"] = []

    # joint windows: the chains and Fisher load once per window, then every setting is compared
    joint_chains = cache_dir(problem, config.seed)
    for j in args.windows:
        window = make_window(problem, injections, j)
        check_symmetry(problem, window)
        truth, f, t_obs = window.truth, window.f, problem.t_obs
        runs = sorted(joint_chains.glob(f"inj{j}_seed*.npz"))
        assert runs, f"no joint MCMC for window {j} in {joint_chains}"
        rng = np.random.default_rng(0)
        mcmc, matched = [], []
        for path in runs:
            physical = np.load(path)["physical"]
            pick = rng.choice(len(physical), min(N_MCMC, len(physical)), replace=False)
            matched.append(match_sources(problem, physical[pick], truth, f))
            mcmc.append(fold_sources(matched[-1], truth, t_obs))
        ref = np.concatenate(mcmc)
        fisher_matched = match_sources(problem, window.fisher, truth, f)
        fisher = fold_sources(fisher_matched, truth, t_obs)
        true_phases = chart_phases(problem, truth, f)
        wrap = (
            lambda p: (chart_phases(problem, p, f) - true_phases + np.pi) % (2 * np.pi)
            - np.pi
        )
        true = fold_sources(truth[None], truth, t_obs)[0]
        phases = dict(mcmc=wrap(np.concatenate(matched)), fisher=wrap(fisher_matched))
        seed_compare = (
            {s: compare(mcmc[1][:, s], mcmc[0][:, s]) for s in range(truth.shape[0])}
            if len(mcmc) > 1
            else {}
        )

        for r in records.values():
            flow_matched = match_sources(problem, r["saved"]["draws"][j], truth, f)
            flow = fold_sources(flow_matched, truth, t_obs)
            phases["flow"] = wrap(flow_matched)
            out = {}
            for s in range(truth.shape[0]):
                r_flow, r_fish = compare(flow[:, s], ref[:, s]), compare(
                    fisher[:, s], ref[:, s]
                )
                r_seed = seed_compare.get(s)
                seed_rows = r_seed[:2] if r_seed else [np.nan * r_flow[0]] * 2
                out[f"s{s}"] = np.stack(
                    [width(ref[:, s]), *r_flow[:2], r_fish[0], *seed_rows]
                )
                out[f"s{s}_in68"], out[f"s{s}_in95"] = r_flow[2], r_flow[3]
                out[f"s{s}_far"] = np.mean(
                    np.abs(flow[:, s, 0] - np.median(ref[:, s, 0])) > 1.0
                )
                for kind, draws in (
                    ("mcmc", ref[:, s]),
                    ("flow", flow[:, s]),
                    ("fisher", fisher[:, s]),
                ):
                    pick = np.random.default_rng(s).choice(
                        len(draws), min(N_PLOT, len(draws)), replace=False
                    )
                    # offsets from the truth: f0 in bins is ~1e4 while its widths are below 1e-3
                    out[f"s{s}_{kind}_offsets"] = (draws[pick] - true[s]).astype(
                        np.float32
                    )
                    out[f"s{s}_{kind}_phases"] = phases[kind][pick, s].astype(
                        np.float32
                    )
            out.update(snr=window.snr, rows=ROWS, window_snr=float(window.window_snr))
            r["joint"].append(window_record(out, j, r["epoch"]))
        print(f"window {j} done (window SNR {window.window_snr:.1f})", flush=True)

    for name, r in records.items():
        del r["saved"]
        (REPORT / "data" / f"{name}.json").write_text(
            json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        per = [s for s in r["widths"]["sources"] if "mcmc" in s]
        ratios = np.array([np.array(s["flow"]) / np.array(s["mcmc"]) for s in per])
        joint = np.array(
            [
                w_s["rows"]["flow/mcmc width"]
                for w in r["joint"]
                for w_s in w["sources"]
            ],
            dtype=float,
        )
        print(
            f"{name}: median flow/MCMC width, marginals ({len(per)}): "
            + " ".join(f"{v:.3g}" for v in np.median(ratios, 0))
        )
        print(
            f"{name}: median flow/MCMC width, windows ({len(joint)}):   "
            + " ".join(f"{v:.3g}" for v in np.nanmedian(joint, 0))
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
