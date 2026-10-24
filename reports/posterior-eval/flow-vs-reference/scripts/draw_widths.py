"""A run's eval draws per source: their widths, the truth's rank among them, and the narrowband MCMC's widths.

Every source of the ten eval injections, matched to the truth and folded (mcmc.fold), in
PARAMETERS: fold's eight columns and the phases phi± = phi0 ± 2 psi at the chart's phase
reference, the combination the data measure (z+ ~ (1 - sin iota)^2 e^{i phi+}, z- ~ (1 + sin
iota)^2 e^{i phi-}). Widths are 1.4826 MAD, circular for phi±. Where the narrowband MCMC
(one ensemble per source, the others at their truth) is cached for the run's problem and seed,
the same widths of its draws. The record goes to data/widths/<run>.json. From the repo root:
    LISA_RUN=<run> uv run --frozen python reports/posterior-eval/flow-vs-reference/scripts/draw_widths.py
"""

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import jax.numpy as jnp
import numpy as np

from canna.lisa import config as run_config
from canna.lisa.chart import physical_to_flow
from canna.lisa.mcmc import fold, load_chains, newest_eval
from canna.lisa.problem import LisaGB

REPORT = Path(__file__).parents[1]
PARAMETERS = [
    "f0 [bins]",
    "log Mc",
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
# MCMC draws kept per source
N_MCMC = 40000


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


def main() -> None:
    config = run_config.load_run()
    problem = LisaGB(**config.problem)
    eval_dir = newest_eval(config)
    d = np.load(eval_dir / "draws.npz")
    chains = load_chains(problem, config.seed, n=N_MCMC)
    print(f"{config.run} {eval_dir.name}: {len(chains)} MCMC'd sources", flush=True)

    sources = []
    for j, (matched, truth) in enumerate(zip(d["matched"], d["truth"])):
        f = d["f"][j]
        for s in range(truth.shape[0]):
            flow = page_coordinates(problem, matched[:, s], truth[s], f)
            true = page_coordinates(problem, truth[s][None], truth[s], f)[0]
            record = dict(
                injection=j,
                source=s,
                snr=round(float(d["source_snr"][j, s]), 3),
                sin_iota=round(float(np.sin(truth[s, 6])), 4),
                flow=widths(flow),
                rank=truth_ranks(flow, true),
            )
            if (j, s) in chains:
                record["mcmc"] = widths(
                    page_coordinates(problem, chains[j, s], truth[s], f)
                )
            sources.append(record)

    out = REPORT / "data" / "widths" / f"{config.run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            dict(
                run=config.run,
                config=config.name,
                total_steps=config.total_steps,
                eval=eval_dir.name,
                epoch=int(d["epoch"]),
                weights=config.weights,
                ode_steps=config.ode_steps,
                time_power=config.time_power,
                draws=int(d["matched"].shape[1]),
                phase_reference=problem.phase_reference,
                parameters=PARAMETERS,
                sources=sources,
            )
        )
        + "\n"
    )

    # the job log: flow / MCMC width per MCMC'd source, and the truth's coverage
    print(f"{'source':8s} {'SNR':>7s} " + "".join(f"{p:>10s}" for p in PARAMETERS))
    for r in sorted(sources, key=lambda r: -r["snr"]):
        if "mcmc" in r:
            ratios = np.array(r["flow"]) / np.array(r["mcmc"])
            cells = "".join(f"{v:10.3g}" for v in ratios)
            print(f"{r['injection']}_{r['source']:<6d} {r['snr']:7.1f} {cells}")
    ranks = np.array([r["rank"] for r in sources])
    for level in (0.68, 0.95):
        inside = np.mean(np.abs(ranks - 0.5) <= level / 2, 0)
        print(
            f"truth in the central {level:.0%}: " + " ".join(f"{v:.2f}" for v in inside)
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
