"""Per-source posterior widths against the ODE integrator: steps, step schedule and update rule, on the eval's injections.

Run from the repo root on the run's commit (slurm/run-src.sh), as convergence.sbatch does:
    LISA_RUN=<run> uv run --frozen python reports/posterior-eval/sampler-convergence/scripts/convergence_probe.py \
        [--draws 256] [--steps 4,8,16,32,64,128,256] [--schedules uniform,power2,log] [--euler uniform,log]
The record goes to data/<run>.json, rewritten after every cell so a cut-short job still leaves one.
"""

import argparse
import json
import time
import warnings
from datetime import date
from functools import partial
from pathlib import Path

warnings.filterwarnings("ignore")
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Float, Int

from canna.lisa import config as run_config
from canna.lisa.chart import flow_to_physical, physical_to_flow, sample_flow
from canna.lisa.injections import N_POSTERIOR, eval_injections, match_sources
from canna.lisa.network import LOSS_TIME_DECADES, LisaFlow
from canna.lisa.sample import CHUNK, sample_posterior
from canna.lisa.simulator import preprocess, sample_observation, source_snrs
from canna.lisa.train import load_trained

REPORT = Path(__file__).parents[1]
COLUMNS = ["f0 [bins]", "log fdot", "l", "b", "Re z+", "Im z+", "Re z−", "Im z−"]


def network_times(schedule: str, steps: int) -> np.ndarray:
    """The steps + 1 network times from 0 to 1 a schedule visits; `uniform` is the eval's grid."""
    k = np.arange(steps + 1) / steps
    if schedule == "uniform":
        return k
    if schedule.startswith("power"):
        return 1 - (1 - k) ** float(schedule.removeprefix("power"))
    assert schedule == "log", schedule
    # 1 - t log-spaced from 1 down to the loss's time floor, then the last step to t = 1
    remaining = 10.0 ** (-LOSS_TIME_DECADES * np.arange(steps) / max(steps - 1, 1))
    return np.append(1 - remaining, 1.0)


@eqx.filter_jit
def sample_on_grid(
    flow: LisaFlow,
    u: Float[Array, "N S 8"],
    y: Float[Array, "F 6"],
    f: Float[Array, ""],
    times: Float[Array, " G"],
    n_steps: Int[Array, ""],
    rule: str,
) -> Float[Array, "N S 8"]:
    """Draws moved along the velocity (m - u) / (1 - t) over times[0..n_steps]: two calls a step (midpoint, the eval's rule) or one (euler)."""
    cache = flow.encode(y, f)

    @eqx.filter_vmap(in_axes=(None, 0))
    def push(flow: LisaFlow, u: Float[Array, "S 8"]) -> Float[Array, "S 8"]:
        def velocity(u, t):
            m, _ = jnp.split(flow(u, t, y, f, cache), 2, axis=-1)
            return (m - u) / (1 - t)

        def step(i, u):
            t, t_next = times[i], times[i + 1]
            if rule == "euler":
                return u + (t_next - t) * velocity(u, t)
            t_mid = (t + t_next) / 2
            return u + (t_next - t) * velocity(u + (t_mid - t) * velocity(u, t), t_mid)

        return jax.lax.fori_loop(0, n_steps, step, u)

    return push(flow, u)


def crps(a: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Sample CRPS of draws a [N, ...] at the truth [...]: E|X - y| - E|X - X'| / 2."""
    n = a.shape[0]
    a = np.sort(a, axis=0)
    weights = (2 * np.arange(1, n + 1) - n - 1).reshape(-1, *[1] * (a.ndim - 1))
    return np.abs(a - truth).mean(0) - (weights * a).sum(0) / n**2


def summarise(problem, post: np.ndarray, truth: np.ndarray, f) -> dict:
    """Per-source width (1.4826 MAD), offset, truth rank and CRPS in COLUMNS, from flow draws [N, S, 8]."""
    physical = np.asarray(
        jax.vmap(partial(flow_to_physical, problem), in_axes=(0, None))(post, f)
    )
    matched = match_sources(problem, physical, truth, f)

    def columns(p):
        x = np.asarray(physical_to_flow(problem, jnp.asarray(p), f))
        return np.concatenate([p[..., :1] * problem.t_obs, x[..., 1:]], axis=-1)

    a, true = columns(matched), columns(truth)
    median = np.median(a, axis=0)
    return dict(
        width=1.4826 * np.median(np.abs(a - median), axis=0),
        offset=median - true,
        rank=np.mean(a < true, axis=0),
        crps=crps(a, true),
    )


def rounded(v: np.ndarray) -> list:
    return np.vectorize(lambda x: float(f"{x:.4g}"))(np.asarray(v, np.float64)).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--steps", default="4,8,16,32,64,128,256")
    parser.add_argument("--schedules", default="uniform,power2,log")
    parser.add_argument("--euler", default="uniform,log")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    steps = [int(s) for s in args.steps.split(",")]
    cells = [("midpoint", s) for s in args.schedules.split(",") if s] + [
        ("euler", s) for s in args.euler.split(",") if s
    ]
    assert args.draws <= CHUNK, f"at most {CHUNK} draws: sample_on_grid does not chunk"

    config = run_config.load_run()
    problem, flow, epoch = load_trained(config)
    out = Path(args.out or REPORT / "data" / f"{config.run}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    latents, windows, snrs, quantiles, noise_keys, _ = eval_injections(
        problem, config.seed
    )

    # the eval's observations and base draws, keyed as eval.py keys them
    ys, u0s, truths, sources = [], [], [], []
    for j, key_n in enumerate(noise_keys):
        latent, f = latents[j], windows[j]
        ys.append(preprocess(problem, sample_observation(problem, key_n, latent, f), f))
        u0s.append(
            jax.vmap(partial(sample_flow, problem), in_axes=(0, None))(
                jr.split(key_n, N_POSTERIOR)[: args.draws], f
            )
        )
        truths.append(np.asarray(latent))
        source_snr = np.asarray(source_snrs(problem, latent, f))
        sources += [
            dict(
                injection=j,
                slot=s,
                snr=round(float(source_snr[s]), 3),
                window_snr=round(float(snrs[j]), 3),
                f0_bins=round(float(truths[-1][s, 0] * problem.t_obs), 3),
            )
            for s in range(problem.n_sources)
        ]

    record = dict(
        run=config.run,
        config=config.name,
        commit=config.commit,
        epoch=int(epoch),
        total_steps=config.total_steps,
        weights=config.weights,
        bf16=config.bf16,
        device=jax.devices()[0].device_kind,
        eval_ode_steps=config.ode_steps,
        log_floor=10.0**-LOSS_TIME_DECADES,
        draws=args.draws,
        steps=steps,
        date=str(date.today()),
        columns=COLUMNS,
        sources=sources,
        cells=[],
    )
    save = lambda: out.write_text(json.dumps(record) + "\n")

    def run_cell(sampler) -> tuple[dict, float, list]:
        rows, posts = [], []
        start = time.perf_counter()
        for j in range(len(ys)):
            post = np.asarray(jax.block_until_ready(sampler(j)))
            posts.append(post)
            rows.append(summarise(problem, post, truths[j], windows[j]))
        seconds = time.perf_counter() - start
        merged = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
        return merged, seconds, posts

    # the eval's own sampler, the yardstick every grid cell is checked against
    eval_sampler = lambda j: sample_posterior(
        flow, u0s[j], ys[j], windows[j], config.ode_steps
    )
    eval_sampler(0)
    merged, seconds, eval_posts = run_cell(eval_sampler)
    record["eval_cell"] = dict(
        rule="eval",
        schedule="uniform",
        steps=config.ode_steps,
        seconds=round(seconds, 2),
        **{k: rounded(v) for k, v in merged.items()},
    )
    print(f"eval sampler, {config.ode_steps} steps: {seconds:.1f} s", flush=True)

    n_max = max(steps)
    for rule, schedule in cells:
        for n in steps:
            times = np.ones(n_max + 1)
            times[: n + 1] = network_times(schedule, n)
            sampler = lambda j: sample_on_grid(
                flow,
                u0s[j],
                ys[j],
                windows[j],
                jnp.asarray(times),
                jnp.asarray(n),
                rule,
            )
            sampler(0)
            merged, seconds, posts = run_cell(sampler)
            calls = (2 if rule == "midpoint" else 1) * n
            cell = dict(
                rule=rule,
                schedule=schedule,
                steps=n,
                calls=calls,
                seconds=round(seconds, 2),
                **{k: rounded(v) for k, v in merged.items()},
            )
            # the grid sampler must reproduce the eval's on the eval's own grid
            if (rule, schedule, n) == ("midpoint", "uniform", config.ode_steps):
                gap = max(float(np.abs(p - q).max()) for p, q in zip(posts, eval_posts))
                record["eval_match"] = gap
                print(f"grid sampler vs sample_posterior: max |Δx| {gap:.3g}")
            record["cells"].append(cell)
            save()
            f0_width = np.median(merged["width"][:, 0])
            print(
                f"{rule:8s} {schedule:8s} {n:4d} steps ({calls:4d} calls): {seconds:7.1f} s,"
                f" median f0 width {f0_width:.4g} bins, median f0 CRPS {np.median(merged['crps'][:, 0]):.4g}",
                flush=True,
            )
    save()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
