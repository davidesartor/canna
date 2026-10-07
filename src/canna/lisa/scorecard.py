"""Per-source scorecard of a trained flow: f0 width, detection and calibration.

The corner plots show ten injections by eye. This measures what the research log tracks,
per source rather than per injection, and over enough injections to see a bias:

- **width**: the MAD of the flow's f0 for that source, in frequency bins (1/t_obs), and
  that width over the ideal one, sqrt(3) / (pi SNR) bins. The ideal is the monochromatic
  Fisher bound. On the 40 eval sources it sits 0-33% below eval.py's full Fisher
  (median 10%), so "x ideal" reads about 10% higher than "x Fisher".
- **found**: whether the f0 width is under a bin. A source the flow has not found is
  spread over the whole window, several bins wide.
- **offset**: the flow's median minus the truth, in bins.
- **calibration**: whether the truth lies inside the central 68% and 95% of the flow's
  draws for that source, and the fraction of draws below the truth (its rank). A
  calibrated flow covers 68% and 95%, with ranks spread uniformly. A biased one shifts
  the ranks, and an over-confident one covers too little.

The ten eval injections (the SNR deciles eval.py plots) come first, with eval.py's keys
and the first N_DRAWS of its base draws, so their rows compare one to one with the
earlier per-source tables. N_RANDOM plain prior draws follow, for the statistics.
Prints a summary and writes outputs/lisa-<config>/scorecard.npz. A non-default --ode_steps
or --dtype adds _ode<N> or _<dtype> to the name (scorecard_ode64.npz, scorecard_float32.npz),
and a run off the GPU adds the backend (scorecard_float32_cpu.npz), so integrator, precision
and device checks sit beside the default one instead of over it. --n_random scores fewer
random injections, for a slow device; the file records how many.
"""

import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from .problem import LisaGB
from .train import load_trained, parse_args, peak_memory_report
from .eval import match_sources, sample_posterior, N_CANDIDATES, N_POSTERIOR, N_QUANTILES, ODE_STEPS

N_DRAWS = 256
N_RANDOM = 200
FOUND_BINS = 1.0
SNR_BANDS = [(0, 15), (15, 40), (40, 100), (100, np.inf)]


def score(problem: LisaGB, draws: np.ndarray, truth: np.ndarray, f) -> dict:
    """Per-source width, offset and calibration of matched draws [N, S, 8] against truth."""
    t_obs = problem.t_obs
    snr = np.asarray(problem.snr(jnp.asarray(truth)[:, None, :], f))
    f0 = match_sources(problem, draws, truth, f)[..., 0] * t_obs  # in bins
    f0_true = truth[:, 0] * t_obs
    median = np.median(f0, axis=0)
    q = np.percentile(f0, [2.5, 16, 84, 97.5], axis=0)
    width = 1.4826 * np.median(np.abs(f0 - median), axis=0)
    return dict(
        snr=snr,
        width=width,
        ideal=np.sqrt(3) / (np.pi * snr),
        offset=median - f0_true,
        rank=np.mean(f0 < f0_true, axis=0),
        in68=(q[1] <= f0_true) & (f0_true <= q[2]),
        in95=(q[0] <= f0_true) & (f0_true <= q[3]),
    )


def score_injection(
    problem: LisaGB,
    flow,
    latent,
    f,
    key_n,
    time_power: float = 1.0,
    n_draws: int = N_DRAWS,
    ode_steps: int = ODE_STEPS,
) -> dict:
    """Simulate one observation, draw from the flow, and score every source in it.

    Uses eval.py's convention for the keys -- the noise from key_n, the base draws from
    the first n_draws of its N_POSTERIOR splits -- so an eval injection gets the same
    data and the same draws here as there.
    """
    y = problem.preprocess(problem.sample_observation(key_n, latent, f), f)
    u0 = jax.vmap(problem.sample_flow, in_axes=(0, None))(
        jr.split(key_n, N_POSTERIOR)[:n_draws], f
    )
    post = sample_posterior(problem, flow, u0, y, f, ode_steps, time_power)
    draws = np.asarray(jax.vmap(problem.flow_to_physical, in_axes=(0, None))(post, f))
    return score(problem, draws, np.asarray(latent), f)


def summary(rows: dict, which: np.ndarray, title: str) -> None:
    print(f"\n{title}")
    print(
        f"{'SNR':>10} {'n':>4} {'found':>7} {'width':>7} {'x ideal':>8}"
        f" {'offset':>8} {'in68':>6} {'in95':>6} {'rank':>6}"
    )
    for lo, hi in SNR_BANDS:
        m = which & (rows["snr"] >= lo) & (rows["snr"] < hi)
        if not m.any():
            continue
        found = m & (rows["width"] < FOUND_BINS)
        # widths and offsets mean something only for found sources; calibration is for all
        med = lambda v: np.median(v[found]) if found.any() else np.nan
        label = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">={lo:g}"
        print(
            f"{label:>10} {m.sum():4d} {found.sum() / m.sum():7.2f}"
            f" {med(rows['width']):7.3f} {med(rows['width'] / rows['ideal']):8.0f}"
            f" {med(rows['offset']):+8.3f} {rows['in68'][m].mean():6.2f}"
            f" {rows['in95'][m].mean():6.2f} {rows['rank'][m].mean():6.2f}"
        )


if __name__ == "__main__":
    args = parse_args()
    state, epoch, out_dir = load_trained(args)
    problem, flow = state.problem, state.flow
    ode_steps = args.ode_steps or ODE_STEPS
    n_random = N_RANDOM if args.n_random is None else args.n_random
    print(
        f"config {args.config}, epoch {epoch}, time_power {args.time_power},"
        f" ode_steps {ode_steps}, dtype {args.dtype}, n_random {n_random},"
        f" backend {jax.default_backend()}",
        flush=True,
    )

    # the same candidates and keys as eval.py
    key_pick, key_window, key_noise = jr.split(jr.key(args.seed), 3)
    windows = jax.vmap(problem.sample_f)(jr.split(key_window, N_CANDIDATES))
    latents = jax.vmap(problem.sample_physical)(jr.split(key_pick, N_CANDIDATES), windows)
    snrs = np.asarray(jax.lax.map(lambda pc: problem.snr(*pc), (latents, windows)))
    quantiles = np.linspace(1.0 / N_QUANTILES, 1.0, N_QUANTILES)
    chosen = np.argsort(snrs)[np.round(quantiles * (N_CANDIDATES - 1)).astype(int)]
    others = np.setdiff1d(np.arange(N_CANDIDATES), chosen)[:n_random]
    noise_keys = list(jr.split(key_noise, N_QUANTILES)) + [
        jr.fold_in(key_noise, 10_000 + int(i)) for i in others
    ]

    rows, injection, is_eval = [], [], []
    start = time.perf_counter()
    for n, (i, key_n) in enumerate(zip(np.concatenate([chosen, others]), noise_keys)):
        rows.append(
            score_injection(
                problem,
                flow,
                latents[i],
                windows[i],
                key_n,
                args.time_power,
                ode_steps=ode_steps,
            )
        )
        injection += [n] * problem.n_sources
        is_eval += [n < N_QUANTILES] * problem.n_sources
        if (n + 1) % 5 == 0:
            print(
                f"[{n + 1}/{len(noise_keys)}] injections scored,"
                f" {time.perf_counter() - start:.0f} s",
                flush=True,
            )

    rows = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    rows["injection"], rows["is_eval"] = np.array(injection), np.array(is_eval)

    print("\nthe ten eval injections, per source")
    print(f"{'inj':>4} {'SNR':>7} {'width':>7} {'x ideal':>8} {'offset':>8}")
    for j in np.flatnonzero(rows["is_eval"]):
        print(
            f"{rows['injection'][j]:4d} {rows['snr'][j]:7.1f} {rows['width'][j]:7.3f}"
            f" {rows['width'][j] / rows['ideal'][j]:8.0f} {rows['offset'][j]:+8.3f}"
        )
    summary(rows, rows["is_eval"], f"the ten eval injections (epoch {epoch}, {ode_steps} RK4 steps)")
    summary(rows, ~rows["is_eval"], f"{n_random} random injections (epoch {epoch}, {ode_steps} RK4 steps)")
    summary(rows, np.ones_like(rows["is_eval"]), "all")

    name = "scorecard"
    name += "" if ode_steps == ODE_STEPS else f"_ode{ode_steps}"
    name += "" if args.dtype == "bfloat16" else f"_{args.dtype}"
    name += "" if jax.default_backend() == "gpu" else f"_{jax.default_backend()}"
    name += ".npz"
    np.savez(out_dir / name, epoch=epoch, ode_steps=ode_steps, n_random=n_random, **rows)
    print(f"\nsaved {out_dir / name}", flush=True)
    if peak_memory_report():
        print(peak_memory_report(), flush=True)
