"""The eval job's scores: the per-source f0 scorecard and the likelihood-free validation, into one scorecard.npz."""

import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from . import config as run_config
from . import validate
from .chart import flow_to_physical, physical_to_flow
from .injections import N_POSTERIOR, eval_injections, match_sources
from .network import LisaFlow
from .problem import LisaGB, f0_window
from .sample import ODE_STEPS, sample_posterior
from .samples import pack_validation, unpack
from .simulator import (
    preprocess,
    sample_observation,
    sample_physical,
    sample_window,
    snr,
)
from .train import load_trained, print_peak_memory

N_DRAWS = 256
FOUND_BINS = 1.0
SNR_BANDS = [(0, 15), (15, 40), (40, 100), (100, np.inf)]


def score(problem: LisaGB, draws: np.ndarray, truth: np.ndarray, f) -> dict:
    """Per-source f0 width, offset and calibration of draws [N, S, 8] against the truth [S, 8]."""
    t_obs = problem.t_obs
    source_snr = np.asarray(snr(problem, jnp.asarray(truth)[:, None, :], f))
    f0 = match_sources(problem, draws, truth, f)[..., 0] * t_obs  # in bins
    f0_true = truth[:, 0] * t_obs
    median = np.median(f0, axis=0)
    q = np.percentile(f0, [2.5, 16, 84, 97.5], axis=0)
    width = 1.4826 * np.median(np.abs(f0 - median), axis=0)
    window = np.stack([np.asarray(edge) for edge in f0_window(problem, f)])
    return dict(
        f0=truth[:, 0],
        x_f=np.asarray(physical_to_flow(problem, jnp.asarray(truth), f))[:, 0],
        f0_window=np.broadcast_to(window, (len(truth), 2)),
        snr=source_snr,
        width=width,
        # the monochromatic Fisher bound, ~10% below the full Fisher on the eval sources
        ideal=np.sqrt(3) / (np.pi * source_snr),
        offset=median - f0_true,
        rank=np.mean(f0 < f0_true, axis=0),
        in68=(q[1] <= f0_true) & (f0_true <= q[2]),
        in95=(q[0] <= f0_true) & (f0_true <= q[3]),
    )


def score_injection(
    problem: LisaGB,
    flow: LisaFlow,
    latent,
    f,
    key_n,
    n_draws: int = N_DRAWS,
    ode_steps: int = ODE_STEPS,
) -> dict:
    """Simulate one observation, draw from the flow, and score every source; keys as in eval."""
    y = preprocess(problem, sample_observation(problem, key_n, latent, f), f)
    z0 = jr.normal(jr.fold_in(key_n, 1), (N_POSTERIOR, *latent.shape))[:n_draws]
    post = sample_posterior(problem, flow, z0, y, f, ode_steps)
    draws = np.asarray(
        jax.vmap(partial(flow_to_physical, problem), in_axes=(0, None))(post, f)
    )
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


def main() -> None:
    config = run_config.load_run()
    problem, flow, epoch = load_trained(config)
    ode_steps, n_random = config.ode_steps, config.n_random
    out_dir = run_config.eval_dir(
        config,
        epoch,
        n_random=n_random,
        n_validation=config.n_validation,
        backend=jax.default_backend(),
    )
    print(
        f"run {config.run}, epoch {epoch},"
        f" ode_steps {ode_steps}, bf16 {config.bf16}, n_random {n_random},"
        f" backend {jax.default_backend()} -> {out_dir}",
        flush=True,
    )

    # the eval grid's injections first, then n_random plain prior draws on the seed's stream 4
    latents, windows, _, grid, noise_keys, _ = eval_injections(problem, config.seed)
    key_window, key_pick, key_noise = jr.split(jr.fold_in(jr.key(config.seed), 4), 3)
    random_windows = jax.vmap(partial(sample_window, problem))(jr.split(key_window, n_random))
    random_latents = jax.vmap(partial(sample_physical, problem))(
        jr.split(key_pick, n_random), random_windows
    )
    latents = jnp.concatenate([latents, random_latents])
    windows = jnp.concatenate([windows, random_windows])
    noise_keys = jnp.concatenate([noise_keys, jr.split(key_noise, n_random)])

    rows, injection, is_eval = [], [], []
    start = time.perf_counter()
    for n, key_n in enumerate(noise_keys):
        rows.append(
            score_injection(
                problem,
                flow,
                latents[n],
                windows[n],
                key_n,
                ode_steps=ode_steps,
            )
        )
        injection += [n] * problem.n_sources
        is_eval += [n < len(grid)] * problem.n_sources
        if (n + 1) % 5 == 0:
            print(
                f"[{n + 1}/{len(noise_keys)}] injections scored, {time.perf_counter() - start:.0f} s",
                flush=True,
            )

    rows = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    rows["injection"], rows["is_eval"] = np.array(injection), np.array(is_eval)

    print("\nthe eval injections, per source")
    print(f"{'inj':>4} {'SNR':>7} {'width':>7} {'x ideal':>8} {'offset':>8}")
    for j in np.flatnonzero(rows["is_eval"]):
        print(
            f"{rows['injection'][j]:4d} {rows['snr'][j]:7.1f} {rows['width'][j]:7.3f}"
            f" {rows['width'][j] / rows['ideal'][j]:8.0f} {rows['offset'][j]:+8.3f}"
        )
    summary(
        rows,
        rows["is_eval"],
        f"{len(grid)} eval injections (epoch {epoch}, {ode_steps} ODE steps)",
    )
    summary(
        rows,
        ~rows["is_eval"],
        f"{n_random} random injections (epoch {epoch}, {ode_steps} ODE steps)",
    )
    summary(rows, np.ones_like(rows["is_eval"]), "all")

    # the f0 rows as card_<name>; with n_validation, the validation samples (packed) and their scores beside them
    card = {f"card_{k}": v for k, v in rows.items()}
    validation = {}
    if config.n_validation:
        samples = validate.sample(config, problem, flow)
        validation = pack_validation(samples)
        validation |= validate.summarise(
            unpack(validation),
            config.seed,
            f"{config.n_validation} prior injections (epoch {epoch}, {ode_steps} ODE steps)",
        )
    path = out_dir / "scorecard.npz"
    np.savez(
        path,
        epoch=epoch,
        ode_steps=ode_steps,
        n_random=n_random,
        t_obs=problem.t_obs,
        **card,
        **validation,
    )
    print(f"\nsaved {path}", flush=True)
    print_peak_memory()


if __name__ == "__main__":
    main()
