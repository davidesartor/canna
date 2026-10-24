"""Likelihood-free validation on prior injections: SBC ranks, joint pre-ranks, TARP, ECDF γ, energy score and CRPS."""

import time
from functools import cache, partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from einops import rearrange
from jaxtyping import Array, Complex, Float, Key

from . import config as run_config
from .chart import flow_to_physical, physical_to_flow, sample_flow
from .network import LisaFlow
from .problem import LisaGB, sample_f, window_power
from .sample import ODE_STEPS, sample_posterior
from .samples import load as load_samples
from .samples import pack_validation
from .simulator import (
    clean_signal,
    log_likelihood,
    preprocess,
    sample_observation,
    sample_physical,
)
from .train import print_peak_memory

N_DRAWS = 127
BATCH = 10
SNR_BANDS = [(0, 15), (15, 40), (40, 100), (100, np.inf)]
COORDS = ["f0", "mc", "lon", "lat", "re_zp", "im_zp", "re_zm", "im_zm"]
GAMMA_SIMS = 1000
ALPHA = 0.05


def quantities(
    problem: LisaGB,
    theta: Float[Array, "M S 8"],
    o: Complex[Array, "F 3"],
    f: Float[Array, ""],
) -> dict:
    """Every test quantity of each of M parameter sets, sources in their own canonical order, loudest first."""
    power = window_power(problem, f)
    h = clean_signal(problem, theta[..., :, None, :], f)
    source_snr = jnp.sqrt(jnp.sum(2.0 * jnp.abs(h) ** 2 / power, axis=(-2, -1)))
    # the same truth-independent label for the truth and every draw: sort by own snr
    order = jnp.argsort(-source_snr, axis=-1)
    take = lambda v: jnp.take_along_axis(v, order, axis=-1)
    theta = jnp.take_along_axis(theta, order[..., None], axis=-2)
    # each source's template projected on the data, <o|h_s> / |h_s|
    overlap = (
        jnp.sum(2.0 * jnp.real(o * jnp.conj(h)) / power, axis=(-2, -1)) / source_snr
    )
    return dict(
        x=physical_to_flow(problem, theta, f),
        f0_bins=theta[..., 0] * problem.t_obs,
        source_snr=take(source_snr),
        overlap=take(overlap),
        log_likelihood=log_likelihood(problem, theta, o, f),
        snr=jnp.sqrt(
            jnp.sum(
                2.0 * jnp.abs(clean_signal(problem, theta, f)) ** 2 / power,
                axis=(-2, -1),
            )
        ),
    )


@eqx.filter_jit
def validate_injection(
    problem: LisaGB,
    flow: LisaFlow,
    key: Key[Array, ""],
    time_power: float = 1.0,
    n_draws: int = N_DRAWS,
    ode_steps: int = ODE_STEPS,
) -> dict:
    """One prior injection and n_draws flow draws on it; row 0 of every quantity is the truth."""
    key_f, key_p, key_n, key_u, key_tarp = jr.split(key, 5)
    f = sample_f(problem, key_f)
    truth = sample_physical(problem, key_p, f)
    o = sample_observation(problem, key_n, truth, f)
    y = preprocess(problem, o, f)
    u0 = jax.vmap(partial(sample_flow, problem), in_axes=(0, None))(
        jr.split(key_u, n_draws), f
    )
    post = sample_posterior(flow, u0, y, f, ode_steps, time_power)
    draws = jax.vmap(partial(flow_to_physical, problem), in_axes=(0, None))(post, f)
    # TARP's reference point (Lemos+23): a prior draw independent of the truth, its own stream
    tarp_ref = sample_flow(problem, key_tarp, f)
    return dict(
        f=f,
        tarp_ref=tarp_ref,
        **quantities(problem, jnp.concat([truth[None], draws]), o, f),
    )


def ranks(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The truth's rank among the draws along axis 1 (truth first), 0..M, ties broken at random."""
    truth, draws = values[:, :1], values[:, 1:]
    below, ties = (draws < truth).sum(1), (draws == truth).sum(1)
    return below + rng.integers(0, ties + 1)


def pre_ranks(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average-rank and band-depth pre-ranks (Thorarinsdottir+16) of each of the m points x[:, i] over the last axis."""
    m = x.shape[1]
    r = (x[:, None, :] <= x[:, :, None]).sum(2)
    return r.mean(-1), ((m - r) * (r - 1)).mean(-1) + (m - 1)


def energy_score(x: np.ndarray) -> np.ndarray:
    """The fair energy score of the draws x[:, 1:] against the truth x[:, 0], over the last axis."""
    truth, draws = x[:, :1], x[:, 1:]
    m = draws.shape[1]
    to_truth = np.linalg.norm(draws - truth, axis=-1).mean(1)
    spread = np.linalg.norm(draws[:, :, None] - draws[:, None], axis=-1).sum((1, 2))
    return to_truth - spread / (2 * m * (m - 1))


def analyse(samples: dict, seed: int = 0) -> dict:
    """Ranks, pre-ranks, y-statistic strata and scores of every injection from the saved samples."""
    rng = np.random.default_rng(seed)
    x = rearrange(samples["x"], "n m s d -> n s m d")
    n, n_sources, m = x.shape[:3]
    x_f, x_mc = x[..., 0], x[..., 1]
    per_slot = {name: x[..., c] for c, name in enumerate(COORDS)}
    per_slot |= {
        "f0_plus_mc": x_f + x_mc,
        "f0_minus_mc": x_f - x_mc,
        "abs_zp": np.hypot(x[..., 4], x[..., 5]),
        "abs_zm": np.hypot(x[..., 6], x[..., 7]),
        "overlap": rearrange(samples["overlap"], "n m s -> n s m"),
        "own_snr": rearrange(samples["source_snr"], "n m s -> n s m"),
    }
    joint = {"log_likelihood": samples["log_likelihood"]}
    # TARP: the truth's rank in distance to a random reference point; its ECDF - z is ECP(α) - α
    ref = samples["tarp_ref"]
    per_slot["tarp"] = np.linalg.norm(x - ref[:, :, None], axis=-1)
    joint["tarp"] = np.linalg.norm(
        rearrange(x, "n s m d -> n m (s d)") - rearrange(ref, "n s d -> n 1 (s d)"),
        axis=-1,
    )

    energy, crps, average, depth = [], [], [], []
    joint_energy, joint_average, joint_depth = [], [], []
    for j in range(n):
        energy.append(energy_score(x[j]))
        crps.append(
            np.stack([energy_score(x[j, :, :, c : c + 1]) for c in range(8)], -1)
        )
        a, d = pre_ranks(x[j])
        average.append(a)
        depth.append(d)
        whole = rearrange(x[j], "s m d -> 1 m (s d)")
        joint_energy.append(energy_score(whole)[0])
        a, d = pre_ranks(whole)
        joint_average.append(a[0])
        joint_depth.append(d[0])
    per_slot |= {"average_rank": np.stack(average), "band_depth": np.stack(depth)}
    joint |= {
        "average_rank": np.stack(joint_average),
        "band_depth": np.stack(joint_depth),
    }

    f0 = rearrange(samples["f0_bins"], "n m s -> n s m")[..., 1:]
    snr_hat = np.median(rearrange(samples["source_snr"], "n m s -> n s m")[..., 1:], -1)
    width = 1.4826 * np.median(np.abs(f0 - np.median(f0, -1, keepdims=True)), -1)
    return dict(
        draws=m - 1,
        rank={
            k: ranks(rearrange(v, "n s m -> (n s) m"), rng).reshape(n, n_sources)
            for k, v in per_slot.items()
        },
        joint_rank={k: ranks(v, rng) for k, v in joint.items()},
        # the strata: the draws' median snr, a function of y alone, never the true snr
        snr_hat=snr_hat,
        joint_snr_hat=np.median(samples["snr"][:, 1:], -1),
        energy=np.stack(energy),
        crps=np.stack(crps),
        joint_energy=np.array(joint_energy),
        # the absolute yardstick: f0 width over the monochromatic Fisher bound at the draws' snr
        width_ratio=width / (np.sqrt(3) / (np.pi * snr_hat)),
    )


def binomial_cdf(n: int, p: np.ndarray) -> np.ndarray:
    """P(Bin(n, p) <= k) for every p in (0, 1) and k = 0..n, [P, n + 1]."""
    k = np.arange(n + 1)
    log_choose = np.concat([[0.0], np.cumsum(np.log(n - k[1:] + 1) - np.log(k[1:]))])
    log_pmf = log_choose + k * np.log(p[:, None]) + (n - k) * np.log1p(-p[:, None])
    return np.minimum(np.cumsum(np.exp(log_pmf), -1), 1.0)


def ecdf_gamma(ranks: np.ndarray, k: int, cdf: np.ndarray) -> np.ndarray:
    """Säilynoja+22's γ of rank sets [..., N] on 0..k-1: twice the smallest binomial tail of their ECDF at z_i = i/k."""
    hist = np.apply_along_axis(np.bincount, -1, ranks, minlength=k)
    counts = hist.cumsum(-1)[..., :-1]
    rows = np.arange(k - 1)
    lower = cdf[rows, counts]
    upper = 1.0 - np.where(counts > 0, cdf[rows, np.maximum(counts - 1, 0)], 0.0)
    return 2 * np.minimum(lower, upper).min(-1)


@cache
def gamma_null(n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The binomial cdfs at z_i = i/k and GAMMA_SIMS draws of γ for N uniform ranks."""
    cdf = binomial_cdf(n, np.arange(1, k) / k)
    uniform = np.random.default_rng(0).integers(0, k, (GAMMA_SIMS, n))
    return cdf, ecdf_gamma(uniform, k, cdf)


def gamma_test(ranks: np.ndarray, k: int) -> tuple[float, np.ndarray]:
    """γ's p-value against uniform ranks, by simulation, and the ECDF - z's simultaneous 1 - ALPHA band."""
    n, z = len(ranks), np.arange(1, k) / k
    cdf, null = gamma_null(n, k)
    p_value = np.mean(null <= ecdf_gamma(ranks, k, cdf))
    critical = np.quantile(null, ALPHA)
    upper_tail = 1.0 - np.concat([np.zeros((k - 1, 1)), cdf[:, :-1]], -1)
    lo = (cdf <= critical / 2).sum(-1)
    hi = n - (upper_tail <= critical / 2).sum(-1)
    return p_value, np.stack([lo / n - z, hi / n - z])


def cell(r: np.ndarray, k: int) -> str:
    """p of γ, the mean rank's offset from the centre, and the share of ranks in the middle half."""
    if len(r) < 10:
        return f"{'-':>17}"
    p, _ = gamma_test(r, k)
    bias = r.mean() / (k - 1) - 0.5
    middle = np.mean((r >= k / 4) & (r < 3 * k / 4))
    return f"{p:5.2f} {bias:+5.2f} {middle:5.2f}"


def report(rows: dict, title: str) -> None:
    k = rows["draws"] + 1
    n_sources = rows["snr_hat"].shape[1]
    print(f"\n{title}")
    print(
        "per cell: p of the ECDF γ test, mean rank / M - 1/2 (> 0: the truth above the draws),"
        " share of ranks in the middle half (> 1/2 too wide, < 1/2 too narrow;"
        " for band depth, a high rank is a central truth: too wide)"
    )
    for lo, hi in [*SNR_BANDS, (0, np.inf)]:
        band = (rows["snr_hat"] >= lo) & (rows["snr_hat"] < hi)
        joint = (rows["joint_snr_hat"] >= lo) & (rows["joint_snr_hat"] < hi)
        label = f"{lo:g}-{hi:g}" if np.isfinite(hi) else (f">={lo:g}" if lo else "all")
        print(
            f"\ndraws' median snr {label}, sources per slot {band.sum(0).tolist()}, injections {joint.sum()}"
        )
        print(
            f"{'':>15}"
            + "".join(f"{f'slot {s}':>18}" for s in range(n_sources))
            + f"{'joint':>18}"
        )
        for name, r in rows["rank"].items():
            line = "".join(
                f" {cell(r[band[:, s], s], k)}" for s in range(n_sources)
            )
            if name in rows["joint_rank"]:
                line += f" {cell(rows['joint_rank'][name][joint], k)}"
            print(f"{name:>15}{line}")
        print(
            f"{'log likelihood':>15}{' ' * 18 * n_sources} {cell(rows['joint_rank']['log_likelihood'][joint], k)}"
        )
        median = lambda v, m: np.median(v[m]) if m.any() else np.nan
        print(
            f"{'energy score':>15}"
            + "".join(
                f"{median(rows['energy'][:, s], band[:, s]):18.3f}"
                for s in range(n_sources)
            )
            + f"{median(rows['joint_energy'], joint):18.3f}"
        )
        for c, name in enumerate(COORDS):
            print(
                f"{'crps_' + name:>15}"
                + "".join(
                    f"{median(rows['crps'][:, s, c], band[:, s]):18.3f}"
                    for s in range(n_sources)
                )
            )
        print(
            f"{'f0 width/ideal':>15}"
            + "".join(
                f"{median(rows['width_ratio'][:, s], band[:, s]):18.1f}"
                for s in range(n_sources)
            )
        )


def sample(
    config: run_config.Config, problem: LisaGB, flow: LisaFlow, epoch: int, out_dir: Path
) -> None:
    """Draw the flow on n_validation prior injections and save every quantity as validation.npz."""
    n, ode_steps = config.n_validation, config.ode_steps
    print(f"\nvalidation: injections {n}, draws {N_DRAWS}", flush=True)
    # a stream of plain prior injections apart from the eval's and the scorecard's
    keys = jr.split(jr.fold_in(jr.key(config.seed), 1), n)
    # BATCH injections per call: one injection's 127 draws leave the GPU latency-bound
    batched = eqx.filter_jit(
        eqx.filter_vmap(validate_injection, in_axes=(None, None, 0, None, None, None))
    )
    rows, start = [], time.perf_counter()
    for i in range(0, n, BATCH):
        rows.append(
            jax.tree.map(
                np.asarray,
                batched(
                    problem,
                    flow,
                    keys[i : i + BATCH],
                    config.time_power,
                    N_DRAWS,
                    ode_steps,
                ),
            )
        )
        if (i + BATCH) % 50 == 0:
            print(
                f"[{i + BATCH}/{n}] injections drawn, {time.perf_counter() - start:.0f} s",
                flush=True,
            )
    rows = {k: np.concat([r[k] for r in rows]) for k in rows[0]}
    path = out_dir / "validation.npz"
    np.savez(path, epoch=epoch, ode_steps=ode_steps, **pack_validation(rows))
    print(f"\nsaved {path}", flush=True)
    print_peak_memory()


def summarise(out_dir: Path, seed: int) -> None:
    """Rank and score validation.npz, print the report, and save validation_scores.npz."""
    samples = load_samples(out_dir / "validation.npz")
    rows = analyse(samples, seed)
    report(
        rows,
        f"{len(samples['f'])} prior injections (epoch {samples['epoch']}, {samples['ode_steps']} ODE steps)",
    )
    flat = {f"rank_{k}": v for k, v in rows["rank"].items()}
    flat |= {f"joint_rank_{k}": v for k, v in rows["joint_rank"].items()}
    flat |= {k: v for k, v in rows.items() if k not in ("rank", "joint_rank")}
    path = out_dir / "validation_scores.npz"
    np.savez(path, epoch=samples["epoch"], **flat)
    print(f"\nsaved {path}", flush=True)
