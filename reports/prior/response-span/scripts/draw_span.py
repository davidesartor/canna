"""How much of each prior source's above-noise power R response bins keep, against a wide jaxgb reference.

A enters the response linearly, so each source's shape is computed once at A = 1 and A is integrated
out exactly: the share of A above the noise in closed form, then stratified draws of A inside it.
f0 is drawn log-uniform over the band and weighted back to the prior, so the visible top is sampled;
fdot is not cut; the 5-sigma ceiling the sampler and autosizing once cut it at gives the old hard bound.

Run from the repo root (JAX: a job, not the login node):
    uv run --frozen python reports/prior/response-span/scripts/draw_span.py reports/prior/response-span/data [sources]
"""

import json
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import lisaorbits
import numpy as np
from einops import rearrange
from jax.scipy.special import ndtr, ndtri
from jaxgb import jaxgb
from jaxtyping import Array, Float, Int

from canna.lisa import config
from canna.lisa.physics import (
    ANTENNA_HARMONICS,
    EARTH_ORBIT_SPEED,
    FDOT_STD,
    SOURCE_COVARIANCE,
    SOURCE_INTERCEPT,
    SOURCE_MEAN,
    SOURCE_RESIDUAL,
    SOURCE_SLOPE,
    SPEED_OF_LIGHT,
    YEAR,
    log10_fdot_mean,
    response_span,
    sample_sky,
)
from canna.lisa.problem import LisaGB, noise_psd

CONFIG = "full-B"
SOURCES = int(sys.argv[2]) if len(sys.argv) > 2 else 65536
# stratified draws of A per source, inside the share of A that puts a bin above the noise
AMPLITUDES = 16
CHUNK = 32
# 4x the old hard bound: what falls beyond it is sinc leakage of ~1e-4 of the power
REFERENCE_POINTS = 4096
# the above-noise shares x whose centred width each draw records, and the R whose loss it records
KEEP = (0.5, 0.9, 0.99, 0.999, 1.0)
POWERS = tuple(2**k for k in range(2, 12))
# a complex noise bin's expected power per channel, in the units where a bin's signal power sums to snr²
NOISE_POWER = 2.0
GRID = 256
# log10 A given (log10 f0, log10 fdot): the prior's conditional Gaussian, as in physics.sample_sources
AMP_FDOT_SLOPE = SOURCE_RESIDUAL[0, 1] / SOURCE_RESIDUAL[0, 0]
AMP_STD = float(np.sqrt(SOURCE_RESIDUAL[1, 1] - AMP_FDOT_SLOPE * SOURCE_RESIDUAL[0, 1]))
# the fdot cut, this many FDOT_STDs above the mean, that once gave every response a hard bound
FDOT_SIGMAS = 5.0


def fdot_ceiling(f0: np.ndarray) -> np.ndarray:
    return 10.0 ** (np.asarray(log10_fdot_mean(f0)) + FDOT_SIGMAS * FDOT_STD)


def draw(
    log_f0_range: Float[Array, "2"], key: jax.Array
) -> tuple[Float[Array, "8"], Float[Array, ""], Float[Array, "J"]]:
    """One source at A = 1 with log f0 uniform on the band, the mean of its log10 A, and its A strata's uniforms."""
    keys = jr.split(key, 7)
    log_f0 = jr.uniform(keys[0], minval=log_f0_range[0], maxval=log_f0_range[1])
    residual_fdot = FDOT_STD * jr.normal(keys[1])
    fdot = 10.0 ** (SOURCE_INTERCEPT[0] + SOURCE_SLOPE[0] * log_f0 + residual_fdot)
    mean_log_a = (
        SOURCE_INTERCEPT[1] + SOURCE_SLOPE[1] * log_f0 + AMP_FDOT_SLOPE * residual_fdot
    )

    # a galactic sky, an isotropic orientation, as simulator.sample_physical
    sky = sample_sky(keys[2])
    psi = jr.uniform(keys[3], maxval=2 * jnp.pi)
    iota = jnp.arcsin(jr.uniform(keys[4], minval=-1.0, maxval=1.0))
    phi0 = jr.uniform(keys[5], maxval=2 * jnp.pi)
    p = jnp.concat(
        [jnp.stack([10.0**log_f0, fdot, 1.0]), sky, jnp.stack([psi, iota, phi0])]
    )
    strata = (jnp.arange(AMPLITUDES) + jr.uniform(keys[6], (AMPLITUDES,))) / AMPLITUDES
    return p, mean_log_a, strata


def kept_power(
    problem: LisaGB,
    reference: jaxgb.JaxGB,
    p: Float[Array, "N 8"],
    mean_log_a: Float[Array, "N"],
    strata: Float[Array, "N J"],
) -> tuple[
    Float[Array, "N"],
    Float[Array, "N"],
    Float[Array, "N J"],
    Int[Array, "N J K"],
    Float[Array, "N J P"],
]:
    """Per source its snr at A = 1 and the prior share of A above the noise; per A draw there, log10 A, the centred width holding each share in KEEP of the above-noise power, and the share each R in POWERS loses."""
    p = p.at[:, 6].set(jnp.pi / 2 - p[:, 6])
    h = jnp.stack(reference.get_tdi(p, 1.5, "AET"), axis=-1)
    kmin = reference.get_kmin(p[:, 0])
    freqs = (kmin[:, None] + jnp.arange(REFERENCE_POINTS)) / problem.t_obs
    shape = 2.0 * jnp.abs(h) ** 2 / (noise_psd(problem, freqs) * problem.t_obs / 2.0)

    # a bin rises above the noise iff A² · peak > NOISE_POWER; A's draws fill that tail, stratified by its survival
    peak = jnp.max(shape, axis=(-2, -1))
    threshold = (0.5 * jnp.log10(NOISE_POWER / peak) - mean_log_a) / AMP_STD
    seen = ndtr(-threshold)
    log_a = mean_log_a[:, None] - AMP_STD * ndtri(
        jnp.maximum(seen[:, None] * strata, 1e-300)
    )
    power = 10.0 ** (2 * log_a)[:, :, None, None] * shape[:, None]
    # per bin and channel, the signal power past the noise's expected power there
    excess = jnp.sum(jnp.maximum(power - NOISE_POWER, 0.0), axis=-1)

    # the share of it inside the 2m bins [rint(f0 T) - m, rint(f0 T) + m), m = 1 … M/2: an R-point response holds m = R/2
    cumulative = jnp.pad(jnp.cumsum(excess, axis=-1), ((0, 0), (0, 0), (1, 0)))
    above = cumulative[..., -1]
    centre, m = REFERENCE_POINTS // 2, jnp.arange(1, REFERENCE_POINTS // 2 + 1)
    kept = (cumulative[..., centre + m] - cumulative[..., centre - m]) / jnp.where(
        above > 0, above, 1.0
    )[..., None]
    kept = jnp.where(above[..., None] > 0, kept, 1.0)
    width = jnp.stack(
        [2 * (jnp.argmax(kept >= x * (1 - 1e-9), axis=-1) + 1) for x in KEEP], axis=-1
    )
    loss = jnp.stack([1.0 - kept[..., r // 2 - 1] for r in POWERS], axis=-1)
    return jnp.sqrt(jnp.sum(shape, axis=(-2, -1))), seen, log_a, width, loss


def span(f0: np.ndarray, fdot: np.ndarray, t_obs: float) -> np.ndarray:
    """response_span before its ceil: 2 (Doppler + drift + antenna) bins."""
    return 2 * (
        (f0 * EARTH_ORBIT_SPEED / SPEED_OF_LIGHT + fdot * t_obs) * t_obs
        + ANTENNA_HARMONICS * t_obs / YEAR
    )


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = jax.devices()[0]
    print(f"device: {device.device_kind}  x64: {jax.config.jax_enable_x64}", flush=True)
    problem = LisaGB(**config.load(CONFIG).problem)
    reference = jaxgb.JaxGB(
        lisaorbits.EqualArmlengthOrbits(), t_obs=problem.t_obs, n=REFERENCE_POINTS
    )
    start = time.perf_counter()

    # the sources, then every A draw's kept share against the reference
    log_f0_range = jnp.log10(jnp.asarray(problem.f0_range))
    p, mean_log_a, strata = jax.vmap(partial(draw, log_f0_range))(
        jr.split(jr.key(0), SOURCES)
    )
    snr_unit, seen, log_a, width, loss = (
        rearrange(np.asarray(a), "c n ... -> (c n) ...")
        for a in jax.lax.map(
            lambda chunk: jax.jit(partial(kept_power, problem, reference))(*chunk),
            tuple(
                rearrange(a, "(c n) ... -> c n ...", n=CHUNK)
                for a in (p, mean_log_a, strata)
            ),
        )
    )
    seconds = time.perf_counter() - start
    f0 = np.asarray(p[:, 0])

    # the prior's f0 marginal over the log-uniform draw, normalised to mean 1
    f0_weight = np.exp(
        -0.5 * ((np.log10(f0) - SOURCE_MEAN[0]) / np.sqrt(SOURCE_COVARIANCE[0, 0])) ** 2
    )
    f0_weight /= f0_weight.mean()
    source_weight = f0_weight * seen
    ess = source_weight.sum() ** 2 / (source_weight**2).sum()

    grid = np.geomspace(*problem.f0_range, GRID)
    np.savez(
        out_dir / "response-span.npz",
        f0=f0,
        f0_weight=f0_weight,
        snr_unit=snr_unit,
        mean_log_a=np.asarray(mean_log_a),
        seen=seen,
        log_a=log_a.astype(np.float32),
        width=width.astype(np.int16),
        loss=loss.astype(np.float32),
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "src"], capture_output=True, text=True
    ).stdout.strip()
    narrow = LisaGB(**config.load("narrow-B").problem)
    record = dict(
        commit=commit,
        src_dirty=bool(dirty),
        device=device.device_kind,
        config=CONFIG,
        sources=SOURCES,
        amplitudes=AMPLITUDES,
        amp_std=AMP_STD,
        noise_power=NOISE_POWER,
        t_obs=problem.t_obs,
        f0_range=list(problem.f0_range),
        response_points=problem.response_points,
        bound_points=1
        << (
            response_span(
                problem.f0_range[1], float(fdot_ceiling(problem.f0_range[1])), problem.t_obs
            )
            - 1
        ).bit_length(),
        narrow_response_points=narrow.response_points,
        narrow_f0_max=narrow.f0_range[1],
        reference_points=REFERENCE_POINTS,
        keep=list(KEEP),
        powers=list(POWERS),
        seconds=seconds,
        grid=dict(
            f0=grid.tolist(),
            ceiling=span(grid, fdot_ceiling(grid), problem.t_obs).tolist(),
            median=span(
                grid, 10.0 ** np.asarray(log10_fdot_mean(grid)), problem.t_obs
            ).tolist(),
        ),
        span_top=float(
            span(
                problem.f0_range[1],
                float(fdot_ceiling(problem.f0_range[1])),
                problem.t_obs,
            )
        ),
    )
    (out_dir / "response-span.json").write_text(json.dumps(record, indent=1) + "\n")
    print(
        f"{SOURCES} sources x {AMPLITUDES} A in {seconds:.0f} s; prior share above the noise {np.mean(source_weight):.4g},"
        f" effective sources {ess:.0f}; weighted median width {dict(zip(KEEP, np.median(width, axis=(0, 1)).tolist()))}",
        flush=True,
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
