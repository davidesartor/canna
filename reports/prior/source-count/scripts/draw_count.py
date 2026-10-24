"""Per prior source its snr at A = 1 and the mean of its log10 A, so the share reaching any snr integrates A out exactly.

f0 is drawn log-uniform over the band and weighted back to the prior, so the visible top is sampled.
The draws are response-span's (same key, same draw), without its kept-power bookkeeping.

Run from the repo root (JAX: a job, not the login node):
    uv run --frozen python reports/prior/source-count/scripts/draw_count.py reports/prior/source-count/data [sources]
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
from jaxgb import jaxgb
from jaxtyping import Array, Float

from canna.lisa import config
from canna.lisa.physics import (
    FDOT_STD,
    SOURCE_COVARIANCE,
    SOURCE_INTERCEPT,
    SOURCE_MEAN,
    SOURCE_RESIDUAL,
    SOURCE_SLOPE,
    sample_sky,
)
from canna.lisa.problem import LisaGB, noise_psd

CONFIG = "full-B"
SOURCES = int(sys.argv[2]) if len(sys.argv) > 2 else 65536
CHUNK = 32
# wide enough that the fastest chirp's whole power lies inside
REFERENCE_POINTS = 4096
# log10 A given (log10 f0, log10 fdot): the prior's conditional Gaussian, as in physics.sample_sources
AMP_FDOT_SLOPE = SOURCE_RESIDUAL[0, 1] / SOURCE_RESIDUAL[0, 0]
AMP_STD = float(np.sqrt(SOURCE_RESIDUAL[1, 1] - AMP_FDOT_SLOPE * SOURCE_RESIDUAL[0, 1]))


def draw(
    log_f0_range: Float[Array, "2"], key: jax.Array
) -> tuple[Float[Array, "8"], Float[Array, ""]]:
    """One source at A = 1 with log f0 uniform on the band, and the mean of its log10 A."""
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
    return p, mean_log_a


def snr_unit(
    problem: LisaGB, reference: jaxgb.JaxGB, p: Float[Array, "N 8"]
) -> Float[Array, "N"]:
    """Per source its snr at A = 1 against the band's A, E, T noise."""
    p = p.at[:, 6].set(jnp.pi / 2 - p[:, 6])
    h = jnp.stack(reference.get_tdi(p, 1.5, "AET"), axis=-1)
    kmin = reference.get_kmin(p[:, 0])
    freqs = (kmin[:, None] + jnp.arange(REFERENCE_POINTS)) / problem.t_obs
    power = 2.0 * jnp.abs(h) ** 2 / (noise_psd(problem, freqs) * problem.t_obs / 2.0)
    return jnp.sqrt(jnp.sum(power, axis=(-2, -1)))


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = jax.devices()[0]
    print(f"device: {device.device_kind}  x64: {jax.config.jax_enable_x64}", flush=True)
    problem = LisaGB(**config.load(CONFIG).problem)
    reference = jaxgb.JaxGB(
        lisaorbits.EqualArmlengthOrbits(), t_obs=problem.t_obs, n=REFERENCE_POINTS
    )
    start = time.perf_counter()

    log_f0_range = jnp.log10(jnp.asarray(problem.f0_range))
    p, mean_log_a = jax.vmap(partial(draw, log_f0_range))(jr.split(jr.key(0), SOURCES))
    snr = rearrange(
        np.asarray(
            jax.lax.map(
                jax.jit(partial(snr_unit, problem, reference)),
                rearrange(p, "(c n) d -> c n d", n=CHUNK),
            )
        ),
        "c n -> (c n)",
    )
    seconds = time.perf_counter() - start
    f0 = np.asarray(p[:, 0])

    # the prior's f0 marginal over the log-uniform draw, normalised to mean 1
    f0_weight = np.exp(
        -0.5 * ((np.log10(f0) - SOURCE_MEAN[0]) / np.sqrt(SOURCE_COVARIANCE[0, 0])) ** 2
    )
    f0_weight /= f0_weight.mean()

    np.savez(
        out_dir / "source-count.npz",
        f0=f0,
        f0_weight=f0_weight,
        snr_unit=snr,
        mean_log_a=np.asarray(mean_log_a),
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "src"], capture_output=True, text=True
    ).stdout.strip()
    record = dict(
        commit=commit,
        src_dirty=bool(dirty),
        device=device.device_kind,
        config=CONFIG,
        sources=SOURCES,
        amp_std=AMP_STD,
        t_obs=problem.t_obs,
        f0_range=list(problem.f0_range),
        response_points=problem.response_points,
        n_sources=problem.n_sources,
        reference_points=REFERENCE_POINTS,
        seconds=seconds,
    )
    (out_dir / "source-count.json").write_text(json.dumps(record, indent=1) + "\n")
    print(
        f"{SOURCES} sources in {seconds:.0f} s; median snr at A = 1 {np.median(snr):.4g}",
        flush=True,
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
