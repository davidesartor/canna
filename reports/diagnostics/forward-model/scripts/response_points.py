"""Each band's response_points against a 16384-point jaxgb reference: what a source's R-bin response misses.

Run from the repo root (JAX: a job, not the login node); a second argument `smoke` cuts the counts:
    uv run --frozen python reports/diagnostics/forward-model/scripts/response_points.py reports/diagnostics/forward-model/data
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import lisaorbits
import numpy as np
from einops import rearrange
from jaxgb import jaxgb
from jaxtyping import Array, Complex, Float

from canna.lisa import config
from canna.lisa.problem import LisaGB, noise_psd, sample_f
from canna.lisa.simulator import sample_physical

SMOKE = len(sys.argv) > 2 and sys.argv[2] == "smoke"
BANDS = ("narrow-B", "full-B")
SEED = 0
WINDOWS = 16 if SMOKE else 1024
CHUNK = 64
REFERENCE_POINTS = 16384
# response_points ladder on the first LADDER_SOURCES sources of each band: how the error falls with R
LADDER = tuple(2**k for k in range(5, 14))
LADDER_SOURCES = 64 if SMOKE else 256

out_dir = Path(sys.argv[1])
out_dir.mkdir(parents=True, exist_ok=True)
start_time = time.time()
commit = subprocess.run(
    ["git", "rev-parse", "HEAD"], capture_output=True, text=True
).stdout.strip()
src_dirty = bool(
    subprocess.run(
        ["git", "status", "--porcelain", "src"], capture_output=True, text=True
    ).stdout.strip()
)
record = dict(
    commit=commit,
    src_dirty=src_dirty,
    jax_version=jax.__version__,
    smoke=SMOKE,
    seed=SEED,
    windows=WINDOWS,
    reference_points=REFERENCE_POINTS,
)
arrays = {}


def tdi(response: jaxgb.JaxGB, params: Float[Array, "N 8"]) -> Complex[Array, "N R 3"]:
    return jnp.stack(response.get_tdi(params, 1.5, "AET"), axis=-1)


def embed(segments: Complex[Array, "N R 3"]) -> Complex[Array, "N M 3"]:
    """An R-bin response on the reference's bins: both centre on rint(f0 T)."""
    offset = (REFERENCE_POINTS - segments.shape[1]) // 2
    padded = jnp.zeros((segments.shape[0], REFERENCE_POINTS, 3), segments.dtype)
    return padded.at[:, offset : offset + segments.shape[1]].set(segments)


for band in BANDS:
    problem = LisaGB(**config.load(band).problem)
    points = problem.response_points
    response = lambda n: jaxgb.JaxGB(
        lisaorbits.EqualArmlengthOrbits(), t_obs=problem.t_obs, n=n
    )
    reference = response(REFERENCE_POINTS)
    assert float(reference.arm_length) == float(problem.response.arm_length)
    offset = (REFERENCE_POINTS - points) // 2

    def compare(params: Float[Array, "N 8"]) -> Float[Array, "5 N"]:
        """Per source: rho, the residual's rho, relative L2 error, reference power outside the R bins, rho at R."""
        exact = tdi(reference, params)
        padded = embed(tdi(problem.response, params))
        kmin = reference.get_kmin(params[:, 0])
        freqs = (kmin[:, None] + jnp.arange(REFERENCE_POINTS)) / problem.t_obs
        # the reference's far bins may reach DC, where the psd is singular; they hold no power
        freqs = jnp.maximum(jnp.abs(freqs), problem.f0_range[0] / 2)
        power = noise_psd(problem, freqs) * problem.t_obs / 2.0
        rho = lambda h: jnp.sqrt(2.0 * jnp.sum(jnp.abs(h) ** 2 / power, axis=(-2, -1)))
        l2 = lambda h: jnp.sum(jnp.abs(h) ** 2, axis=(-2, -1))
        inside = l2(exact[:, offset : offset + points])
        return jnp.stack(
            [
                rho(exact),
                rho(padded - exact),
                jnp.sqrt(l2(padded - exact) / l2(exact)),
                1.0 - inside / l2(exact),
                rho(padded),
            ]
        )

    # prior windows across the band, every source compared on its own response
    key_f, key_p = jr.split(jr.key(SEED))
    fs = jax.vmap(lambda k: sample_f(problem, k))(jr.split(key_f, WINDOWS))
    p = jax.lax.map(
        lambda kf: sample_physical(problem, kf[0], kf[1]),
        (jr.split(key_p, WINDOWS), fs),
        batch_size=64,
    )
    p = rearrange(p, "w s d -> (w s) d")
    fdot = np.asarray(p[:, 1])
    # jaxgb's parameters, as source_segments makes them
    p = p.at[:, 6].set(jnp.pi / 2 - p[:, 6])
    rho, rho_res, rel, outside, rho_r = np.asarray(
        rearrange(
            jax.lax.map(jax.jit(compare), rearrange(p, "(c n) d -> c n d", n=CHUNK)),
            "c k n -> k (c n)",
        )
    )
    f0 = np.asarray(p[:, 0])
    # f0's offset from its nearest bin: a boxcar leaks a sinc whose tails the R bins cut
    bin_offset = f0 * problem.t_obs - np.rint(f0 * problem.t_obs)
    on_bin, off_bin = np.abs(bin_offset) < 0.05, np.abs(bin_offset) > 0.45

    # the same comparison at every R of the ladder, on the first sources
    ladder_params = p[:LADDER_SOURCES]
    ladder_exact = tdi(reference, ladder_params)
    ladder_norm = jnp.sqrt(jnp.sum(jnp.abs(ladder_exact) ** 2, axis=(-2, -1)))
    ladder = np.stack(
        [
            np.asarray(
                jnp.sqrt(
                    jnp.sum(
                        jnp.abs(embed(tdi(response(n), ladder_params)) - ladder_exact)
                        ** 2,
                        axis=(-2, -1),
                    )
                )
                / ladder_norm
            )
            for n in LADDER
        ]
    )
    ladder_median = np.median(ladder, axis=1)
    # the slope of log median error against log R, from the band's R to a quarter of the reference
    above = (np.asarray(LADDER) >= points) & (np.asarray(LADDER) <= REFERENCE_POINTS // 4)
    slope = np.polyfit(np.log(np.asarray(LADDER)[above]), np.log(ladder_median[above]), 1)[0]

    name = band.split("-")[0]
    record |= {
        f"{name}_response_points": points,
        f"{name}_f0_lo_hz": problem.f0_range[0],
        f"{name}_f0_hi_hz": problem.f0_range[1],
        f"{name}_sources": int(f0.size),
        f"{name}_rel_median": float(np.median(rel)),
        f"{name}_rel_max": float(rel.max()),
        f"{name}_rel_median_on_bin": float(np.median(rel[on_bin])),
        f"{name}_rel_median_off_bin": float(np.median(rel[off_bin])),
        f"{name}_outside_median": float(np.median(outside)),
        f"{name}_outside_max": float(outside.max()),
        f"{name}_rho_res_median": float(np.median(rho_res)),
        f"{name}_rho_res_max": float(rho_res.max()),
        f"{name}_rho_res_over_1": float((rho_res > 1.0).mean()),
        f"{name}_rho_ratio_max_dev": float(np.abs(rho_r / rho - 1.0).max()),
        f"{name}_ladder_points": list(LADDER),
        f"{name}_ladder_rel_median": ladder_median.tolist(),
        f"{name}_ladder_slope": float(slope),
    }
    arrays |= {
        f"{name}_f0": f0,
        f"{name}_fdot": fdot,
        f"{name}_bin_offset": bin_offset,
        f"{name}_rho": rho,
        f"{name}_rho_res": rho_res,
        f"{name}_rel": rel,
        f"{name}_outside": outside,
        f"{name}_ladder_points": np.asarray(LADDER),
        f"{name}_ladder_rel": ladder,
    }
    print(
        f"{band}: R={points} vs {REFERENCE_POINTS}, {f0.size} sources: rel median"
        f" {np.median(rel):.2e} max {rel.max():.2e}; on-bin median"
        f" {np.median(rel[on_bin]):.2e}, off-bin {np.median(rel[off_bin]):.2e};"
        f" outside max {outside.max():.2e}; residual rho median"
        f" {np.median(rho_res):.2e} max {rho_res.max():.2e}, > 1 for"
        f" {(rho_res > 1).mean():.2%}; ladder median"
        f" {dict(zip(LADDER, np.round(ladder_median, 5)))}, slope {slope:.2f}"
    )


np.savez(out_dir / "response-points.npz", **arrays)
record["wall_seconds"] = time.time() - start_time
(out_dir / "response-points.json").write_text(json.dumps(record, indent=2) + "\n")
print(f"wrote {out_dir / 'response-points.json'} in {record['wall_seconds']:.0f} s")
