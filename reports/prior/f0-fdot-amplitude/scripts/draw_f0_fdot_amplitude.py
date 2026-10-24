"""Draw LisaGB's (f0, fdot, A) catalogue prior per band: every source's parameters, snr and z at the prior's frame, the prior's curves and C(f0).

Then the windows at each snr cut of VISIBLE_SNRS (LisaGB.visible_snr) on the first CUT_WINDOWS windows' keys, with sample_window's weights.

Run from the repo root (JAX: a job, not the login node), optionally with fewer windows for a smoke test:
    uv run --frozen python reports/prior/f0-fdot-amplitude/scripts/draw_f0_fdot_amplitude.py reports/prior/f0-fdot-amplitude/data [windows]
"""

import json
import math
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from einops import rearrange
from jax.scipy.special import log_ndtr

from canna.lisa import config
from canna.lisa.chart import physical_to_flow, prior_moments
from canna.lisa.physics import (
    AMP_STD,
    ANTENNA_HARMONICS,
    EARTH_ORBIT_SPEED,
    SOURCE_COVARIANCE,
    SOURCE_INTERCEPT,
    SOURCE_MEAN,
    SOURCE_RESIDUAL,
    SOURCE_SLOPE,
    SPEED_OF_LIGHT,
    YEAR,
    equatorial_to_galactic,
    log10_amplitude_mean,
    log10_fdot_mean,
    sample_sky,
)
from canna.lisa.problem import LisaGB, amplitude_unit, sample_f, window_centre
from canna.lisa.simulator import (
    VISIBLE_WINDOWS,
    sample_physical,
    sample_window,
    snr,
    source_snrs,
)

BANDS = ("narrow", "full")
WINDOWS = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
GRID = 97
CUT = math.log10(1e-4)  # the catalogue's f0 cut, where the fit truncates the parent
SKY_SOURCES = 8192 if WINDOWS >= 8192 else 512
SKY_CHUNK = 256
SKY_GRID = 16
SKY_FLAT_BELOW = 1.5e-3  # [Hz]
MONOCHROMATIC_DRIFT = 1e-4  # [bins] fdot T^2 of the C(f0) sources
# the snr cuts the page's slider steps through; each draws VISIBLE_WINDOWS single-source jaxgb responses per window
VISIBLE_SNRS = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0)
CUT_WINDOWS = WINDOWS // 2
WEIGHT_WINDOWS = WINDOWS // 8


def draw(problem: LisaGB, key: jax.Array) -> dict:
    """One window, by the catalogue at visible_snr = 0: its sources in f0 order, their snrs, z at the prior's frame and A in the window's amplitude unit."""
    key_f, key_p = jr.split(key)
    f = (sample_window if problem.visible_snr > 0 else sample_f)(problem, key_f)
    p = sample_physical(problem, key_p, f)
    p = p[jnp.argsort(p[:, 0])]
    # the flow's target z1 = (x1 - mu) / sigma, in the frame the network starts from (delta = 0)
    mean, std = prior_moments(problem, f)
    return dict(
        f=f,
        p=p,
        z=(physical_to_flow(problem, p, f) - mean) / std,
        snr=source_snrs(problem, p, f),
        unit_f=amplitude_unit(problem, f),
        unit_f0=jax.vmap(partial(amplitude_unit, problem))(p[:, 0]),
    )


def window_weights(problem: LisaGB, key: jax.Array) -> dict:
    """sample_window's candidates at the key draw's own key: their f0, log10 P(snr >= cut | them), the resample's ESS / M and its pick."""
    key_f, _ = jr.split(key)
    key_f0, key_sky, key_psi, key_iota, key_pick = jr.split(key_f, 5)
    shape = (VISIBLE_WINDOWS, 1)
    log_f0 = jr.uniform(
        key_f0,
        shape,
        minval=math.log10(problem.f0_range[0]),
        maxval=math.log10(problem.f0_range[1]),
    )
    f0 = 10.0**log_f0
    f = jax.vmap(partial(window_centre, problem))(f0[:, 0])
    sky = jax.vmap(sample_sky)(jr.split(key_sky, VISIBLE_WINDOWS))
    psi = jr.uniform(key_psi, shape, maxval=2 * jnp.pi)
    iota = jnp.arcsin(jr.uniform(key_iota, shape, minval=-1.0, maxval=1.0))
    unit = amplitude_unit(problem, f)[:, None]
    p = jnp.concat(
        [f0, 10.0 ** log10_fdot_mean(f0), unit, sky, psi, iota, jnp.zeros(shape)],
        axis=-1,
    )
    unit_snr = jax.vmap(lambda p, f: snr(problem, p[None], f))(p, f)
    log10_cut = jnp.log10(problem.visible_snr * unit[:, 0] / unit_snr)
    cut = (log10_cut - log10_amplitude_mean(f0[:, 0])) / AMP_STD
    log_p = -0.5 * (log_f0[:, 0] - SOURCE_MEAN[0]) ** 2 / SOURCE_COVARIANCE[0, 0]
    log_w = log_p + log_ndtr(-cut)
    w = jnp.exp(log_w - log_w.max())
    return dict(
        cand_log_f0=log_f0[:, 0],
        log_mass=log_ndtr(-cut) / math.log(10.0),
        ess=w.sum() ** 2 / (w**2).sum() / VISIBLE_WINDOWS,
        pick=f[jr.categorical(key_pick, log_w)],
    )


def source_arrays(problem: LisaGB, d: dict) -> dict:
    """Per source: the parameters, snr, drift, response span, sky latitude, inclination and the flow's z."""
    p, t = d["p"], problem.t_obs
    drift = p[..., 1] * t**2
    doppler = p[..., 0] * EARTH_ORBIT_SPEED / SPEED_OF_LIGHT * t
    return dict(
        f=d["f"],
        log_f0=np.log10(p[..., 0]),
        log_fdot=np.log10(p[..., 1]),
        log_a=np.log10(p[..., 2]),
        snr=d["snr"],
        drift=drift,
        span=2 * np.ceil(doppler + drift + ANTENNA_HARMONICS * t / YEAR),
        z=d["z"],
        log_a_unit=np.log10(p[..., 2] / d["unit_f"][:, None]),
        sin_iota=np.sin(p[..., 6]),
        b=np.asarray(equatorial_to_galactic(p[..., 3], p[..., 4])[1]),
    )


def draw_windows(problem: LisaGB, n: int) -> dict:
    d = jax.lax.map(
        partial(draw, problem), jr.split(jr.key(0), WINDOWS)[:n], batch_size=128
    )
    return {k: np.asarray(v) for k, v in jax.block_until_ready(d).items()}


def band_record(problem: LisaGB, d: dict, seconds: float) -> dict:
    """The band's problem, the prior's curves against log10 f0 and the snr-7 amplitude."""
    f_lo, f_hi = problem.f0_range
    grid = np.linspace(math.log10(f_lo), math.log10(f_hi), GRID)
    cond = SOURCE_INTERCEPT[:, None] + SOURCE_SLOPE[:, None] * grid
    unit_grid = np.asarray(
        jax.vmap(partial(amplitude_unit, problem))(jnp.asarray(10.0**grid))
    )
    # kappa: the snr at one amplitude_unit of the source's own f0; <kappa^2> sets the snr-7 amplitude
    p = d["p"]
    kappa2 = (d["snr"] / (p[..., 2] / d["unit_f0"])) ** 2
    return dict(
        problem=dict(
            f0_range=[f_lo, f_hi],
            t_obs=problem.t_obs,
            n_sources=problem.n_sources,
            response_points=problem.response_points,
            window_bins=problem.window_bins,
        ),
        windows=int(p.shape[0]),
        seconds=seconds,
        curves=dict(
            log_f0=grid.tolist(),
            fdot_mean=cond[0].tolist(),
            amp_mean=cond[1].tolist(),
            amp_snr7=np.log10(7.0 * unit_grid / math.sqrt(kappa2.mean())).tolist(),
        ),
        kappa2_mean=float(kappa2.mean()),
    )


def sky_average(problem: LisaGB) -> dict:
    """C(f0) = <rho^2> at A = amplitude_unit(f0), the same orientations at every f0, under an isotropic and the galactic sky."""
    keys = jr.split(jr.key(1), 6)
    n = SKY_SOURCES
    galactic = jax.vmap(sample_sky)(jr.split(keys[2], n))
    skies = dict(
        isotropic=(
            jr.uniform(keys[0], (n,), maxval=2 * jnp.pi),
            jnp.arcsin(jr.uniform(keys[1], (n,), minval=-1.0, maxval=1.0)),
        ),
        galactic=(galactic[:, 0], galactic[:, 1]),
    )
    psi = jr.uniform(keys[3], (n,), maxval=2 * jnp.pi)
    iota = jnp.arcsin(jr.uniform(keys[4], (n,), minval=-1.0, maxval=1.0))
    phi0 = jr.uniform(keys[5], (n,), maxval=2 * jnp.pi)
    grid = np.geomspace(*problem.f0_range, SKY_GRID)
    fdot = MONOCHROMATIC_DRIFT / problem.t_obs**2

    # each source alone in the window centred on it, so rho^2 = C(f0) for that orientation
    @jax.jit
    def kappa_squared(f0, lon, lat):
        ones = jnp.ones(n)
        unit = amplitude_unit(problem, f0)
        p = jnp.stack(
            [f0 * ones, fdot * ones, unit * ones, lon, lat, psi, iota, phi0], axis=-1
        )
        chunks = rearrange(p, "(c b) d -> c b d", b=SKY_CHUNK)
        rho = jax.lax.map(partial(source_snrs, problem, f=f0), chunks)
        return rho.ravel() ** 2

    out = dict(f0=grid.tolist(), sources=n, flat_below=SKY_FLAT_BELOW)
    for name, (lon, lat) in skies.items():
        c = np.stack(
            [np.asarray(kappa_squared(jnp.asarray(f0), lon, lat)) for f0 in grid]
        )
        low = c[grid < SKY_FLAT_BELOW].mean(0)
        c_low = low.mean()
        ratio = c.mean(1) / c_low
        # paired with the low-f0 mean over the same orientations, so the ratio's error is the drift's alone
        out[name] = dict(
            C=c.mean(1).tolist(),
            C_se=(c.std(1) / math.sqrt(n)).tolist(),
            C_low=float(c_low),
            C_low_se=float(low.std() / math.sqrt(n)),
            ratio=ratio.tolist(),
            ratio_se=(
                (c - ratio[:, None] * low).std(1) / math.sqrt(n) / c_low
            ).tolist(),
        )
    return out


def prior_record() -> dict:
    sigma = np.sqrt(np.diag(SOURCE_COVARIANCE))
    return dict(
        mean=SOURCE_MEAN.tolist(),
        cov=SOURCE_COVARIANCE.tolist(),
        std=sigma.tolist(),
        corr=(SOURCE_COVARIANCE / np.outer(sigma, sigma)).tolist(),
        slope=SOURCE_SLOPE.tolist(),
        intercept=SOURCE_INTERCEPT.tolist(),
        residual_std=np.sqrt(np.diag(SOURCE_RESIDUAL)).tolist(),
        residual_corr=float(
            SOURCE_RESIDUAL[0, 1]
            / math.sqrt(SOURCE_RESIDUAL[0, 0] * SOURCE_RESIDUAL[1, 1])
        ),
        cut=CUT,
    )


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = jax.devices()[0]
    print(f"device: {device.device_kind}  x64: {jax.config.jax_enable_x64}", flush=True)
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
        prior=prior_record(),
        bands={},
    )
    for band in BANDS:
        training = LisaGB(**config.load(f"{band}-B").problem)
        problem = LisaGB(**config.load(f"{band}-B").problem | dict(visible_snr=0.0))
        start = time.perf_counter()
        d = draw_windows(problem, WINDOWS)
        seconds = time.perf_counter() - start
        record["bands"][band] = band_record(problem, d, seconds) | dict(
            training_visible_snr=training.visible_snr
        )
        record["bands"][band]["sky_average"] = sky_average(problem)

        np.savez(out_dir / f"{band}.npz", **source_arrays(problem, d))
        print(
            f"{band}: {WINDOWS} windows in {seconds:.1f} s; snr > 7 for {(d['snr'] > 7).mean():.4f},"
            f" windows with none {(~(d['snr'] > 7).any(1)).mean():.4f}",
            flush=True,
        )

        # the same window keys under each snr cut, and sample_window's weights for the first of them
        cuts, cut_seconds = [], []
        for k in VISIBLE_SNRS:
            cut_problem = LisaGB(
                **config.load(f"{band}-B").problem | dict(visible_snr=k)
            )
            start = time.perf_counter()
            d = draw_windows(cut_problem, CUT_WINDOWS)
            weights = jax.lax.map(
                partial(window_weights, cut_problem),
                jr.split(jr.key(0), WINDOWS)[:WEIGHT_WINDOWS],
                batch_size=128,
            )
            weights = jax.tree.map(np.asarray, weights)
            cut_seconds.append(time.perf_counter() - start)
            # the mirror must pick sample_window's own centres
            assert np.allclose(weights.pop("pick"), d["f"][:WEIGHT_WINDOWS], rtol=0)
            cuts.append(source_arrays(cut_problem, d) | weights)
            print(
                f"{band} snr >= {k:g}: {cut_seconds[-1]:.1f} s; windows with a source at snr >= k"
                f" {(d['snr'] >= k).any(1).mean():.4f}, median ESS/M {np.median(weights['ess']):.3f}",
                flush=True,
            )
        np.savez(
            out_dir / f"{band}-cut.npz",
            visible_snr=np.array(VISIBLE_SNRS),
            **{key: np.stack([c[key] for c in cuts]) for key in cuts[0]},
        )
        record["bands"][band]["cut"] = dict(
            visible_snr=list(VISIBLE_SNRS),
            windows=CUT_WINDOWS,
            candidates=VISIBLE_WINDOWS,
            weight_windows=WEIGHT_WINDOWS,
            seconds=cut_seconds,
        )
    out = out_dir / "f0-fdot-amplitude.json"
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
