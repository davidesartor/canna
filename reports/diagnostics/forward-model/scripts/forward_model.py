"""The forward model's three checks: jaxgb's TDI X against Babak et al.'s, the sky-averaged sensitivity's slope, the noise generator.

Run from the repo root (JAX: a job, not the login node); a second argument `smoke` cuts the counts:
    uv run --frozen python reports/diagnostics/forward-model/scripts/forward_model.py reports/diagnostics/forward-model/data
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
import numpy as np
from einops import rearrange
from jaxtyping import Array, Complex, Float
from scipy import stats

from canna.lisa import config
from canna.lisa.physics import SPEED_OF_LIGHT, fdot_from_chirp_mass
from canna.lisa.problem import (
    LisaGB,
    noise_psd,
    sample_f,
    single_link_noise,
    window_freqs,
    window_power,
)
from canna.lisa.simulator import sample_observation, sample_physical

SMOKE = len(sys.argv) > 2 and sys.argv[2] == "smoke"
BAND = "full-B"

TDI_SEED = 0
TDI_WINDOWS = 64 if SMOKE else 2048
TDI_CHUNK = 256
TDI_LOW = 3e-3  # [Hz]
TDI_HIGH = (9e-3, 11e-3)  # [Hz]

SKY_SEED = 0
SKY_DRAWS = 256
SKY_GRID = np.logspace(np.log10(2e-4), np.log10(3e-2), 16)
SKY_FIT_POINTS = 8
SKY_FLAT_BELOW = 4e-3  # [Hz]

NOISE_SEED = 0
NOISE_DRAWS = 100 if SMOKE else 3000
NOISE_F = 1.40457e-3  # [Hz]
NOISE_PROBLEM = dict(n_sources=1, response_points=256, f0_range=(1.0e-3, 2.0e-3))
NOISE_KS_STRIDE = 37

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
    commit=commit, src_dirty=src_dirty, jax_version=jax.__version__, smoke=SMOKE
)

problem = LisaGB(**config.load(BAND).problem)
response = problem.response
f_star = float(np.mean(response.fstar))
arm_length = float(np.mean(response.arm_length))
print(f"{BAND}: response_points {problem.response_points}, f* {f_star * 1e3:.3f} mHz")


def tdi_x(params: Float[Array, "N 8"]) -> tuple[Complex[Array, "N R"], ...]:
    """jaxgb's X as get_tdi returns it, rebuilt by hand, and Babak et al.'s eq. 14 X, on the same link envelopes."""
    gs = response._get_gs(params)
    x = (params[:, 0:1] + params[:, 1:2] * response.tm) / f_star
    delay = jnp.exp(-1j * x)
    one_minus_d2 = 1.0 - delay**2
    g12, g23, g31, g13, g32, g21 = rearrange(gs, "n l t -> l n t")
    jaxgb_x = -one_minus_d2 * ((g21 - g31) + delay * (g12 - g13))
    babak_x = one_minus_d2 * ((g13 - g12) + delay * (g31 - g21))
    spectra = (
        jnp.fft.fftshift(jnp.fft.fft(jnp.stack([jaxgb_x, babak_x]), axis=-1), axes=-1)
        * 0.5
        * response.dtm
    )
    return response.get_tdi(params, 1.5, "XYZ")[0], spectra[0], spectra[1]


def tdi_compare(params: Float[Array, "N 8"]) -> Float[Array, "3 N"]:
    """Per source: relative L2 difference Babak - jaxgb, power ratio - 1, and the rebuild's error."""
    reference, jaxgb_x, babak_x = tdi_x(params)
    norm = partial(jnp.linalg.norm, axis=-1)
    return jnp.stack(
        [
            norm(babak_x - jaxgb_x) / norm(jaxgb_x),
            (norm(babak_x) / norm(jaxgb_x)) ** 2 - 1.0,
            norm(jaxgb_x - reference) / norm(reference),
        ]
    )


# TDI: prior windows across the full band, every source compared on its own response
key_f, key_p = jr.split(jr.key(TDI_SEED))
fs = jax.vmap(partial(sample_f, problem))(jr.split(key_f, TDI_WINDOWS))
p = jax.lax.map(
    lambda kf: sample_physical(problem, kf[0], kf[1]),
    (jr.split(key_p, TDI_WINDOWS), fs),
    batch_size=64,
)
p = rearrange(p, "w s d -> (w s) d")
p = p.at[:, 1].set(fdot_from_chirp_mass(p[:, 1], p[:, 0]))
p = p.at[:, 6].set(jnp.pi / 2 - p[:, 6])
rel, power_dev, reimpl = np.asarray(
    rearrange(
        jax.lax.map(tdi_compare, rearrange(p, "(c n) d -> c n d", n=TDI_CHUNK)),
        "c k n -> k (c n)",
    )
)
f0 = np.asarray(p[:, 0])
low = f0 <= TDI_LOW
high = (f0 >= TDI_HIGH[0]) & (f0 <= TDI_HIGH[1])
slope = np.polyfit(np.log(f0), np.log(rel), 1)[0]
record |= dict(
    tdi_band=BAND,
    tdi_seed=TDI_SEED,
    tdi_windows=TDI_WINDOWS,
    tdi_sources=int(f0.size),
    tdi_low_hz=TDI_LOW,
    tdi_high_lo_hz=TDI_HIGH[0],
    tdi_high_hi_hz=TDI_HIGH[1],
    tdi_sources_low=int(low.sum()),
    tdi_sources_high=int(high.sum()),
    tdi_rel_max_low=float(rel[low].max()),
    tdi_rel_median_low=float(np.median(rel[low])),
    tdi_rel_max_high=float(rel[high].max()),
    tdi_rel_median_high=float(np.median(rel[high])),
    tdi_rel_max=float(rel.max()),
    tdi_rel_loglog_slope=float(slope),
    tdi_power_dev_max_low=float(np.abs(power_dev[low]).max()),
    tdi_power_dev_median_high=float(np.median(np.abs(power_dev[high]))),
    tdi_power_dev_max_high=float(np.abs(power_dev[high]).max()),
    tdi_power_dev_max=float(np.abs(power_dev).max()),
    tdi_jaxgb_reimpl_max=float(reimpl.max()),
)
np.savez(out_dir / "tdi.npz", f0=f0, rel=rel, power_dev=power_dev, f_star=f_star)
print(
    f"TDI over {f0.size} sources: rel max {rel[low].max():.3e} / median"
    f" {np.median(rel[low]):.3e} up to 3 mHz, median {np.median(rel[high]):.3e} /"
    f" max {rel[high].max():.3e} at 10 mHz ({high.sum()}), slope {slope:.3f};"
    f" |power - 1| max {np.abs(power_dev).max():.3e}; rebuilt vs get_tdi {reimpl.max():.1e}"
)

# sky-averaged sensitivity from the Monte-Carlo A, E response of plus-polarised unit sources
rng = np.random.default_rng(SKY_SEED)
lon = rng.uniform(0, 2 * np.pi, SKY_DRAWS)
lat = np.arcsin(rng.uniform(-1, 1, SKY_DRAWS))
psi = rng.uniform(0, 2 * np.pi, SKY_DRAWS)
phi0 = rng.uniform(0, 2 * np.pi, SKY_DRAWS)
ones = np.ones(SKY_DRAWS)
response_power = []
for f_c in SKY_GRID:
    params = np.stack(
        [f_c * ones, 0 * ones, ones, lon, lat, psi, np.pi / 2 * ones, phi0], axis=-1
    )
    aet = jnp.stack(response.get_tdi(jnp.asarray(params), 1.5, "AET"), axis=1)
    response_power.append(
        8.0 * jnp.mean(jnp.sum(jnp.abs(aet) ** 2, axis=-1), axis=0) / problem.t_obs**2
    )
response_power = np.asarray(jnp.stack(response_power))
psd = np.asarray(noise_psd(problem, jnp.asarray(SKY_GRID)))
x = 2 * np.pi * SKY_GRID * arm_length / SPEED_OF_LIGHT
s_sky = 1.0 / np.sum(response_power[:, :2] / psd[:, :2], axis=-1)
s_sky_aet = 1.0 / np.sum(response_power / psd, axis=-1)
s_sky_length = 1.0 / np.sum(
    response_power[:, :2] / (psd[:, :2] / (4 * x[:, None] ** 2)), axis=-1
)
p_oms, p_acc = (
    np.asarray(v) for v in single_link_noise(problem, jnp.asarray(SKY_GRID))
)
s_robson = (
    10.0
    / (3.0 * arm_length**2)
    * (
        p_oms
        + 2.0
        * (1.0 + np.cos(SKY_GRID / f_star) ** 2)
        * p_acc
        / (2 * np.pi * SKY_GRID) ** 4
    )
    * (1.0 + 0.6 * (SKY_GRID / f_star) ** 2)
)
log_f = np.log(SKY_GRID)
fit = slice(0, SKY_FIT_POINTS)
flat = SKY_GRID < SKY_FLAT_BELOW
ratio, ratio_aet, ratio_length = (
    s / s_robson for s in (s_sky, s_sky_aet, s_sky_length)
)
record |= dict(
    sky_seed=SKY_SEED,
    sky_draws=SKY_DRAWS,
    sky_grid_points=SKY_GRID.size,
    sky_grid_lo_hz=float(SKY_GRID[0]),
    sky_grid_hi_hz=float(SKY_GRID[-1]),
    sky_fit_points=SKY_FIT_POINTS,
    sky_fit_hi_hz=float(SKY_GRID[SKY_FIT_POINTS - 1]),
    sky_slope=float(np.polyfit(log_f[fit], np.log(ratio[fit]), 1)[0]),
    sky_slope_below_4mHz=float(np.polyfit(log_f[flat], np.log(ratio[flat]), 1)[0]),
    sky_slope_aet=float(np.polyfit(log_f[fit], np.log(ratio_aet[fit]), 1)[0]),
    sky_slope_length=float(np.polyfit(log_f[fit], np.log(ratio_length[fit]), 1)[0]),
    sky_ratio_min_below_4mHz=float(ratio[flat].min()),
    sky_ratio_max_below_4mHz=float(ratio[flat].max()),
    sky_ratio_mean_below_4mHz=float(ratio[flat].mean()),
)
np.savez(
    out_dir / "sky.npz",
    f=SKY_GRID,
    ratio=ratio,
    ratio_aet=ratio_aet,
    ratio_length=ratio_length,
)
print("sky-averaged S / Robson:", np.array2string(ratio, precision=4))
print(
    f"  slope {record['sky_slope']:+.4f} (first {SKY_FIT_POINTS}),"
    f" {record['sky_slope_below_4mHz']:+.4f} (< 4 mHz), A+E+T {record['sky_slope_aet']:+.4f},"
    f" fractional length {record['sky_slope_length']:+.4f}"
)

# noise generator: pure-noise windows whitened by window_power
noise_problem = LisaGB(**NOISE_PROBLEM)
f_window = jnp.asarray(NOISE_F)
silent = jnp.array([[NOISE_F, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
noise = jax.lax.map(
    lambda k: sample_observation(noise_problem, k, silent, f_window),
    jr.split(jr.key(NOISE_SEED), NOISE_DRAWS),
)
power = window_power(noise_problem, f_window)
whitened = np.asarray(noise / jnp.sqrt(power))
re, im = whitened.real, whitened.imag
bins = noise_problem.window_bins
abs2 = np.abs(whitened) ** 2

empirical = (np.abs(np.asarray(noise)) ** 2).mean(axis=0) * 2.0 / noise_problem.t_obs
psd_ratio = empirical / np.asarray(
    noise_psd(noise_problem, window_freqs(noise_problem, f_window))
)
half = bins // 2
tilt = np.abs(psd_ratio[half:].mean(axis=0) - psd_ratio[:half].mean(axis=0))

nn = np.asarray(
    2.0 * jnp.sum(jnp.abs(noise[:, :, :2]) ** 2 / power[:, :2], axis=(-2, -1))
)
dof = 2 * bins * 2
record |= dict(
    noise_seed=NOISE_SEED,
    noise_draws=NOISE_DRAWS,
    noise_window_bins=bins,
    noise_response_points=noise_problem.response_points,
    noise_f_hz=NOISE_F,
    noise_t_obs=noise_problem.t_obs,
    noise_mean_re=float(re.mean()),
    noise_mean_im=float(im.mean()),
    noise_var_re=float(re.var()),
    noise_var_im=float(im.var()),
    noise_corr_re_im=float(np.corrcoef(re.ravel(), im.ravel())[0, 1]),
    noise_abs2=float(abs2.mean()),
    noise_abs2_a=float(abs2[..., 0].mean()),
    noise_abs2_e=float(abs2[..., 1].mean()),
    noise_abs2_t=float(abs2[..., 2].mean()),
    noise_corr_floor=float(1 / np.sqrt(NOISE_DRAWS * bins)),
    noise_corr_adjacent=float(
        np.corrcoef(re[:, :-1, 0].ravel(), re[:, 1:, 0].ravel())[0, 1]
    ),
    noise_corr_a_e=float(np.corrcoef(re[..., 0].ravel(), re[..., 1].ravel())[0, 1]),
    noise_corr_a_t=float(np.corrcoef(re[..., 0].ravel(), re[..., 2].ravel())[0, 1]),
    noise_corr_seeds=float(
        np.corrcoef(re[:-1, :, 0].ravel(), re[1:, :, 0].ravel())[0, 1]
    ),
    noise_ks_re_p=float(
        stats.kstest(
            re[:, ::NOISE_KS_STRIDE, 0].ravel(), stats.norm(0, np.sqrt(0.5)).cdf
        ).pvalue
    ),
    noise_psd_ratio_mean_a=float(psd_ratio[:, 0].mean()),
    noise_psd_ratio_mean_e=float(psd_ratio[:, 1].mean()),
    noise_psd_ratio_mean_t=float(psd_ratio[:, 2].mean()),
    noise_psd_tilt_a=float(tilt[0]),
    noise_psd_tilt_e=float(tilt[1]),
    noise_psd_tilt_t=float(tilt[2]),
    noise_psd_tilt_floor=float(2 / np.sqrt(NOISE_DRAWS * bins)),
    noise_nn_dof=dof,
    noise_nn_mean=float(nn.mean()),
    noise_nn_var=float(nn.var()),
    noise_nn_ks_p=float(stats.kstest(nn, stats.chi2(dof).cdf).pvalue),
)
print(
    f"noise, {NOISE_DRAWS} windows of {bins} bins: var re {re.var():.5f} im {im.var():.5f},"
    f" corr {record['noise_corr_re_im']:+.5f}, E|z|^2 {abs2.mean():.5f},"
    f" tilt A/E/T {np.array2string(tilt, precision=4)} (floor {record['noise_psd_tilt_floor']:.4f}),"
    f" <n|n> mean {nn.mean():.1f} var {nn.var():.0f} (dof {dof}), KS p {record['noise_nn_ks_p']:.3f}"
)

record["wall_seconds"] = time.time() - start_time
(out_dir / "forward-model.json").write_text(json.dumps(record, indent=2) + "\n")
print(f"wrote {out_dir / 'forward-model.json'} in {record['wall_seconds']:.0f} s")
