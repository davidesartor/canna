"""Physical constants, the (f0, fdot, A) prior fitted to a WD-WD catalogue and the galactic sky prior."""

import math

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Float, Key

SPEED_OF_LIGHT = 299792458.0  # [m/s]
SUN_MASS = 1.98892e30  # [kg]
GRAVITATIONAL_CONSTANT = 6.67430e-11  # [m^3 kg^-1 s^-2]
EARTH_ORBIT_SPEED = 29785.0  # [m/s]
YEAR = 31557600.0  # LISA's orbital period [s]
# the antenna pattern's highest harmonic of the orbital frequency (Cutler 1998, arXiv:gr-qc/9703068)
ANTENNA_HARMONICS = 4

# (log10 f0 [Hz], log10 fdot [Hz/s], log10 A) of the double white dwarfs of a galaxy catalogue
# (wdwd_cat_lam_0p0_alpha_1p0_zerotides_MultiCGalaxy, f0 >= 1e-4 Hz): a Gaussian, fitted by maximum
# likelihood truncated at that cut. This is the untruncated parent; the f0 window truncates it again
SOURCE_MEAN = np.array([-3.742259, -20.57385, -23.267532])
SOURCE_COVARIANCE = np.array(
    [
        [0.151855, 0.528604, 0.073046],
        [0.528604, 1.872439, 0.286668],
        [0.073046, 0.286668, 0.097662],
    ]
)
# (log10 fdot, log10 A) given log10 f0: a linear mean, SOURCE_SLOPE per dex, and a fixed residual covariance
SOURCE_SLOPE = SOURCE_COVARIANCE[0, 1:] / SOURCE_COVARIANCE[0, 0]
SOURCE_INTERCEPT = SOURCE_MEAN[1:] - SOURCE_SLOPE * SOURCE_MEAN[0]
SOURCE_RESIDUAL = SOURCE_COVARIANCE[1:, 1:] - np.outer(SOURCE_SLOPE, SOURCE_COVARIANCE[0, 1:])
FDOT_STD = float(np.sqrt(SOURCE_RESIDUAL[0, 0]))

# galactic-binary sky, fitted to the sky of a WD-WD catalogue of this many sources: bulge, thick disk, thin disk, isotropic floor
SKY_CATALOGUE_SOURCES = 9_000_000
SKY_WEIGHTS = np.array([0.2515, 0.1559, 0.5848, 0.0078])
# (l, sin b) widths [rad]: the bulge's Gaussian stds, each disk's Laplace scales
BULGE_SCALES = (0.04165, 0.04385)
THICK_DISK_SCALES = (0.6466, 0.12718)
THIN_DISK_SCALES = (0.34931, 0.03199)
# rotates an equatorial (ICRS, jaxgb's) unit vector into galactic coordinates
EQUATORIAL_TO_GALACTIC = np.array(
    [
        [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
        [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
        [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669],
    ]
)


def log10_fdot_mean(f0: Float[Array, "..."]) -> Float[Array, "..."]:
    """The prior's mean log10 fdot [Hz/s] at f0 [Hz]."""
    return SOURCE_INTERCEPT[0] + SOURCE_SLOPE[0] * jnp.log10(f0)


def response_span(f0: float, fdot: float, t_obs: float) -> int:
    """Bins the response of a source at f0 covers: twice its doppler sideband, chirp drift and antenna-pattern sidebands."""
    drift = fdot * t_obs
    doppler = f0 * EARTH_ORBIT_SPEED / SPEED_OF_LIGHT
    return 2 * math.ceil((doppler + drift) * t_obs + ANTENNA_HARMONICS * t_obs / YEAR)


def sample_sources(
    key: Key[Array, ""], f_lo: Float[Array, ""], f_hi: Float[Array, ""], n: int
) -> Float[Array, "n 3"]:
    """n (f0, fdot, A) from the source prior, f0 truncated to [f_lo, f_hi]."""
    key_f, key_fdot, key_amp = jr.split(key, 3)
    mean, std = SOURCE_MEAN[0], np.sqrt(SOURCE_COVARIANCE[0, 0])
    bounds = (jnp.log10(jnp.stack([f_lo, f_hi])) - mean) / std
    log_f0 = mean + std * jr.truncated_normal(key_f, bounds[0], bounds[1], (n,))

    # fdot's residual, then A's given it: the conditional Gaussian
    residual_fdot = FDOT_STD * jr.normal(key_fdot, (n,))
    slope = SOURCE_RESIDUAL[0, 1] / SOURCE_RESIDUAL[0, 0]
    amp_std = np.sqrt(SOURCE_RESIDUAL[1, 1] - slope * SOURCE_RESIDUAL[0, 1])
    residual_amp = slope * residual_fdot + amp_std * jr.normal(key_amp, (n,))

    log_rest = SOURCE_INTERCEPT + SOURCE_SLOPE * log_f0[:, None]
    log_rest = log_rest + jnp.stack([residual_fdot, residual_amp], axis=-1)
    return 10.0 ** jnp.concat([log_f0[:, None], log_rest], axis=-1)


def source_log_density(log10_source: Float[Array, "... 3"]) -> Float[Array, "..."]:
    """log p(log10 f0, log10 fdot, log10 A) up to a constant: the parent Gaussian, which f0's truncation only rescales."""
    x = log10_source - SOURCE_MEAN
    return -0.5 * jnp.einsum("...i,ij,...j->...", x, np.linalg.inv(SOURCE_COVARIANCE), x)


def equatorial_to_galactic(
    lon: Float[Array, "..."], lat: Float[Array, "..."]
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Equatorial (RA, Dec) to galactic (l in (-pi, pi], b) [rad]."""
    v = jnp.stack([jnp.cos(lat) * jnp.cos(lon), jnp.cos(lat) * jnp.sin(lon), jnp.sin(lat)])
    x, y, z = jnp.tensordot(jnp.asarray(EQUATORIAL_TO_GALACTIC), v, axes=1)
    return jnp.arctan2(y, x), jnp.arctan2(z, jnp.hypot(x, y))


def galactic_to_equatorial(
    l: Float[Array, "..."], b: Float[Array, "..."]
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Galactic (l, b) to equatorial (RA in [0, 2 pi), Dec) [rad]; any real (l, b) is a direction."""
    v = jnp.stack([jnp.cos(b) * jnp.cos(l), jnp.cos(b) * jnp.sin(l), jnp.sin(b)])
    x, y, z = jnp.tensordot(jnp.asarray(EQUATORIAL_TO_GALACTIC).T, v, axes=1)
    return jnp.mod(jnp.arctan2(y, x), 2 * jnp.pi), jnp.arctan2(z, jnp.hypot(x, y))


def sample_sky(key: Key[Array, ""]) -> Float[Array, "2"]:
    """One direction from the galactic sky prior, as the equatorial (RA, Dec) jaxgb reads [rad]."""
    key_component, key_normal, key_uniform, key_sign = jr.split(key, 4)
    component = jr.choice(key_component, SKY_WEIGHTS.size, p=jnp.asarray(SKY_WEIGHTS))
    half_width = jnp.array([jnp.pi, 1.0])
    uniform = jr.uniform(key_uniform, (2,))

    # every component's (l, sin b): a Gaussian bulge, inverse-cdf Laplace disks cut at half_width, a uniform floor
    disk_scales = jnp.asarray([THICK_DISK_SCALES, THIN_DISK_SCALES])
    disks = -disk_scales * jnp.log1p(-uniform * (1 - jnp.exp(-half_width / disk_scales)))
    disks = disks * jr.rademacher(key_sign, (2,))
    candidates = jnp.stack(
        [
            jnp.asarray(BULGE_SCALES) * jr.normal(key_normal, (2,)),
            disks[0],
            disks[1],
            half_width * (2 * uniform - 1),
        ]
    )
    l, sin_b = candidates[component]
    return jnp.stack(galactic_to_equatorial(l, jnp.arcsin(jnp.clip(sin_b, -1, 1))))


def sky_log_density(
    lon: Float[Array, "..."], lat: Float[Array, "..."]
) -> Float[Array, "..."]:
    """log p(RA, Dec) w.r.t. dRA dDec: the prior per steradian, in galactic (l, sin b), times cos Dec."""
    l, b = equatorial_to_galactic(lon, lat)
    x = jnp.stack([l, jnp.sin(b)], axis=-1)
    half_width = jnp.array([jnp.pi, 1.0])
    bulge = jax.scipy.stats.norm.pdf(x, scale=jnp.asarray(BULGE_SCALES)).prod(-1)
    thick, thin = (
        (jnp.exp(-jnp.abs(x) / s) / (2 * s * (1 - jnp.exp(-half_width / s)))).prod(-1)
        for s in (jnp.asarray(THICK_DISK_SCALES), jnp.asarray(THIN_DISK_SCALES))
    )
    floor = jnp.full_like(l, 1 / (4 * jnp.pi))
    density = jnp.tensordot(
        jnp.asarray(SKY_WEIGHTS), jnp.stack([bulge, thick, thin, floor]), axes=1
    )
    return jnp.log(density) + jnp.log(jnp.cos(lat))


def galactic_scales() -> tuple[float, float]:
    """Std of galactic l and b under the sky prior, by 1-D quadrature: every component is separable in (l, sin b)."""
    scales = np.array([BULGE_SCALES, THICK_DISK_SCALES, THIN_DISK_SCALES])
    stds = []
    for axis, half_width in enumerate((np.pi, 1.0)):
        x = np.linspace(-half_width, half_width, 200001)
        value = x if axis == 0 else np.arcsin(x)
        bulge = np.exp(-0.5 * (x / scales[0, axis]) ** 2) / (
            np.sqrt(2 * np.pi) * scales[0, axis]
        )
        thick, thin = (
            np.exp(-np.abs(x) / s) / (2 * s * (1 - np.exp(-half_width / s)))
            for s in scales[1:, axis]
        )
        floor = np.full_like(x, 1 / (2 * half_width))
        density = SKY_WEIGHTS @ np.stack([bulge, thick, thin, floor])
        stds.append(float(np.sqrt(np.trapezoid(density * value**2, x))))
    return stds[0], stds[1]
