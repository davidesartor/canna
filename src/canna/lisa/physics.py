"""Physical constants, the chirp of a circular binary, the Korol et al. 2022 chirp-mass prior and the galactic sky prior."""

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

# double-white-dwarf primary of Korol et al. 2022 (arXiv:2109.10972, Kepler 2015): a 3-Gaussian mixture
PRIMARY_WEIGHTS = np.array([0.81, 0.14, 0.05])
PRIMARY_MEANS = np.array([0.65, 0.57, 0.81])
PRIMARY_STDS = np.array([0.044, 0.097, 0.187])
# points per axis of chirp_mass_moments' quadrature grid over (m1, m2)
CHIRP_MASS_GRID = 256

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


def fdot_from_chirp_mass(
    chirp_mass: Float[Array, "..."], f0: Float[Array, "..."]
) -> Float[Array, "..."]:
    """Radiation-reaction chirp of a circular binary, for chirp mass in solar masses."""
    ratio = 96.0 / 5.0 * jnp.pi ** (8 / 3) * f0 ** (11 / 3)
    return ratio * (
        GRAVITATIONAL_CONSTANT * SUN_MASS / SPEED_OF_LIGHT**3 * chirp_mass
    ) ** (5 / 3)


def response_span(f0: float, chirp_mass: float, t_obs: float) -> int:
    """Bins the response of a source at f0 covers: twice its doppler sideband, chirp drift and antenna-pattern sidebands."""
    drift = float(fdot_from_chirp_mass(chirp_mass, f0)) * t_obs
    doppler = f0 * EARTH_ORBIT_SPEED / SPEED_OF_LIGHT
    return 2 * math.ceil((doppler + drift) * t_obs + ANTENNA_HARMONICS * t_obs / YEAR)


def sample_chirp_mass(
    key: Key[Array, ""], mass_range: tuple[float, float]
) -> Float[Array, "1"]:
    """One chirp mass [Msun]: a truncated-mixture primary m1 and m2 ~ U[m_min, m1]."""
    m_min, m_max = mass_range
    means, stds = jnp.asarray(PRIMARY_MEANS), jnp.asarray(PRIMARY_STDS)
    key_component, key_primary, key_secondary = jr.split(key, 3)

    # truncation leaves each component a different share of the mixture
    low, high = (m_min - means) / stds, (m_max - means) / stds
    kept = jax.scipy.stats.norm.cdf(high) - jax.scipy.stats.norm.cdf(low)
    k = jr.choice(key_component, means.size, p=jnp.asarray(PRIMARY_WEIGHTS) * kept)

    m1 = means[k] + stds[k] * jr.truncated_normal(key_primary, low[k], high[k])
    m2 = jr.uniform(key_secondary, minval=m_min, maxval=m1)
    return jnp.atleast_1d((m1 * m2) ** 0.6 / (m1 + m2) ** 0.2)


def chirp_mass_moments(
    mass_range: tuple[float, float],
) -> tuple[Float[Array, "1"], Float[Array, "1"]]:
    """Mean and std of log Mc, on a grid over the truncated primary and the flat mass ratio."""
    m_min, m_max = mass_range
    m1 = jnp.linspace(m_min, m_max, CHIRP_MASS_GRID)[:, None]
    ratio = jnp.linspace(0.0, 1.0, CHIRP_MASS_GRID)
    m2 = m_min + ratio * (m1 - m_min)
    log_mc = 0.6 * jnp.log(m1 * m2) - 0.2 * jnp.log(m1 + m2)

    means, stds, weights = (
        jnp.asarray(a) for a in (PRIMARY_MEANS, PRIMARY_STDS, PRIMARY_WEIGHTS)
    )
    weight = (jax.scipy.stats.norm.pdf(m1[..., None], means, stds) * weights).sum(-1)
    weight = weight / weight.sum() / ratio.size
    mean = jnp.atleast_1d((weight * log_mc).sum())
    return mean, jnp.atleast_1d(jnp.sqrt((weight * (log_mc - mean) ** 2).sum()))


def chirp_mass_support(mass_range: tuple[float, float]) -> tuple[float, float]:
    # both extremes are equal-mass pairs, where the chirp mass is m / 2^(1/5)
    return mass_range[0] * 2**-0.2, mass_range[1] * 2**-0.2


def chirp_mass_log_density(
    mass_range: tuple[float, float], n_grid: int = 1024, n_m1: int = 8000
) -> tuple[np.ndarray, np.ndarray]:
    """log p(log Mc) on a grid, by quadrature over m1: p(Mc) = int dm1 p(m1) / (m1 - m_min) / (dMc/dm2)."""
    from scipy.special import ndtr

    m_min, m_max = mass_range
    w, mu, sd = PRIMARY_WEIGHTS, PRIMARY_MEANS, PRIMARY_STDS
    norm = lambda z: np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    kept = ndtr((m_max - mu) / sd) - ndtr((m_min - mu) / sd)
    m1 = np.linspace(m_min, m_max, n_m1 + 1)[1:]
    p1 = (w * norm((m1[:, None] - mu) / sd) / sd).sum(-1) / (w * kept).sum()
    chirp = lambda a, b: (a * b) ** 0.6 / (a + b) ** 0.2

    lo, hi = chirp_mass_support(mass_range)
    log_mc = np.linspace(np.log(lo), np.log(hi), n_grid)
    mc = np.exp(log_mc)[:, None]
    low, high = chirp(m1, m_min)[None], chirp(m1, m1)[None]
    valid = (mc >= low) & (mc <= high)
    a, b = np.full_like(mc * m1, m_min), np.broadcast_to(m1, (n_grid, n_m1)).copy()
    for _ in range(60):  # bisection for m2 on [m_min, m1]
        mid = 0.5 * (a + b)
        below = chirp(m1[None], mid) < mc
        a, b = np.where(below, mid, a), np.where(below, b, mid)
    m2 = 0.5 * (a + b)
    slope = chirp(m1[None], m2) * (0.6 / m2 - 0.2 / (m1[None] + m2))
    integrand = np.where(valid, p1[None] / (m1[None] - m_min) / slope, 0.0)
    density = np.trapezoid(integrand, m1, axis=1)  # p(Mc)
    with np.errstate(divide="ignore"):
        return log_mc, np.log(density * np.exp(log_mc))  # p(log Mc) = p(Mc) Mc


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
