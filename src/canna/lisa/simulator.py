"""The prior over sources, their jaxgb A/E/T response in the window, and the noisy, whitened data."""

import math
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import log_ndtr
from jaxtyping import Array, Complex, Float, Int, Key

from .physics import (
    AMP_STD,
    SOURCE_COVARIANCE,
    SOURCE_MEAN,
    log10_amplitude_mean,
    log10_fdot_mean,
    sample_sky,
    sample_sources,
)
from .problem import (
    LisaGB,
    amplitude_unit,
    f0_window,
    window_centre,
    window_power,
    window_start,
)

# the log-uniform window centres sample_window resamples by the visible prior
VISIBLE_WINDOWS = 64


def source_segments(
    problem: LisaGB, p: Float[Array, "S 8"], f: Float[Array, ""]
) -> tuple[Int[Array, "S N"], Complex[Array, "S N 3"]]:
    """Each source's own `response_points` A, E, T bins and their indices in the window."""
    # jaxgb wants iota as colatitude
    p = p.at[..., 6].set(jnp.pi / 2 - p[..., 6])
    segments = jnp.stack(
        problem.response.get_tdi(p, tdi_generation=1.5, tdi_combination="AET"), axis=-1
    )
    start = problem.response.get_kmin(p[:, 0]) - window_start(problem, f)
    return start[:, None] + jnp.arange(problem.response_points), segments


def clean_signal(
    problem: LisaGB, p: Float[Array, "... S 8"], f: Float[Array, ""]
) -> Complex[Array, "... F 3"]:
    def combine(sources: Float[Array, "S 8"]) -> Complex[Array, "F 3"]:
        idxs, segments = source_segments(problem, sources, f)
        return (
            jnp.zeros((problem.window_bins, 3), dtype=segments.dtype)
            .at[idxs]
            .add(segments)
        )

    return jnp.vectorize(combine, signature="(s,d)->(f,c)")(p)


def snr(
    problem: LisaGB, p: Float[Array, "... S 8"], f: Float[Array, ""]
) -> Float[Array, "..."]:
    """Matched-filter snr, rho = sqrt(<h|h>) = sqrt(2 sum |h|^2 / power)."""
    integrand = (
        2.0 * jnp.abs(clean_signal(problem, p, f)) ** 2 / window_power(problem, f)
    )
    return jnp.sqrt(jnp.sum(integrand, axis=(-2, -1)))


def source_snrs(
    problem: LisaGB, p: Float[Array, "... S 8"], f: Float[Array, ""]
) -> Float[Array, "... S"]:
    """Each source's own matched-filter snr, as if it were alone in the window."""
    return snr(problem, p[..., :, None, :], f)


def sample_physical(
    problem: LisaGB,
    key: Key[Array, ""],
    f: Float[Array, ""],
) -> Float[Array, "S 8"]:
    """The window's sources from the prior, [f0, fdot, A, lon, lat, psi, iota, phi0], in no particular order.

    (f0, fdot, A) come from the catalogue's Gaussian with f0 inside the window, so most sources are faint.
    """
    keys = jr.split(key, 5)
    shape = (problem.n_sources, 1)
    sources = sample_sources(keys[0], *f0_window(problem, f), problem.n_sources)
    # a galactic sky, an isotropic orientation: psi uniform, sin iota uniform
    sky = jax.vmap(sample_sky)(jr.split(keys[1], problem.n_sources))
    psi = jr.uniform(keys[2], shape, maxval=2 * jnp.pi)
    iota = jnp.arcsin(jr.uniform(keys[3], shape, minval=-1.0, maxval=1.0))
    phi0 = jr.uniform(keys[4], shape, maxval=2 * jnp.pi)
    return jnp.concat([sources, sky, psi, iota, phi0], axis=-1)


def sample_window(problem: LisaGB, key: Key[Array, ""]) -> Float[Array, ""]:
    """The window's centre, by about the visible prior's f0, p(f0) P(snr >= visible_snr | f0); the sources stay the catalogue's.

    Log-uniform candidates are resampled by p(f0) over their density times P(snr >= visible_snr | f0, sky, orientation),
    one prior sky and orientation each, alone in the window centred on it.
    """
    key_f0, key_sky, key_psi, key_iota, key_pick = jr.split(key, 5)
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
    # one amplitude unit at the mean fdot: snr is linear in A and all but blind to fdot and phi0
    unit = amplitude_unit(problem, f)[:, None]
    p = jnp.concat(
        [f0, 10.0 ** log10_fdot_mean(f0), unit, sky, psi, iota, jnp.zeros(shape)],
        axis=-1,
    )
    unit_snr = jax.vmap(lambda p, f: snr(problem, p[None], f))(p, f)

    # where snr = visible_snr cuts log10 A given f0, in stds of its Gaussian
    log10_cut = jnp.log10(problem.visible_snr * unit[:, 0] / unit_snr)
    cut = (log10_cut - log10_amplitude_mean(f0[:, 0])) / AMP_STD
    log_p = -0.5 * (log_f0[:, 0] - SOURCE_MEAN[0]) ** 2 / SOURCE_COVARIANCE[0, 0]
    return f[jr.categorical(key_pick, log_p + log_ndtr(-cut))]


def sample_observation(
    problem: LisaGB,
    key: Key[Array, ""],
    p: Float[Array, "... S 8"],
    f: Float[Array, ""],
) -> Complex[Array, "... F 3"]:
    spectra = clean_signal(problem, p, f)
    real, imag = jr.normal(key, (2, *spectra.shape)) * jax.lax.rsqrt(2.0)
    return spectra + jnp.sqrt(window_power(problem, f)) * (real + 1j * imag)


def log_likelihood(
    problem: LisaGB,
    p: Float[Array, "... S 8"],
    o: Complex[Array, "... F 3"],
    f: Float[Array, ""],
) -> Float[Array, "..."]:
    residual = o - clean_signal(problem, p, f)
    return -jnp.sum(jnp.abs(residual) ** 2 / window_power(problem, f), axis=(-2, -1))


def preprocess(
    problem: LisaGB, o: Complex[Array, "... F 3"], f: Float[Array, ""]
) -> Float[Array, "... F 6"]:
    """The whitened window, unit noise variance per part: real and imaginary A, E, T."""
    o = o * jax.lax.rsqrt(window_power(problem, f) / 2.0)
    return jnp.concat([o.real, o.imag], axis=-1)
