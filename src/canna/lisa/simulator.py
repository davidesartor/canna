"""The prior over sources, their jaxgb A/E/T response in the window, and the noisy, whitened data."""

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Complex, Float, Int, Key

from .physics import fdot_from_chirp_mass, sample_chirp_mass, sample_sky
from .problem import LisaGB, f0_window, window_power, window_start


def source_segments(
    problem: LisaGB, p: Float[Array, "S 8"], f: Float[Array, ""]
) -> tuple[Int[Array, "S N"], Complex[Array, "S N 3"]]:
    """Each source's own `response_points` A, E, T bins and their indices in the window."""
    # jaxgb wants the chirp, not the chirp mass, and iota as colatitude
    p = p.at[..., 1].set(fdot_from_chirp_mass(p[..., 1], p[..., 0]))
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
    problem: LisaGB, key: Key[Array, ""], f: Float[Array, ""]
) -> Float[Array, "S 8"]:
    """The window's sources from the prior, [f0, Mc, A, lon, lat, psi, iota, phi0], loudest first.

    Each source's own snr is log-uniform over snr_range, its amplitude whatever gives it that snr.
    """
    keys = jr.split(key, 7)
    shape = (problem.n_sources, 1)
    log_f = jnp.log(jnp.stack(f0_window(problem, f)))

    f0 = jnp.exp(jr.uniform(keys[0], shape, minval=log_f[0], maxval=log_f[1]))
    mc = jax.vmap(partial(sample_chirp_mass, mass_range=problem.mass_range))(
        jr.split(keys[1], problem.n_sources)
    )
    log_snr = jnp.log(jnp.asarray(problem.snr_range))
    target_snr = jnp.exp(
        jr.uniform(keys[2], (problem.n_sources,), minval=log_snr[0], maxval=log_snr[1])
    )
    # a galactic sky, an isotropic orientation: psi uniform, sin iota uniform
    sky = jax.vmap(sample_sky)(jr.split(keys[3], problem.n_sources))
    psi = jr.uniform(keys[4], shape, maxval=2 * jnp.pi)
    iota = jnp.arcsin(jr.uniform(keys[5], shape, minval=-1.0, maxval=1.0))
    phi0 = jr.uniform(keys[6], shape, maxval=2 * jnp.pi)
    ones = jnp.ones(shape)
    unit = jnp.concat([f0, mc, ones, sky, psi, iota, phi0], axis=-1)
    # snr is linear in A: scale each unit-amplitude source to its drawn snr
    p = unit.at[:, 2].multiply(target_snr / source_snrs(problem, unit, f))
    return p[jnp.argsort(-target_snr)]


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
