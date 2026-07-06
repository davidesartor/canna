import os
from typing import Literal, Callable
from jaxtyping import Array, Float, Scalar, Key
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from einops import rearrange

import lisaorbits
from jaxgb import jaxgb
from wdm_transform.transforms import from_freq_to_wdm

from . import inverse_cdfs, noise_utils

YEAR_s = 360 * 24 * 3600
MONTH_s = 28 * 24 * 3600
WEEK_s = 7 * 24 * 3600
DAY_s = 24 * 3600

MAX_FREQUENCY_Hz = 3e-3  # top of the LISA analysis band
SAMPLING_STEP_s = 1.0 / (2.0 * MAX_FREQUENCY_Hz)  # Nyquist sampling step (~167 s)
ARM_LENGTH_m = 2.5e9
SPEED_OF_LIGHT_m_s = 299792458.0

@eqx.filter_jit
def prior_inverse_cdf(u: Float[Array, "... 8"]) -> Float[Array, "... 8"]:
    """Inverse CDF of the Galactic Binary prior.

    Maps i.i.d. Uniform(0, 1) samples ``u`` to parameter samples drawn from the prior.
    The last axis indexes the 8 parameters
    ``[f0, fdot, A, ra, dec, psi, iota, phi0]``.

    Parameters
    ----------
    u : Float[Array, "... 8"]
        Uniform(0, 1) samples.

    Returns
    -------
    Float[Array, "... 8"]
        Parameter samples ``[f0, fdot, A, ra, dec, psi, iota, phi0]``.
    """
    u_f0, u_fdot, u_A, u_ra, u_dec, u_psi, u_iota, u_phi0 = jnp.split(u, 8, axis=-1)

    # Log-uniform parameters: exp(log(lo) + u * (log(hi) - log(lo)))
    f0 = inverse_cdfs.log_uniform(u_f0, range=(1e-4, 3e-3))
    fdot = inverse_cdfs.log_uniform(u_fdot, range=(1e-22, 4e-18))
    A = inverse_cdfs.log_uniform(u_A, range=(1e-25, 1.7e-23))

    # Uniform angles
    ra = inverse_cdfs.uniform(u_ra, range=(0.0, 2.0 * jnp.pi))
    psi = inverse_cdfs.uniform(u_psi, range=(0.0, jnp.pi))
    phi0 = inverse_cdfs.uniform(u_phi0, range=(-jnp.pi, jnp.pi))

    # Isotropic sky / orientation: dec ~ cos(dec) [-pi/2, pi/2] and iota ~ sin(iota) [0, pi]
    dec = inverse_cdfs.cosine_pdf(u_dec, range=(-jnp.pi / 2.0, jnp.pi / 2.0))
    iota = inverse_cdfs.cosine_pdf(u_iota, range=(0.0, jnp.pi))
    return jnp.concat([f0, fdot, A, ra, dec, psi, iota, phi0], axis=-1)


@eqx.filter_jit
def clean_signal(
    params: Float[Array, "S 8"],
    t_obs: float = YEAR_s,
    dt: float = SAMPLING_STEP_s,
) -> Float[Array, "T 3"]:
    """
    Compute the clean A/E/T TDI time-domain signal for a Galactic Binary.
    Parameters
    ----------
    params : array_like, shape (n_sources, 8)
        ``[f0 (Hz), fdot (Hz/s), A, ra (rad), dec (rad), psi (rad), iota (rad), phi0 (rad)]``
    t_obs : float
        Observation time in seconds (default: 1 year).
    dt : float
        Time step in seconds (default: Nyquist for 3 mHz band, ~167 s).

    Returns
    -------
    jnp.ndarray, shape (n_times, 3), complex128
        Time-domain signal for channels A (0), E (1), T (2).
    """
    n_samples = int(t_obs / dt)
    n_freqs = len(jnp.fft.rfftfreq(n_samples, dt))

    # logal segments of the TDI response
    orbit = lisaorbits.EqualArmlengthOrbits()
    jgb = jaxgb.JaxGB(orbit, t_obs=t_obs, t0=0.0, n=256)
    segments = jgb.get_tdi(params, tdi_generation=1.5, tdi_combination="AET")
    segments = jnp.stack(segments, axis=0).astype(jnp.complex128)  # shape (S, 3, n)

    # insert each segment into the correct place in the full frequency array
    start_idx = jgb.get_kmin(params[:, 0])  # shape (S,)
    idx = start_idx[:, None] + jnp.arange(256, dtype=jnp.int32)  # (S, 256)

    # Index-add into (3, n_freqs); duplicates sum coherently. Drop sentinel.
    full = jnp.zeros((3, n_freqs), dtype=jnp.complex128)
    full = full.at[:, idx].add(segments, mode="drop")

    # Inverse FFT to get time-domain signal
    signal = jnp.fft.irfft(full, n=n_samples, axis=-1)
    return signal.T  # (n_samples, 3)


def noise_psd(
    channel: Literal["A", "E", "T"],
    A: float = 3.0,
    P: float = 15.0,
    L: float = ARM_LENGTH_m,
) -> Callable[[Float[Array, "..."]], Float[Array, "..."]]:
    # TODO: documentation for these parameters/formulas
    @eqx.filter_jit
    def psd_f(f: Float[Array, "..."]) -> Float[Array, "..."]:
        fstar = 1.0 / (2.0 * jnp.pi * L / SPEED_OF_LIGHT_m_s)
        tdi15_factor = 4.0 * jnp.sin(f / fstar) * f / fstar
        if channel in "AE":
            n_tilda = (
                0.5
                * (2.0 + jnp.cos(f / fstar))
                * (P / L) ** 2
                * 1e-24
                * (1.0 + (0.002 / f) ** 4)
                + 2.0
                * (1.0 + jnp.cos(f / fstar) + jnp.cos(f / fstar) ** 2)
                * (A / L) ** 2
                * 1e-30
                * (1.0 + (0.0004 / f) ** 2)
                * (1.0 + (f / 0.008) ** 4)
                * (1.0 / (2.0 * jnp.pi * f)) ** 4
            )
        elif channel == "T":
            n_tilda = (
                1e-24
                * (1.0 - jnp.cos(f / fstar))
                * (P / L) ** 2
                * (1.0 + (0.002 / f) ** 4)
                + 2.0
                * (1.0 - jnp.cos(f / fstar)) ** 2
                * (A / L) ** 2
                * 1e-30
                * (1.0 + (0.0004 / f) ** 2)
                * (1.0 + (f / 0.008) ** 4)
                * (1.0 / (2.0 * jnp.pi * f)) ** 4
            )
        else:
            raise ValueError(f"Invalid channel: {channel}. Must be 'A', 'E', or 'T'.")

        return tdi15_factor * n_tilda

    return psd_f


@eqx.filter_jit
def sample_noise(
    key: Key,
    t_obs: float = YEAR_s,
    dt: float = SAMPLING_STEP_s,
) -> Float[Array, "T 3"]:
    """
    Draw a time-domain instrumental noise realization for A, E, T channels.

    Generates colored Gaussian noise whose rFFT power spectral density matches
    the TDI 1.5 instrumental PSD (no galactic foreground).

    Parameters
    ----------
    key : jnp.random.Key
        Random key for generating noise.
    t_obs : float
        Observation time in seconds (default: 1 year).
    dt : float
        Time step in seconds (default: ~167 s).

    Returns
    -------
    jnp.ndarray, shape (n_times, 3)
        Time-domain noise for channels A (0), E (1), T (2).
    """
    key_a, key_e, key_t = jr.split(key, 3)
    noise_a = noise_utils.sample_noise(key_a, t_obs, dt, psd_function=noise_psd("A"))
    noise_e = noise_utils.sample_noise(key_e, t_obs, dt, psd_function=noise_psd("E"))
    noise_t = noise_utils.sample_noise(key_t, t_obs, dt, psd_function=noise_psd("T"))
    noise = jnp.stack([noise_a, noise_e, noise_t], axis=-1)
    return noise


@eqx.filter_jit
def preprocess_datastream(
    datastream: Float[Array, "T 3"],
    t_obs: float = YEAR_s,
    dt: float = SAMPLING_STEP_s,
) -> Float[Array, "F T*C"]:
    # convert to freq domain and crop the frequency axis to a multiple of 32 for the WDM grid
    # TODO: make the 32 time bins configurable
    freqs = jnp.fft.rfftfreq(int(t_obs / dt), dt)
    freqs = freqs[: (len(freqs) // 32) * 32]
    datastream = jnp.fft.rfft(datastream, axis=0)
    datastream = datastream[: len(freqs)]

    # whiten each channel of the datastream by its nominal PSD
    for i, ch in enumerate("AET"):
        psd = jnp.where(freqs > 0, noise_psd(ch)(freqs), 0.0)  # type: ignore
        datastream = datastream.at[:, i].divide(
            jnp.where(psd == 0.0, 1.0, jnp.sqrt(psd))
        )

    # process the data to make it more digestible
    # TODO: make the 32 time bins configurable
    y = from_freq_to_wdm(
        datastream.T,
        nt=32,
        nf=len(datastream) // 32,
        a=1.0 / 3.0,
        d=1.0,
        dt=SAMPLING_STEP_s,
        backend="jax",
    )
    y = rearrange(y, "c t f -> t (f c)")
    y = jnp.arcsinh(y)  # avoids grossly large values in the WDM transform
    return y


@eqx.filter_jit
def geodesic(
    t: Scalar, u0: Float[Array, "8"], u1: Float[Array, "8"]
) -> Float[Array, "8"]:
    """Probability path point at time ``t`` between base ``u0`` and target ``u1``."""
    # TODO make it depend on geometry
    return u0 + t * (u1 - u0)


@eqx.filter_jit
def get_train_batch(
    key: Key,
    batch_size: int,
    n_sources: int,
    t_obs: float = YEAR_s,
    dt: float = SAMPLING_STEP_s,
    noise_scale: float = 1.0,
) -> tuple[Float[Array, "S 8"], Float[Array, "S 8"], Scalar, Float[Array, "T F*C"]]:
    def train_sample(
        key: Key,
    ) -> tuple[Float[Array, "S 8"], Float[Array, "S 8"], Scalar, Float[Array, "T F*C"]]:
        key_u1, key_u0, key_t, key_y = jr.split(key, 4)
        u1 = jr.uniform(key_u1, shape=(n_sources, 8))
        u0 = jr.uniform(key_u0, shape=(n_sources, 8))
        t = jr.uniform(key_t, minval=0.0, maxval=1.0)

        # generate the conditioning signal
        params = prior_inverse_cdf(u1)
        signal = clean_signal(params, t_obs=t_obs, dt=dt)
        noise = sample_noise(key_y, t_obs=t_obs, dt=dt)
        datastream = signal + noise_scale * noise
        y = preprocess_datastream(datastream, t_obs=t_obs, dt=dt)

        # conditional flow-matching target
        xt = geodesic(t, u0, u1)
        dx = jax.jacobian(geodesic)(t, u0, u1)
        return xt, dx, t, y

    return jax.vmap(train_sample)(jr.split(key, batch_size))
