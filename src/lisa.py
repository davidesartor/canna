import itertools
from typing import Literal, Callable
from jaxtyping import Array, Float, Scalar, Key
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import einops

import lisaorbits
from jaxgb import jaxgb
from wdm_transform.transforms import from_freq_to_wdm

from . import inverse_cdfs, noise_utils

YEAR = 360 * 24 * 3600  # [s]
MONTH = 28 * 24 * 3600  # [s]
WEEK = 7 * 24 * 3600  # [s]
DAY = 24 * 3600  # [s]

MAX_FREQUENCY = 3e-3  # [Hz] top of the LISA analysis band
SAMPLING_STEP = 1.0 / (2.0 * MAX_FREQUENCY)  # [s] Nyquist sampling step (~167 s)
ARM_LENGTH = 2.5e9  # [m]
SPEED_OF_LIGHT = 299792458.0  # [m/s]
GRAVITATIONAL_CONSTANT = 6.67430e-11  # [m^3 kg^-1 s^-2]
SUN_MASS = 1.98892e30  # [kg]


PARAMETER_NAMES = ["f0", "fdot", "A", "ra", "dec", "psi", "iota", "phi0"]
PERIODIC = jnp.array([False, False, False, True, False, True, False, True])


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

    # f0: log-uniform over the analysis band. arXiv:2606.29039
    f0 = inverse_cdfs.log_uniform(u_f0, range=(1e-4, MAX_FREQUENCY))

    # A: log-uniform over the detectable strain range. arXiv:2402.13701, arXiv:2606.29039
    A = inverse_cdfs.log_uniform(u_A, range=(1e-24, 1e-22))

    # fdot: uniform on [0, fdot_max(f0)], with fdot_max (Mc = mc_max_msun) scaling as
    # f0^(11/3) from GW radiation reaction. arXiv:2402.13701, arXiv:2606.29039
    fdot_chirp_coeff = (
        (96.0 / 5.0)
        * jnp.pi ** (8.0 / 3.0)
        * (GRAVITATIONAL_CONSTANT * SUN_MASS / SPEED_OF_LIGHT**3) ** (5.0 / 3.0)
    )  # ~5.8e-7
    mc_max_msun = 1.0  # heaviest chirp mass
    fdot_max = fdot_chirp_coeff * mc_max_msun ** (5.0 / 3.0) * f0 ** (11.0 / 3.0)
    fdot = inverse_cdfs.uniform(u_fdot, range=(0.0, fdot_max))

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
    t_obs: float = YEAR,
    dt: float = SAMPLING_STEP,
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


@eqx.filter_jit
def optimal_snr(
    params: Float[Array, "S 8"],
    t_obs: float = YEAR,
    dt: float = SAMPLING_STEP,
) -> Scalar:
    """Physical matched-filter optimal SNR of a GB signal against the LISA PSD.
    Uses the standard one-sided convention
    ``SNR^2 = 4 * df * sum_{f>0} |h~(f)|^2 / S(f)`` with the continuous FT
    ``h~(f) = dt * rfft(h)`` and ``df = 1/(n_samples*dt)``, summed over A/E/T.
    ----------
    params : array_like, shape (n_sources, 8)
        ``[f0 (Hz), fdot (Hz/s), A, ra (rad), dec (rad), psi (rad), iota (rad), phi0 (rad)]`
    t_obs : float
        Observation time in seconds (default: 1 year).
    dt : float
        Time step in seconds (default: Nyquist for 3 mHz band, ~167 s).
    """
    n_samples = int(t_obs / dt)
    freqs = jnp.fft.rfftfreq(n_samples, dt)
    h = jnp.fft.rfft(clean_signal(params, t_obs=t_obs, dt=dt), axis=0)  # (F, 3)
    prefactor = 4.0 * dt / n_samples  # = 4 * df * dt**2
    snr2 = 0.0
    for i, ch in enumerate("AET"):
        psd = jnp.where(freqs > 0, noise_psd(ch)(freqs), jnp.inf)  # type: ignore
        snr2 = snr2 + jnp.sum(jnp.where(freqs > 0, jnp.abs(h[:, i]) ** 2 / psd, 0.0))
    return jnp.sqrt(prefactor * snr2)


def noise_psd(
    channel: Literal["A", "E", "T"],
    A: float = 3.0,
    P: float = 15.0,
    L: float = ARM_LENGTH,
) -> Callable[[Float[Array, "..."]], Float[Array, "..."]]:
    # TODO: documentation for these parameters/formulas
    @eqx.filter_jit
    def psd_f(f: Float[Array, "..."]) -> Float[Array, "..."]:
        fstar = 1.0 / (2.0 * jnp.pi * L / SPEED_OF_LIGHT)
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
    t_obs: float = YEAR,
    dt: float = SAMPLING_STEP,
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
    t_obs: float = YEAR,
    dt: float = SAMPLING_STEP,
) -> Float[Array, "F T*C"]:
    # convert to freq domain and crop the frequency axis to a multiple of 32 for the WDM grid
    # TODO: make the 32 time bins configurable
    n_samples = int(t_obs / dt)
    freqs = jnp.fft.rfftfreq(n_samples, dt)
    freqs = freqs[: (len(freqs) // 32) * 32]
    datastream = jnp.fft.rfft(datastream, axis=0)
    datastream = datastream[: len(freqs)]

    # whiten each channel to unit variance per rfft bin
    for i, ch in enumerate("AET"):
        psd = jnp.where(freqs > 0, noise_psd(ch)(freqs), 0.0)  # type: ignore
        var = psd * n_samples / (2.0 * dt)  # E[|rfft(noise)|^2] per bin
        datastream = datastream.at[:, i].divide(
            jnp.where(var == 0.0, 1.0, jnp.sqrt(var))
        )

    # process the data to make it more digestible
    # TODO: make the 32 time bins configurable
    y = from_freq_to_wdm(
        datastream.T,
        nt=32,
        nf=len(datastream) // 32,
        a=1.0 / 3.0,
        d=1.0,
        dt=SAMPLING_STEP,
        backend="jax",
    )
    y = einops.rearrange(y, "c t f -> t (f c)")
    y = jnp.arcsinh(y)  # avoids grossly large values in the WDM transform
    return y


@eqx.filter_jit
def log_map(x0: Float[Array, "8"], x1: Float[Array, "8"]) -> Float[Array, "8"]:
    """Tangent vector at ``x0`` pointing to ``x1`` (the flow-matching velocity)."""
    d = x1 - x0
    return jnp.where(PERIODIC, d - 2.0 * jnp.round(d / 2.0), d)


@eqx.filter_jit
def exp_map(x0: Float[Array, "8"], v: Float[Array, "8"]) -> Float[Array, "8"]:
    """Move from ``x0`` along tangent ``v``."""
    x = x0 + v
    return jnp.where(PERIODIC, ((x + 1.0) % 2.0) - 1.0, x)


@eqx.filter_jit
def geodesic(
    t: Scalar, x0: Float[Array, "8"], x1: Float[Array, "8"]
) -> Float[Array, "8"]:
    """Probability path point at time ``t`` between base ``x0`` and target ``x1``."""
    return exp_map(x0, t * log_map(x0, x1))


@eqx.filter_jit
def match_sources(
    x0: Float[Array, "S 8"], x1: Float[Array, "S 8"]
) -> tuple[Float[Array, "S 8"], Scalar]:
    """Reorder ``x0`` rows to minimize total pairwise cost against ``x1`` (rows align 1:1).
    returns the reordered ``x0`` and the total cost. Brute-forces all permutations, since ``n_sources`` is small.
    TODO: replace with an actual assignment algorithm (e.g. Hungarian) to support larger ``n_sources``.
    """
    n = x0.shape[0]

    def cost(x0: Float[Array, "8"], x1: Float[Array, "8"]) -> Scalar:
        return jnp.sum(log_map(x0, x1) ** 2)

    cost = jax.vmap(cost, in_axes=(None, 0))  # vectorize over x1
    cost = jax.vmap(cost, in_axes=(0, None))  # vectorize over x0
    C = cost(x0, x1)

    if n <= 4:
        perms = jnp.array(list(itertools.permutations(range(n))))  # (P, S)
        costs = jnp.sum(C[jnp.arange(n)[None, :], perms], axis=-1)  # (P,)
        sigma = perms[jnp.argmin(costs)]
        return x0[jnp.argsort(sigma)], jnp.min(costs)

    raise NotImplementedError(
        f"match_sources only brute-forces permutations for n_sources<=4, got {n}. "
        "TODO: implement an assignment algorithm (e.g. Hungarian) for larger n_sources."
    )


@eqx.filter_jit
def get_train_batch(
    key: Key,
    batch_size: int,
    n_sources: int,
    t_obs: float = YEAR,
    dt: float = SAMPLING_STEP,
    noise_scale: float = 1.0,
) -> tuple[
    Float[Array, "S 8"],
    Float[Array, "S 8"],
    Scalar,
    Float[Array, "T F*C"],
    Float[Array, "S 8"],
    Float[Array, "S 8"],
    Float[Array, "S 8"],
    Float[Array, "T 3"],
]:
    def train_sample(
        key: Key,
    ) -> tuple[
        Float[Array, "S 8"],
        Float[Array, "S 8"],
        Scalar,
        Float[Array, "T F*C"],
        Float[Array, "S 8"],
        Float[Array, "S 8"],
        Float[Array, "S 8"],
        Float[Array, "T 3"],
    ]:
        key_x1, key_x0, key_t, key_y = jr.split(key, 4)
        x1 = jr.uniform(key_x1, shape=(n_sources, 8), minval=-1.0, maxval=1.0)
        x0 = jr.uniform(key_x0, shape=(n_sources, 8), minval=-1.0, maxval=1.0)
        x0, _ = match_sources(x0, x1)  # align sources to minimize transport cost
        t = jr.uniform(key_t, minval=0.0, maxval=1.0)

        # generate the conditioning signal
        u1 = (x1 + 1.0) / 2.0  # U(-1, 1) -> U(0,1)
        params = prior_inverse_cdf(u1)
        signal = clean_signal(params, t_obs=t_obs, dt=dt)
        noise = sample_noise(key_y, t_obs=t_obs, dt=dt)
        datastream = signal + noise_scale * noise
        y = preprocess_datastream(datastream, t_obs=t_obs, dt=dt)

        # conditional flow-matching target
        xt = geodesic(t, x0, x1)
        dx = jax.jacobian(geodesic)(t, x0, x1)
        return xt, dx, t, y, x0, x1, params, datastream

    return jax.vmap(train_sample)(jr.split(key, batch_size))
