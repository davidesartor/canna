from typing import Callable, Optional
from jaxtyping import Array, Float, Complex, Key
import jax
import jax.numpy as jnp
import jax.random as jr


def sample_noise(
    key: Key,
    t_obs: float,
    dt: float,
    *,
    psd_function: Optional[Callable[[Float[Array, "F"]], Float[Array, "F"]]] = None,
) -> Float[Array, "T"]:
    """
    Draw a time domain realization of instrumental noise with a given PSD.

    Parameters
    ----------
    key : jnp.random.Key
        Random key for generating noise.
    t_obs : float
        Observation time in seconds (default: 1 year).
    dt : float
        Time step in seconds (default: ~167 s).
    psd_function : callable, optional
        power spectral density function. If None, uses uniform white noise (psd=1).
    Returns
    -------
    jnp.ndarray, shape (n_times,)
        Time-domain noise with the given PSD. If ``psd_function`` is None, returns white noise.
    """

    n_times = int(t_obs / dt)
    f = jnp.fft.rfftfreq(n_times, dt)
    real, imag = jr.normal(key, (2, len(f)))
    noise_f = real + 1j * imag
    if psd_function is not None:
        psd = jnp.where(f > 0, psd_function(f), 0.0)
        noise_f = jnp.sqrt(psd * n_times / (4.0 * dt)) * noise_f
    noise_t = jnp.fft.irfft(noise_f, n=n_times)
    return noise_t