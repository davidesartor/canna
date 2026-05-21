import jax
import jax.numpy as jnp
import lisaorbits
from jaxgb.jaxgb import JaxGB

# ── Physical defaults ─────────────────────────────────────────────────────────

_YR = 365 * 24 * 3600       # 1 year in seconds
_FMAX = 3e-3                 # Hz — top of the LISA analysis band
_DT = 1.0 / (2.0 * _FMAX)   # Nyquist sampling step (~167 s)
_L_LISA = 2.5e9              # m  — LISA arm length
_c = 299792458.0             # m/s


# ── Instrumental noise PSD (TDI 1.5, instrumental only) ──────────────────────
# Reproduced from lisa_common.py; avoids importing noise.py which has
# module-level plotting code and a JAX-only import.

def _ntilda_e(f, A: float = 3.0, P: float = 15.0, L: float = _L_LISA):
    fstar = 1.0 / (2.0 * jnp.pi * L / _c)
    return (
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


def _ntilda_t(f, A: float = 3.0, P: float = 15.0, L: float = _L_LISA):
    fstar = 1.0 / (2.0 * jnp.pi * L / _c)
    return (
        (1.0 - jnp.cos(f / fstar)) * (P / L) ** 2 * 1e-24 * (1.0 + (0.002 / f) ** 4)
        + 2.0
        * (1.0 - jnp.cos(f / fstar)) ** 2
        * (A / L) ** 2
        * 1e-30
        * (1.0 + (0.0004 / f) ** 2)
        * (1.0 + (f / 0.008) ** 4)
        * (1.0 / (2.0 * jnp.pi * f)) ** 4
    )


def _tdi15_factor(f, L: float = _L_LISA):
    fstar = 1.0 / (2.0 * jnp.pi * L / _c)
    return 4.0 * jnp.sin(f / fstar) * f / fstar


def _noise_psd(channel: int, freqs: jnp.ndarray) -> jnp.ndarray:
    """Instrumental TDI 1.5 PSD.  channel: 0=A, 1=E, 2=T."""
    return jnp.where(freqs>0,_tdi15_factor(freqs)*(_ntilda_t(freqs) if channel == 2 else _ntilda_e(freqs)),freqs*0)


def sample_params(rng=None) -> jnp.ndarray:
    """Sample a Galactic Binary parameter vector.

    Uses ``gb_prior.draw_source_prior_and_params`` from the repository when
    available; otherwise falls back to uniform (angles) and log-uniform
    (positive scalars) priors over a physically reasonable range.

    Parameters
    ----------
    rng : jnp.random.Generator, optional
    seed : int, optional
        Ignored when *rng* is provided.

    Returns
    -------
    jnp.ndarray, shape (8,)
        ``[f0, fdot, A, ra, dec, psi, iota, phi0]``
    """
    if rng is None:
        import numpy as np
        rng = np.random.default_rng(42)

    # Fallback priors matching gb_prior bounds
    f0 = jnp.exp(rng.uniform(jnp.log(1e-4), jnp.log(3e-3)))
    fdot = jnp.exp(rng.uniform(jnp.log(5e-19), jnp.log(4e-18)))
    A = jnp.exp(rng.uniform(jnp.log(6e-24), jnp.log(1.7e-23)))
    ra = rng.uniform(0.0, 2.0 * jnp.pi)
    dec = jnp.arcsin(rng.uniform(-1.0, 1.0))
    psi = rng.uniform(0.0, jnp.pi)
    iota = jnp.arccos(rng.uniform(-1.0, 1.0))
    phi0 = rng.uniform(-jnp.pi, jnp.pi)
    return jnp.array([f0, fdot, A, ra, dec, psi, iota, phi0])


def clean_signal(
    params,
    t_obs: float = _YR,
    dt: float = _DT,
) -> jnp.ndarray:
    
    """
    Compute the clean A/E/T TDI frequency-domain signal for a Galactic Binary.

    Wraps JaxGB (``jaxgb.jaxgb.JaxGB``) and places the band-limited output
    into a zero-padded full-length rFFT array, following the same convention
    as ``data_generation.py``.

    Parameters
    ----------
    params : array_like, shape (8,)
        ``[f0 (Hz), fdot (Hz/s), A, ra (rad), dec (rad), psi (rad), iota (rad), phi0 (rad)]``
    t_obs : float
        Observation time in seconds (default: 1 year).
    dt : float
        Time step in seconds (default: Nyquist for 3 mHz band, ~167 s).

    Returns
    -------
    jnp.ndarray, shape (3, n_freqs), complex128
        rFFT coefficients for channels A (0), E (1), T (2) on the full
        frequency grid ``jnp.fft.rfftfreq(int(t_obs/dt), dt)``.
    """
    

    jax.config.update("jax_enable_x64", True)

    n_samples = int(t_obs / dt)
    n_freqs = n_samples // 2 + 1

    orbit = lisaorbits.EqualArmlengthOrbits()
    jgb = JaxGB(orbit, t_obs=t_obs, t0=0.0, n=256)

    params_j = jnp.asarray(params, dtype=jnp.float64)
    
    a_loc, e_loc, t_loc = jgb.get_tdi(
        params_j, tdi_generation=1.5, tdi_combination="AET"
    )
    kmin = int(jnp.asarray(jgb.get_kmin(params_j[None, 0:1])).reshape(-1)[0])

    def place_local_tdi(segment) -> jnp.ndarray:
        """Place a band-limited TDI segment into a zero-padded full-length frequency array."""
        full = jnp.zeros(n_freqs, dtype=jnp.complex128)
        seg = jnp.asarray(segment, dtype=jnp.complex128).reshape(-1)
        end = min(kmin + seg.size, n_freqs)
        if end > kmin:
            full = full.at[kmin:end].set(seg[: end - kmin])
        return full

    return jnp.stack([place_local_tdi(a_loc), place_local_tdi(e_loc), place_local_tdi(t_loc)])



def sample_noise(
    t_obs: float = _YR,
    dt: float = _DT,
    seed=None,
    rng=None,
) -> jnp.ndarray:
    
    """
    Draw a time-domain instrumental noise realization for A, E, T channels.

    Generates colored Gaussian noise whose rFFT power spectral density matches
    the TDI 1.5 instrumental PSD (no galactic foreground), using the same
    prescription as ``data_generation.py``:

        noise_f = sqrt(psd) * (z_r + i z_i) / sqrt(2),   z ~ N(0,1)
        noise_t = irfft(noise_f, n=n_samples)

    Parameters
    ----------
    t_obs : float
        Observation time in seconds (default: 1 year).
    dt : float
        Time step in seconds (default: ~167 s).
    seed : int, optional
        RNG seed (ignored when *rng* is given).
    rng : jnp.random.Generator, optional

    Returns
    -------
    jnp.ndarray, shape (3, n_samples)
        Time-domain noise for channels A (0), E (1), T (2).
    """
    
    if rng is None:
        import numpy as np
        rng = np.random.default_rng(seed)

    n_samples = int(t_obs / dt)
    freqs = jnp.fft.rfftfreq(n_samples, dt)
    n_freqs = len(freqs)

    noise_t = jnp.empty((3, n_samples))
    for ch in range(3):
        psd = _noise_psd(ch, freqs)
        white = rng.standard_normal(n_freqs) + 1j * rng.standard_normal(n_freqs)
        noise_f = jnp.sqrt(psd) * white / jnp.sqrt(2.0)
        noise_t = noise_t.at[ch].set(jnp.fft.irfft(noise_f, n=n_samples))

    return noise_t