import itertools
import os
from jaxtyping import Array, Bool, Float, Scalar, Key
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from einops import rearrange

import lisaorbits
from jaxgb import jaxgb
from wdm_transform.transforms import from_freq_to_wdm

from . import inverse_cdfs

YEAR = 365 * 24 * 3600  # [s]
MONTH = 30 * 24 * 3600  # [s]
WEEK = 7 * 24 * 3600  # [s]
DAY = 24 * 3600  # [s]

ARM_LENGTH = 2.5e9  # [m]
SPEED_OF_LIGHT = 299792458.0  # [m/s]
GRAVITATIONAL_CONSTANT = 6.67430e-11  # [m^3 kg^-1 s^-2]
SUN_MASS = 1.98892e30  # [kg]
MAX_FREQUENCY = 12.0e-3  # [Hz] top of the LISA analysis band
SAMPLING_STEP = 1.0 / (2.0 * MAX_FREQUENCY)  # [s] Nyquist sampling step (~42 s)

CHANNEL_NAMES = ["A", "E", "T"]
PARAMETER_NAMES = ["f0", "fdot", "A", "ra", "dec", "psi", "iota", "phi0"]
PERIODIC = jnp.array([False, False, False, True, False, True, False, True])

N_SOURCES = int(os.environ.get("N_SOURCES", 1))
T_OBS = YEAR

SIMPLIFIED_PROBLEM = os.environ.get("SIMPLIFIED_PROBLEM", "0")
SIMPLIFIED_PROBLEM = SIMPLIFIED_PROBLEM not in ("0", "", "false", "False")
if SIMPLIFIED_PROBLEM:
    MASK = jnp.array([0, 1, 2, 5])  # f0, fdot, A, psi
    PARAMETER_NAMES = ["f0", "fdot", "A", "psi"]
    PERIODIC: Bool[Array, "P"] = jnp.array([False, False, False, True])
    T_OBS = MONTH

# fixed sampling grid: round n_samples down to a power of two, then re-derive T_OBS
N_SAMPLES = 1 << int(T_OBS / SAMPLING_STEP).bit_length()
T_OBS = N_SAMPLES * SAMPLING_STEP


@eqx.filter_jit
def logarithmic_map(
    u0: Float[Array, "... P"], u1: Float[Array, "... P"]
) -> Float[Array, "... P"]:
    """Tangent vector at ``u0`` pointing to ``u1`` (the flow-matching velocity)."""
    d = u1 - u0
    return jnp.where(PERIODIC, d - jnp.round(d), d)


@eqx.filter_jit
def exponential_map(
    u0: Float[Array, "... P"], v: Float[Array, "... P"]
) -> Float[Array, "... P"]:
    """Move from ``u0`` along tangent ``v`` (wrapping the periodic dims)."""
    x = u0 + v
    return jnp.where(PERIODIC, x % 1.0, x)


@eqx.filter_jit
def geodesic(
    t: Scalar, u0: Float[Array, "... P"], u1: Float[Array, "... P"]
) -> Float[Array, "... P"]:
    """Probability path point at time ``t`` between base ``u0`` and target ``u1``."""
    return exponential_map(u0, t * logarithmic_map(u0, u1))


@eqx.filter_jit
def match_sources(
    u0: Float[Array, "S D"],
    u1: Float[Array, "S D"],
) -> tuple[Float[Array, "S D"], Scalar]:
    """Reorder ``u0`` rows to minimize total pairwise cost against ``u1``."""
    n_sources = u0.shape[0]

    def cost(a: Float[Array, "P"], b: Float[Array, "P"]) -> Scalar:
        return jnp.sum(logarithmic_map(a, b) ** 2)

    pairwise_cost = jax.vmap(jax.vmap(cost, in_axes=(None, 0)), in_axes=(0, None))
    C = pairwise_cost(u0, u1)

    if n_sources <= 4:
        perms = jnp.array(list(itertools.permutations(range(n_sources))))  # (P, S)
        costs = jnp.sum(C[perms, jnp.arange(n_sources)[None, :]], axis=-1)  # (P,)
        best = jnp.argmin(costs)
        return u0[perms[best]], costs[best]

    # Hungarian is O(n^3) and GPU-hostile, so for many sources use randomized
    # local search: align both by SNR, then each round pair targets at random
    # and apply every beneficial pairwise swap. Cost decreases monotonically.
    idx = jnp.arange(n_sources)
    order0, order1 = jnp.argsort(u0[:, 2]), jnp.argsort(u1[:, 2])  # amplitude ~ SNR
    a = jnp.zeros(n_sources, dtype=jnp.int32).at[order1].set(order0.astype(jnp.int32))

    # targets to pair each round: drop the lowest-SNR one when the count is odd
    pool = order1[n_sources % 2 :]
    key = jr.key(0)

    def sweep(i: int, a: Float[Array, "S"]) -> Float[Array, "S"]:
        perm = jr.permutation(jr.fold_in(key, i), pool)
        p, q = perm[::2], perm[1::2]  # disjoint target pairs
        ap, aq = a[p], a[q]
        gain = C[ap, p] + C[aq, q] - C[aq, p] - C[ap, q]
        swap = gain > 0.0
        return a.at[p].set(jnp.where(swap, aq, ap)).at[q].set(jnp.where(swap, ap, aq))

    a = jax.lax.fori_loop(0, 8 * n_sources.bit_length(), sweep, a)
    return u0[a], jnp.sum(C[a, idx])


@eqx.filter_jit
def prior_inverse_cdf(u: Float[Array, "... 8"]) -> Float[Array, "... 8"]:
    """Inverse CDF of the Galactic Binary prior: Uniform(0,1)^8 to physical params."""
    u_f0, u_fdot, u_A, u_ra, u_dec, u_psi, u_iota, u_phi0 = jnp.split(u, 8, axis=-1)

    # Parameter priors form arXiv:2606.29039 # TODO: not true, need updated ref
    f0 = inverse_cdfs.log_uniform(u_f0, range=(1e-4, MAX_FREQUENCY))
    A = inverse_cdfs.log_uniform(u_A, range=(1e-24, 1e-22))
    fdot = inverse_cdfs.log_uniform(u_fdot, range=(1e-22, 1e-18))

    # Uniform angles
    ra = inverse_cdfs.uniform(u_ra, range=(0.0, 2.0 * jnp.pi))
    psi = inverse_cdfs.uniform(u_psi, range=(0.0, jnp.pi))
    phi0 = inverse_cdfs.uniform(u_phi0, range=(-jnp.pi, jnp.pi))

    # Isotropic sky / orientation: dec ~ cos(dec) [-pi/2, pi/2] and iota ~ sin(iota) [0, pi]
    dec = inverse_cdfs.cosine_pdf(u_dec, range=(-jnp.pi / 2.0, jnp.pi / 2.0))
    iota = inverse_cdfs.cosine_pdf(u_iota, range=(0.0, jnp.pi))
    return jnp.concat([f0, fdot, A, ra, dec, psi, iota, phi0], axis=-1)


@eqx.filter_jit
def clean_signal(params: Float[Array, "S 8"]) -> Float[Array, "T C"]:
    """Clean A/E/T TDI time-domain signal for a Galactic Binary, shape (n_times, n_channels)."""
    n_freqs = len(jnp.fft.rfftfreq(N_SAMPLES, SAMPLING_STEP))

    # frequency-domain TDI response segments per source
    orbit = lisaorbits.EqualArmlengthOrbits()
    jgb = jaxgb.JaxGB(orbit, t_obs=T_OBS, t0=0.0, n=256)
    segments = jgb.get_tdi(params, tdi_generation=1.5, tdi_combination="AET")
    segments = jnp.stack(segments, axis=0).astype(jnp.complex128)  # (S, 3, n)

    # index-add each segment into the full spectrum (duplicates sum coherently, drop sentinel)
    start_idx = jgb.get_kmin(params[:, 0])  # (S,)
    idx = start_idx[:, None] + jnp.arange(256, dtype=jnp.int32)  # (S, 256)
    full = jnp.zeros((3, n_freqs), dtype=jnp.complex128)
    full = full.at[:, idx].add(segments, mode="drop")

    signal = jnp.fft.irfft(full, n=N_SAMPLES, axis=-1)  # (C, T)
    return rearrange(signal, "c t -> t c")


@eqx.filter_jit
def noise_psd(
    freqs: Float[Array, "F"], A: float = 3.0, P: float = 15.0
) -> Float[Array, "3 F"]:
    """Analytic TDI 1.5 instrumental PSD for the A/E/T channels.

    Returns the one-sided PSD evaluated at ``freqs``, stacked as ``[A, E, T]`` along
    the leading axis (A and E share the same PSD). The DC bin (and any ``f == 0``)
    is set to zero; negative ``fftfreq`` bins use ``|f|``.
    """
    # TODO: documentation for these parameters/formulas
    f = jnp.abs(freqs)
    fs = jnp.where(f > 0, f, 1.0)  # avoid div-by-zero at DC; nulled below
    fstar = 1.0 / (2.0 * jnp.pi * ARM_LENGTH / SPEED_OF_LIGHT)
    tdi15_factor = 4.0 * jnp.sin(fs / fstar) * fs / fstar
    n_ae = (
        0.5
        * (2.0 + jnp.cos(fs / fstar))
        * (P / ARM_LENGTH) ** 2
        * 1e-24
        * (1.0 + (0.002 / fs) ** 4)
        + 2.0
        * (1.0 + jnp.cos(fs / fstar) + jnp.cos(fs / fstar) ** 2)
        * (A / ARM_LENGTH) ** 2
        * 1e-30
        * (1.0 + (0.0004 / fs) ** 2)
        * (1.0 + (fs / 0.008) ** 4)
        * (1.0 / (2.0 * jnp.pi * fs)) ** 4
    )
    n_t = (
        1e-24
        * (1.0 - jnp.cos(fs / fstar))
        * (P / ARM_LENGTH) ** 2
        * (1.0 + (0.002 / fs) ** 4)
        + 2.0
        * (1.0 - jnp.cos(fs / fstar)) ** 2
        * (A / ARM_LENGTH) ** 2
        * 1e-30
        * (1.0 + (0.0004 / fs) ** 2)
        * (1.0 + (fs / 0.008) ** 4)
        * (1.0 / (2.0 * jnp.pi * fs)) ** 4
    )
    psd = tdi15_factor * jnp.stack([n_ae, n_ae, n_t], axis=0)  # (3, F)
    return jnp.where(f > 0, psd, 0.0)


@eqx.filter_jit
def sample_noise(key: Key) -> Float[Array, "T C"]:
    """Time-domain A/E/T instrumental noise matching the TDI 1.5 PSD, shape (n_times, n_channels)."""
    psd = noise_psd(jnp.fft.rfftfreq(N_SAMPLES, SAMPLING_STEP))

    # colored draw: shape white gaussian by the per-bin noise std in the frequency domain
    white = jr.normal(key, (3, N_SAMPLES))
    colored = jnp.fft.rfft(white, axis=-1) * jnp.sqrt(psd / (2.0 * SAMPLING_STEP))
    noise = jnp.fft.irfft(colored, n=N_SAMPLES, axis=-1)  # (C, T)
    return rearrange(noise, "c t -> t c")


@eqx.filter_jit
def optimal_snr(params: Float[Array, "S 8"]) -> Scalar:
    """Combined matched-filter optimal SNR of a Galactic Binary's clean A/E/T signal against the TDI 1.5 PSD."""
    signal = rearrange(clean_signal(params), "t c -> c t")
    freqs = jnp.fft.rfftfreq(N_SAMPLES, SAMPLING_STEP)
    spectra = jnp.fft.rfft(signal, axis=-1)
    psd = noise_psd(freqs)  # (3, F)

    integrand = jnp.abs(spectra) ** 2 / jnp.where(psd > 0.0, psd, 1.0)
    integrand = jnp.where(psd > 0.0, integrand, 0.0)
    return jnp.sqrt(2.0 * SAMPLING_STEP / N_SAMPLES * jnp.sum(integrand))


@eqx.filter_jit
def preprocess_datastream(
    datastream: Float[Array, "T C"], nf: int = 1024
) -> Float[Array, "T F C"]:
    """Whiten the A/E/T datastream and convert to a WDM time-frequency image (the network's conditioning y)."""
    datastream = rearrange(datastream, "t c -> c t")
    # crop so the FFT length is a multiple of nf*2 (keeps the WDM time grid even)
    n_samples = (N_SAMPLES // (2 * nf)) * (2 * nf)
    datastream = datastream[:, :n_samples]
    psd = noise_psd(jnp.fft.fftfreq(n_samples, SAMPLING_STEP))
    nt = n_samples // nf

    # whiten every channel in the frequency domain by the expected per-bin noise std
    spectra = jnp.fft.fft(datastream, axis=-1)
    avg_noise_power = psd * n_samples / (2.0 * SAMPLING_STEP)
    scale = jnp.where(avg_noise_power > 0.0, jnp.sqrt(avg_noise_power), 1.0)
    spectra = spectra / scale

    # from_freq_to_wdm is batch-first: (C, F) -> (C, T, nf)
    y = from_freq_to_wdm(
        spectra, nt=nt, nf=nf, a=1.0 / 3.0, d=1.0, dt=SAMPLING_STEP, backend="jax"
    )

    # compress range and drop the zero-frequency band
    y = jnp.arcsinh(y)[..., 1:]
    return rearrange(y, "c t f -> t f c")


@eqx.filter_jit
def get_physics_sample(
    key: Key,
    n_sources: int = N_SOURCES,
) -> tuple[
    Float[Array, "S P"],
    Float[Array, "S P"],
    Float[Array, "T C"],
    Float[Array, "T C"],
    Float[Array, "T F C"],
    Float[Array, "T F C"],
]:
    """One datastream ``(u, params, datastream, signal, y, y_clean)`` from injected physics priors."""
    key_params, key_noise = jr.split(key, 2)

    # sample normalized params from the prior and generate signal
    u = jr.uniform(key_params, shape=(n_sources, 8))
    params = prior_inverse_cdf(u)
    signal = clean_signal(params)

    # sample noise and add to the signal to get the datastream
    noise = sample_noise(key_noise)
    datastream = signal + noise

    # preprocess conditioning signals to WDM images
    y = preprocess_datastream(datastream)
    y_clean = preprocess_datastream(signal)

    # only keep relevant parameters
    if SIMPLIFIED_PROBLEM:
        u = u[..., MASK]
        params = params[..., MASK]
    return u, params, datastream, signal, y, y_clean
