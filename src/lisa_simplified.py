from jaxtyping import Array, Float, Key, Scalar
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import einops
from wdm_transform.transforms import from_freq_to_wdm

from . import lisa

# re-export constants used downstream
from .lisa import DAY, MONTH, SAMPLING_STEP, WEEK, YEAR

INFERRED_PARAMS = ["f0", "A", "fdot", "psi"]
MASK = jnp.array([name in INFERRED_PARAMS for name in lisa.PARAMETER_NAMES])
PARAMETER_NAMES = [name for name in lisa.PARAMETER_NAMES if name in INFERRED_PARAMS]


def pad(v: Float[Array, "... 4"]) -> Float[Array, "... 8"]:
    # back into full 8-param slots (zeros elsewhere)
    padded = jnp.zeros(v.shape[:-1] + (8,), dtype=v.dtype)
    return padded.at[..., MASK].set(v)


def log_map(
    x0: Float[Array, "... 4"], x1: Float[Array, "... 4"]
) -> Float[Array, "... 4"]:
    """log_map restricted to the inferred dims."""
    return lisa.log_map(pad(x0), pad(x1))[..., MASK]


def exp_map(
    x0: Float[Array, "... 4"], v: Float[Array, "... 4"]
) -> Float[Array, "... 4"]:
    """exp_map restricted to the inferred dims."""
    return lisa.exp_map(pad(x0), pad(v))[..., MASK]


def match_sources(x0: Float[Array, "... 4"], x1: Float[Array, "... 4"]):
    """match_sources restricted to the inferred dims (nuisance slots contribute nothing)."""
    x0_matched, cost = lisa.match_sources(pad(x0), pad(x1))
    return x0_matched[..., MASK], cost


@eqx.filter_jit
def prior_inverse_cdf(u: Float[Array, "... 4"]) -> Float[Array, "... 4"]:
    """prior_inverse_cdf restricted to the inferred dims."""
    return lisa.prior_inverse_cdf(pad(u))[..., MASK]


@eqx.filter_jit
def optimal_snr(
    params: Float[Array, "S 4"],
    t_obs: float = YEAR,
    dt: float = SAMPLING_STEP,
) -> Scalar:
    """optimal_snr restricted to the inferred dims."""
    return lisa.optimal_snr(pad(params), t_obs=t_obs, dt=dt)


@eqx.filter_jit
def preprocess_datastream(
    datastream: Float[Array, "T 3"],
    t_obs: float = YEAR,
    dt: float = SAMPLING_STEP,
) -> Float[Array, "F T*C"]:
    # crop the datastream so frequency axis is to a multiple of 32 for the WDM grid
    # TODO: make the 32 time bins configurable
    n_samples = int(t_obs / dt)
    nt = 64
    n_samples = (n_samples // (2 * nt)) * (2 * nt)  # keep FFT length a multiple of nt*2
    nf = n_samples // nt  # guaranteed even since n_samples is a multiple of 2*nt
    datastream = datastream[:n_samples]

    # convert to frequency domain for whitening and WDM transform
    freqs = jnp.fft.fftfreq(n_samples, dt)
    datastream = jnp.fft.fft(datastream, axis=0)

    # whitening the datastream by the expected noise PSD
    for i, ch in enumerate("AET"):
        psd = jnp.where(freqs != 0, lisa.noise_psd(ch)(jnp.abs(freqs)), 0.0)  # type: ignore
        var = psd * n_samples / (2.0 * dt)  # E[|fft(noise)|^2] per bin
        datastream = datastream.at[:, i].divide(
            jnp.where(var == 0.0, 1.0, jnp.sqrt(var))
        )

    # process the data to make it more digestible
    y = from_freq_to_wdm(
        datastream.T,
        nt=nt,
        nf=nf,
        a=1.0 / 3.0,
        d=1.0,
        dt=SAMPLING_STEP,
        backend="jax",
    )
    y = einops.rearrange(y, "c t f -> t (f c)")
    y = jnp.arcsinh(y)  # avoids grossly large values in the WDM transform
    return y


@eqx.filter_jit
def get_train_batch(
    key: Key,
    batch_size: int,
    n_sources: int,
    t_obs: float = YEAR,
    dt: float = SAMPLING_STEP,
    snr_threshold: float = 0.0,
) -> tuple[
    Float[Array, "S 4"],
    Float[Array, "S 4"],
    Scalar,
    Float[Array, "T F*C"],
    Float[Array, "S 4"],
    Float[Array, "S 4"],
    Float[Array, "S 4"],
    Float[Array, "T 3"],
]:
    def train_sample(key: Key):
        key_x1, key_x0, key_t, key_y = jr.split(key, 4)
        x1 = jr.uniform(key_x1, shape=(n_sources, 8), minval=-1.0, maxval=1.0)
        x0 = jr.uniform(key_x0, shape=(n_sources, 8), minval=-1.0, maxval=1.0)
        x0, _ = lisa.match_sources(x0, x1)  # align sources to minimize transport cost
        t = jr.uniform(key_t, minval=0.0, maxval=1.0)

        # generate the conditioning signal
        u1 = (x1 + 1.0) / 2.0  # U(-1, 1) -> U(0,1)
        params = lisa.prior_inverse_cdf(u1)
        signal = lisa.clean_signal(params, t_obs=t_obs, dt=dt)
        noise = lisa.sample_noise(key_y, t_obs=t_obs, dt=dt)

        # amplify signals that have low SNR, up to snr_threshold; louder signals
        # are left untouched
        signal = signal * jnp.maximum(
            1.0, snr_threshold / lisa.optimal_snr(params, t_obs=t_obs, dt=dt)
        )

        datastream = signal + noise
        y = preprocess_datastream(datastream, t_obs=t_obs, dt=dt)

        # conditional flow-matching target
        xt = lisa.geodesic(t, x0, x1)
        dx = jax.jacobian(lisa.geodesic)(t, x0, x1)
        return xt, dx, t, y, x0, x1, params, datastream

    xt, dx, t, y, x0, x1, params, datastream = jax.vmap(train_sample)(
        jr.split(key, batch_size)
    )
    xt, dx, x0, x1, params = [el[..., MASK] for el in (xt, dx, x0, x1, params)]
    return xt, dx, t, y, x0, x1, params, datastream
