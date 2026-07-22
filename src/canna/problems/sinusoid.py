from functools import partial
import math
from jaxtyping import Array, Complex, Float, Key
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from wdm_transform.transforms import from_freq_to_wdm_band

from .. import charts, geometries, priors
from .base import Problem


class NoisySinusoid(Problem):
    """Superposed sinusoids in white noise, their parameters an unordered set."""

    n_sources: int = eqx.field(static=True, default=2)
    t_obs: float = eqx.field(static=True, default=10000.0)
    sampling_step: float = eqx.field(static=True, default=1.0)
    noise_level: float = eqx.field(static=True, default=1.0)
    wdm_freq_bands: int = eqx.field(static=True, default=512)
    patch_downsample: int = eqx.field(static=True, default=4)
    amp_range: tuple[float, float] = eqx.field(static=True, default=(0.01, 1.0))
    freq_range: tuple[float, float] = eqx.field(static=True, default=(1e-2, 1e-1))

    @property
    def prior(self) -> priors.Set:
        return priors.Set(
            priors.Product(
                amp=priors.LogUniform(*self.amp_range),
                freq=priors.LogUniform(*self.freq_range),
                phase=priors.PeriodicUniform(),
            ),
            size=self.n_sources,
        )

    @property
    def chart(self) -> charts.Product:
        return self.prior.chart

    @property
    def geometry(self) -> geometries.Set:
        return self.prior.geometry

    def sample_physical(self, key: Key[Array, ""]) -> Float[Array, "S 3"]:
        return self.prior(key)

    def sample_point(self, key: Key[Array, ""]) -> Float[Array, "S 4"]:
        return self.chart.forward(self.sample_physical(key))

    def clean_signal(self, p: Float[Array, "... S 3"]) -> Float[Array, "... T 2"]:
        t = jnp.arange(0, self.t_obs, self.sampling_step)
        amp, freq, phase = jnp.split(p, 3, axis=-1)
        angle = 2.0 * jnp.pi * freq * t + phase
        sin = jnp.sum(amp * jnp.sin(angle), axis=-2)
        cos = jnp.sum(amp * jnp.cos(angle), axis=-2)
        return jnp.stack([sin, cos], axis=-1)

    def sample_observation(
        self, key: Key[Array, ""], p: Float[Array, "... S 3"], clean: bool = False
    ) -> Float[Array, "... T 2"]:
        signal = self.clean_signal(p)
        if not clean:
            noise = jr.normal(key, signal.shape)
            noise = noise * jnp.sqrt(self.noise_level / (2.0 * self.sampling_step))
            signal = signal + noise
        return signal

    def snr(self, p: Float[Array, "... S 3"]) -> Float[Array, "..."]:
        amp = p[..., 0]
        power = self.t_obs * jnp.sum(amp**2, axis=-1)
        return jnp.sqrt(2.0 * power / self.noise_level)

    def log_likelihood(
        self, p: Float[Array, "... S 3"], o: Float[Array, "... T 2"]
    ) -> Float[Array, "..."]:
        variance = self.noise_level / (2.0 * self.sampling_step)
        residual = o - self.clean_signal(p)
        return -0.5 * jnp.sum(residual**2, axis=(-2, -1)) / variance

    def preprocess(self, o: Float[Array, "... T 2"]) -> Float[Array, "... t f 2"]:
        # go to frequency domain first -- take the data as given, no zero padding
        T = o.shape[-2]
        spectrum = jnp.fft.rfft(o, axis=-2)
        df = 1.0 / (T * self.sampling_step)

        # the prior band, in rfft bins
        fmin, fmax = self.freq_range
        kmin, kmax = math.floor(fmin / df), math.floor(fmax / df)
        band_bins = kmax - kmin + 1

        # tile the grid: nt time rows (multiple of patch_downsample), nf full freq
        # channels; keep wdm_freq_bands channels of nt//2 bins spanning the band
        pd = self.patch_downsample
        nt = pd * math.ceil(2 * band_bins / (self.wdm_freq_bands * pd))
        half = nt // 2
        nf = (T // nt) - (T // nt) % 2
        nfreqs_fourier = nt * nf // 2 + 1

        # center the kept block on the band, aligned to a channel boundary, then
        # slice the real spectrum across the whole block -- no zero fill
        block = self.wdm_freq_bands * half
        mmin = max(0, (kmin - (block - band_bins) // 2) // half)
        kmin = mmin * half
        band = spectrum[..., kmin : kmin + block, :]

        @partial(jnp.vectorize, signature="(w,c)->(t,f,c)")
        @partial(jax.vmap, in_axes=-1, out_axes=-1)
        def to_wdm(channel: Complex[Array, "w"]) -> Float[Array, "t f"]:
            return from_freq_to_wdm_band(
                channel,
                df=df,
                nfreqs_fourier=nfreqs_fourier,
                kmin=kmin,
                nfreqs_wdm=nf,
                ntimes_wdm=nt,
                mmin=mmin,
                nf_sub_wdm=self.wdm_freq_bands,
                a=1.0 / 3.0,
                d=1.0,
                backend="jax",
            )

        return jnp.arcsinh(to_wdm(band))  # tame extreme values
