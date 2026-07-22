from functools import partial
import math

from jaxtyping import Array, Complex, Float, Key
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

import lisaorbits
from jaxgb import jaxgb
from wdm_transform.transforms import from_freq_to_wdm_band

from .. import charts, geometries, priors
from .base import Problem

SPEED_OF_LIGHT = 299792458.0  # [m/s]
SUN_MASS = 1.98892e30  # [kg]
GRAVITATIONAL_CONSTANT = 6.67430e-11  # [m^3 kg^-1 s^-2]


class LisaGB(Problem):
    """Galactic binaries seen by LISA, their parameters an unordered set."""

    # static: these set array shapes, and a tracer cannot
    n_sources: int = eqx.field(static=True, default=4)
    t_obs: float = eqx.field(static=True, default=2 * 365.25 * 24 * 60 * 60)  # 2yr [s]
    sampling_step: float = eqx.field(static=True, default=0.25)  # [s]
    wdm_freq_bands: int = eqx.field(static=True, default=16384)
    patch_downsample: int = eqx.field(static=True, default=16)

    orbit: lisaorbits.Orbits = eqx.field(
        static=True, default=lisaorbits.EqualArmlengthOrbits()
    )
    response_points: int = eqx.field(static=True, default=256)
    oms_noise: float = eqx.field(static=True, default=15.0)
    acceleration_noise: float = eqx.field(static=True, default=3.0)

    # realistic priors from #TODO: add reference
    f0_range: tuple[float, float] = eqx.field(static=True, default=(1e-4, 12.0e-3))
    fdot_range: tuple[float, float] = eqx.field(static=True, default=(1e-18, 1e-15))
    a_range: tuple[float, float] = eqx.field(static=True, default=(1e-24, 1e-22))

    # static too: JaxGB is a plain object, not a pytree, so it cannot be traced
    response: jaxgb.JaxGB = eqx.field(init=False, static=True)

    def __post_init__(self):
        self.response = jaxgb.JaxGB(
            self.orbit,
            t_obs=self.t_obs,
            n=self.response_points,
        )
        # jaxgb gives the raw +f0 lobe; if its band straddles DC (kmin < 0) the
        # sub-DC bins are the -f0 lobe's tail jaxgb does not fold in, so forbid it
        kmin = int(self.response.get_kmin(jnp.array([self.f0_range[0]]))[0])
        assert kmin >= 0, (
            f"t_obs={self.t_obs:.3e}s too short for f0_min={self.f0_range[0]:.1e}Hz: "
            f"band straddles DC (kmin={kmin}). "
            f"Need t_obs >= {self.response_points / 2 / self.f0_range[0]:.3e}s."
        )

    @property
    def prior(self) -> priors.Set:
        return priors.Set(
            priors.Product(
                f0=priors.LogUniform(*self.f0_range),
                fdot=priors.LogUniform(*self.fdot_range),
                amp=priors.LogUniform(*self.a_range),
                sky=priors.Isotropic(2),
                orientation=priors.Isotropic(2),
                phi0=priors.PeriodicUniform(),
            ),
            size=self.n_sources,
        )

    @property
    def chart(self) -> charts.Product:
        return self.prior.chart

    @property
    def geometry(self) -> geometries.Set:
        return self.prior.geometry

    @property
    def f0_range_bins(self) -> tuple[int, int]:
        # first and last rfft bins any source in the band can occupy
        fmin = math.floor(self.f0_range[0] * self.t_obs - self.response_points / 2)
        fmax = math.ceil(self.f0_range[1] * self.t_obs + self.response_points / 2)
        return fmin, fmax

    def sample_physical(self, key: Key[Array, ""]) -> Float[Array, "S 8"]:
        return self.prior(key)

    def sample_point(self, key: Key[Array, ""]) -> Float[Array, "S 11"]:
        return self.chart.forward(self.sample_physical(key))

    def clean_signal(self, p: Float[Array, "... S 8"]) -> Complex[Array, "... F 3"]:
        # flip iota from latitude [-pi/2, pi/2] to colatitude [0, pi] for jaxgb
        p = p.at[..., 6].set(jnp.pi / 2 - p[..., 6])

        def combine(sources: Float[Array, "S 8"]) -> Complex[Array, "F 3"]:
            # get the local freq response for each source
            segments = self.response.get_tdi(
                sources, tdi_generation=1.5, tdi_combination="AET"
            )
            segments = jnp.stack(segments, axis=-1)

            # place each segment in its positive band on the one-sided grid
            fmin, fmax = self.f0_range_bins
            start = self.response.get_kmin(sources[:, 0]) - fmin
            idxs = start[:, None] + jnp.arange(self.response_points)
            full = jnp.zeros((fmax - fmin + 1, 3), dtype=segments.dtype)
            full = full.at[idxs].add(segments)
            return full

        return jnp.vectorize(combine, signature="(s,d)->(f,c)")(p)

    def noise_psd(self, f: Float[Array, "..."]) -> Float[Array, "... 3"]:
        # first-generation TDI (1.5) noise PSD, stacked A/E/T on the last axis
        # Babak, Hewitson & Petiteau 2021, arXiv:2108.01167
        arm_length = self.response.arm_length
        f = jnp.abs(f)
        arm_phase = 2.0 * jnp.pi * arm_length / SPEED_OF_LIGHT * f
        transfer = jnp.cos(arm_phase)
        optical_metrology = (
            (self.oms_noise / arm_length) ** 2 * 1e-24 * (1.0 + (0.002 / f) ** 4)
        )
        test_mass_acceleration = (
            (self.acceleration_noise / arm_length) ** 2
            * 1e-30
            * (1.0 + (0.0004 / f) ** 2)
            * (1.0 + (f / 0.008) ** 4)
            / (2.0 * jnp.pi * f) ** 4
        )
        n_ae = 2.0 * test_mass_acceleration * (
            1.0 + transfer + transfer**2
        ) + 0.5 * optical_metrology * (2.0 + transfer)
        n_t = 2.0 * test_mass_acceleration * (
            1.0 - transfer
        ) ** 2 + optical_metrology * (1.0 - transfer)

        n = jnp.stack([n_ae, n_ae, n_t], axis=-1)
        tdi15_factor = 4.0 * jnp.sin(arm_phase) ** 2
        return n * tdi15_factor[..., None]

    def sample_observation(
        self, key: Key[Array, ""], p: Float[Array, "... S 8"], clean: bool = False
    ) -> Complex[Array, "... F 3"]:
        spectra = self.clean_signal(p)
        if not clean:  # add colored noise
            fmin, fmax = self.f0_range_bins
            freqs = 1 / self.t_obs * jnp.arange(fmin, fmax + 1)
            power = self.noise_psd(freqs) * self.t_obs / (4.0 * self.sampling_step**2)
            real, imag = jr.normal(key, (2, *spectra.shape)) * jax.lax.rsqrt(2.0)
            noise = jnp.sqrt(power) * (real + 1j * imag)
            spectra = spectra + noise
        return spectra

    def snr(self, p: Float[Array, "... S 8"]) -> Float[Array, "..."]:
        spectra = self.clean_signal(p)
        fmin, fmax = self.f0_range_bins
        freqs = 1 / self.t_obs * jnp.arange(fmin, fmax + 1)
        power = self.noise_psd(freqs) * self.t_obs / (4.0 * self.sampling_step**2)
        integrand = jnp.abs(spectra) ** 2 / power
        return jnp.sqrt(jnp.sum(integrand, axis=(-2, -1)))

    def log_likelihood(
        self, p: Float[Array, "... S 8"], o: Complex[Array, "... F 3"]
    ) -> Float[Array, "..."]:
        fmin, fmax = self.f0_range_bins
        freqs = 1 / self.t_obs * jnp.arange(fmin, fmax + 1)
        power = self.noise_psd(freqs) * self.t_obs / (4.0 * self.sampling_step**2)
        residual = o - self.clean_signal(p)
        return -jnp.sum(jnp.abs(residual) ** 2 / power, axis=(-2, -1))

    def preprocess(self, o: Complex[Array, "... F 3"]) -> Float[Array, "... t f 3"]:
        # whiten the datastream by the nominal noise psd
        fmin, fmax = self.f0_range_bins
        freqs = 1 / self.t_obs * jnp.arange(fmin, fmax + 1)
        power = self.noise_psd(freqs) * self.t_obs / (4.0 * self.sampling_step**2)
        o = jnp.where(freqs[..., None] > 0.0, o * jax.lax.rsqrt(power), o)

        # tile the grid: nt time rows (multiple of patch_downsample), nf full freq
        # channels; keep wdm_freq_bands channels of nt//2 bins spanning the band
        n = int(self.t_obs // self.sampling_step)
        band_bins = fmax - fmin + 1
        pd = self.patch_downsample
        nt = pd * math.ceil(2 * band_bins / (self.wdm_freq_bands * pd))
        half = nt // 2
        nf = (n // nt) - (n // nt) % 2
        nfreqs_fourier = nt * nf // 2 + 1

        # center the kept block on the band, aligned to a channel boundary; the
        # data spans only the band, so zero-fill the block around it
        block = self.wdm_freq_bands * half
        mmin = max(0, (fmin - (block - band_bins) // 2) // half)
        kmin = mmin * half
        left = fmin - kmin
        pad = [(0, 0)] * (o.ndim - 2) + [(left, block - band_bins - left), (0, 0)]
        o = jnp.pad(o, pad)

        # convert the one-sided frequency-domain data to WDM bands
        @partial(jnp.vectorize, signature="(w,c)->(t,f,c)")
        @partial(jax.vmap, in_axes=-1, out_axes=-1)
        def to_wdm(channel: Complex[Array, "w"]) -> Float[Array, "t f"]:
            return from_freq_to_wdm_band(
                channel,
                df=1.0 / self.t_obs,
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

        # compress dynamic range for better training stability
        return jnp.arcsinh(to_wdm(o))
