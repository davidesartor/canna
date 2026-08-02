from typing import Optional
import math

from jaxtyping import Array, Complex, Float, Key
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

import lisaorbits
from jaxgb import jaxgb
from ..wdm import from_freq_to_wdm_band
from .. import charts, geometries, priors
from .base import Problem

SPEED_OF_LIGHT = 299792458.0  # [m/s]
SUN_MASS = 1.98892e30  # [kg]
GRAVITATIONAL_CONSTANT = 6.67430e-11  # [m^3 kg^-1 s^-2]
SUN_MASS_TIME = GRAVITATIONAL_CONSTANT * SUN_MASS / SPEED_OF_LIGHT**3  # [s]
EARTH_ORBIT_SPEED = 29785.0  # [m/s]
MIN_RESPONSE_POINTS = 256  # enough time samples to resolve the orbital modulation


def fdot_from_chirp_mass(
    chirp_mass: Float[Array, "..."], f0: Float[Array, "..."]
) -> Float[Array, "..."]:
    """Radiation-reaction chirp of a circular binary, for chirp mass in solar masses."""
    smt = GRAVITATIONAL_CONSTANT * SUN_MASS / SPEED_OF_LIGHT**3  # [s]
    ratio = 96.0 / 5.0 * jnp.pi ** (8 / 3) * f0 ** (11 / 3)
    return ratio * (smt * chirp_mass) ** (5 / 3)


def chirp_mass_from_fdot(
    fdot: Float[Array, "..."], f0: Float[Array, "..."]
) -> Float[Array, "..."]:
    """Inverse of `fdot_from_chirp_mass`, in solar masses."""
    ratio = 5.0 / 96.0 * jnp.pi ** (-8 / 3) * f0 ** (-11 / 3)
    smt = GRAVITATIONAL_CONSTANT * SUN_MASS / SPEED_OF_LIGHT**3  # [s]
    return (ratio * fdot) ** (3 / 5) / smt


class ChirpMass(priors.Prior):
    """Chirp mass of a double white dwarf, drawn through its two component masses.

    This is the observationally driven Galactic DWD model of Korol et al. 2022
    (MNRAS 511, 5936; arXiv:2109.10972). The heavier component follows the
    single-white-dwarf mass function of Kepler et al. 2015 (MNRAS 446, 4078), a
    Gaussian mixture here truncated to [m_min, m_max]; the lighter one is uniform on
    [m_min, m1], the flat mass ratio close binaries are observed to follow (Moe &
    Di Stefano 2017). The induced chirp mass peaks near 0.45 Msun, with a median
    around 0.43 and a tail out to ~0.9. Korol et al. also swap the secondary for a
    uniform draw on [0.2, 1.2] when m1 falls in the ELM regime below 0.25 Msun; the
    default mixture never gets there, so that branch is left out.
    Embeds to R with Euclidean geometry.
    """

    weights: Float[Array, "K"] = eqx.field(
        converter=jnp.asarray, default=(0.81, 0.14, 0.05)
    )
    means: Float[Array, "K"] = eqx.field(
        converter=jnp.asarray, default=(0.65, 0.57, 0.81)
    )
    stds: Float[Array, "K"] = eqx.field(
        converter=jnp.asarray, default=(0.044, 0.097, 0.187)
    )
    m_min: float = eqx.field(static=True, default=0.15)
    m_max: float = eqx.field(static=True, default=1.4)

    # the chart's whitening constants, computed by quadrature
    log_mean: Float[Array, "1"] = eqx.field(init=False)
    log_std: Float[Array, "1"] = eqx.field(init=False)

    def __post_init__(self):
        # the induced log chirp mass moments have no closed form, so take them on a
        # grid over the truncated primary mixture and the flat mass ratio
        m1 = jnp.linspace(self.m_min, self.m_max, 256)[:, None]
        ratio = jnp.linspace(0.0, 1.0, 256)
        m2 = self.m_min + ratio * (m1 - self.m_min)
        log_mc = 0.6 * jnp.log(m1 * m2) - 0.2 * jnp.log(m1 + m2)

        # equinox runs the field converters after __post_init__, so convert by hand here
        means, stds, weights = (
            jnp.asarray(f) for f in (self.means, self.stds, self.weights)
        )
        density = jax.scipy.stats.norm.pdf(m1[..., None], means, stds)
        weight = (density * weights).sum(-1)
        weight = weight / weight.sum() / ratio.size

        self.log_mean = jnp.atleast_1d((weight * log_mc).sum())
        self.log_std = jnp.atleast_1d(
            jnp.sqrt((weight * (log_mc - self.log_mean) ** 2).sum())
        )

    def __call__(self, key: Key[Array, ""]) -> Float[Array, "1"]:
        key_component, key_primary, key_secondary = jr.split(key, 3)

        # truncation leaves each component a different share of the mixture
        low = (self.m_min - self.means) / self.stds
        high = (self.m_max - self.means) / self.stds
        kept = jax.scipy.stats.norm.cdf(high) - jax.scipy.stats.norm.cdf(low)
        k = jr.choice(key_component, self.weights.size, p=self.weights * kept)

        z = jr.truncated_normal(key_primary, low[k], high[k])
        m1 = self.means[k] + self.stds[k] * z
        m2 = jr.uniform(key_secondary, minval=self.m_min, maxval=m1)
        return jnp.atleast_1d((m1 * m2) ** 0.6 / (m1 + m2) ** 0.2)

    @property
    def support(self) -> tuple[float, float]:
        # both extremes are equal-mass pairs, where the chirp mass is m / 2^(1/5)
        return self.m_min * 2**-0.2, self.m_max * 2**-0.2

    @property
    def geometry(self) -> geometries.Euclidean:
        return geometries.Euclidean(1)

    @property
    def chart(self) -> charts.LogAffine:
        scale = 1 / self.log_std
        return charts.LogAffine(shift=-scale * self.log_mean, scale=scale)


class LisaGB(Problem):
    """Galactic binaries seen by LISA, their parameters an unordered set."""

    n_sources: int = eqx.field(static=True, default=4)
    t_obs: float = eqx.field(static=True, default=2 * 365.25 * 24 * 60 * 60)  # 2yr [s]
    sampling_step: float = eqx.field(static=True, default=0.25)  # [s]
    wdm_freq_bands: int = eqx.field(static=True, default=16384)
    patch_downsample: int = eqx.field(static=True, default=16)

    orbit: lisaorbits.Orbits = eqx.field(
        static=True, default=lisaorbits.EqualArmlengthOrbits()
    )
    # None sizes the response window from t_obs and the priors
    response_points: Optional[int] = eqx.field(static=True, default=None)
    oms_noise: float = eqx.field(static=True, default=15.0)
    acceleration_noise: float = eqx.field(static=True, default=3.0)

    # white dwarf component masses, from which the chirp mass prior is built
    mass_range: tuple[float, float] = eqx.field(static=True, default=(0.15, 1.4))
    f0_range: tuple[float, float] = eqx.field(static=True, default=(1e-4, 12.0e-3))
    a_range: tuple[float, float] = eqx.field(static=True, default=(1e-24, 1e-22))

    # static too: JaxGB is a plain object, not a pytree, so it cannot be traced
    response: jaxgb.JaxGB = eqx.field(init=False, static=True)

    def __post_init__(self):
        # the response is response_points bins wide and centred on f0, so half of it
        # has to hold the annual doppler sideband plus the whole (upward) chirp drift.
        # the DC-straddle bound below caps it from the other side, and the two cross
        # over at short baselines, so this cannot be a fixed default
        drift = self.fdot_range[1] * self.t_obs
        doppler = self.f0_range[1] * EARTH_ORBIT_SPEED / SPEED_OF_LIGHT
        span = 2 * math.ceil((doppler + drift) * self.t_obs)
        needed = 1 << (max(span, MIN_RESPONSE_POINTS) - 1).bit_length()

        if self.response_points is None:
            self.response_points = needed

        assert needed <= self.response_points, (
            f"response_points={self.response_points} too narrow for a "
            f"Mc={self.chirp_mass_range[1]:.2f}Msun source at "
            f"f0_max={self.f0_range[1]:.1e}Hz over t_obs={self.t_obs:.3e}s: "
            f"it needs {needed} bins. "
            f"Raise response_points, or lower f0_range[1]/mass_range[1]."
        )
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
    def chirp_mass_prior(self) -> ChirpMass:
        return ChirpMass(m_min=self.mass_range[0], m_max=self.mass_range[1])

    @property
    def chirp_mass_range(self) -> tuple[float, float]:
        return self.chirp_mass_prior.support

    @property
    def fdot_range(self) -> tuple[float, float]:
        # fdot is not sampled: it is what the chirp mass and f0 priors imply
        low = fdot_from_chirp_mass(self.chirp_mass_range[0], self.f0_range[0])
        high = fdot_from_chirp_mass(self.chirp_mass_range[1], self.f0_range[1])
        return float(low), float(high)

    @property
    def prior(self) -> priors.Set:
        return priors.Set(
            priors.Product(
                f0=priors.LogUniform(*self.f0_range),
                chirp_mass=self.chirp_mass_prior,
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
        # column 1 carries the chirp mass; jaxgb wants the chirp it drives
        p = p.at[..., 1].set(fdot_from_chirp_mass(p[..., 1], p[..., 0]))

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

        band_bins = fmax - fmin + 1
        pd = self.patch_downsample
        nt = pd * math.ceil(2 * band_bins / (self.wdm_freq_bands * pd))
        half = nt // 2

        # center the kept block on the band, aligned to a channel boundary; the
        # data spans only the band, so zero-fill the block around it
        block = self.wdm_freq_bands * half
        mmin = max(0, (fmin - (block - band_bins) // 2) // half)
        kmin = mmin * half
        left = fmin - kmin
        pad = [(0, 0)] * (o.ndim - 2) + [(left, block - band_bins - left), (0, 0)]
        o = jnp.pad(o, pad)

        # convert the one-sided frequency-domain data to WDM bands; the transform
        # runs on the last axis, so the tdi channels ride along as a batch axis
        wdm = from_freq_to_wdm_band(
            jnp.moveaxis(o, -1, -2),
            ntimes=nt,
            nfreq_bands=self.wdm_freq_bands,
            mmin=mmin,
        )

        # compress dynamic range for better training stability
        return jnp.arcsinh(jnp.moveaxis(wdm, -3, -1))
