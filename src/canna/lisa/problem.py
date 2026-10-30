"""The LisaGB problem configuration, its sliding frequency window and the TDI noise."""

import math

import equinox as eqx
import jax.numpy as jnp
import lisaorbits
from jaxgb import jaxgb
from jaxtyping import Array, Float, Int, Key

from .physics import SPEED_OF_LIGHT, sample_sources

WINDOW_STRIDE = 2  # window starts snap to even bins
# the window spans 4 response bands; f0 keeps half a band clear of each edge, so 3 are drawable
WINDOW_RESPONSES = 4


class LisaGB(eqx.Module):
    """Galactic binaries seen by LISA in one sliding frequency window."""

    n_sources: int = eqx.field(static=True, default=4)
    t_obs: float = eqx.field(static=True, default=2 * 365.25 * 24 * 60 * 60)  # 2yr [s]
    # a source's response bins: 128 keeps 99% of the above-noise power of 99% of full-B sources (reports/prior/response-span)
    response_points: int = eqx.field(static=True, default=128)
    oms_noise: float = eqx.field(static=True, default=15.0)
    acceleration_noise: float = eqx.field(static=True, default=3.0)
    f0_range: tuple[float, float] = eqx.field(static=True, default=(1e-4, 12.0e-3))
    # "middle" charts phi0 at mid-observation, phi0 - pi f0 T - pi fdot T^2 / 4: the data phase at f0's bin
    phase_reference: str = eqx.field(static=True, default="middle")
    # window centres follow about the visible prior's f0, p(f0) P(snr >= visible_snr | f0) (simulator.sample_window)
    visible_snr: float = eqx.field(static=True, default=7.0)
    response: jaxgb.JaxGB = eqx.field(init=False, static=True)

    def __post_init__(self):
        self.f0_range = tuple(self.f0_range)
        assert self.phase_reference in ("start", "middle"), self.phase_reference
        assert (
            self.response_points & (self.response_points - 1) == 0
        ), f"response_points={self.response_points} is not a power of two"

        self.response = jaxgb.JaxGB(
            lisaorbits.EqualArmlengthOrbits(), t_obs=self.t_obs, n=self.response_points
        )
        # jaxgb does not fold the -f0 lobe in, so the band must not straddle DC
        kmin = int(self.response.get_kmin(jnp.array([self.f0_range[0]]))[0])
        assert kmin >= 0, f"f0_min={self.f0_range[0]:.1e}Hz straddles DC (kmin={kmin})"
        lo, hi = self.window_index_range
        assert lo <= hi, f"the {self.window_bins}-bin window overruns f0_range"

    @property
    def window_bins(self) -> int:
        return WINDOW_RESPONSES * self.response_points

    @property
    def window_index_range(self) -> tuple[int, int]:
        # window_start = index * WINDOW_STRIDE; the ends are where the interior just covers f0_range
        guard = self.response_points // 2
        lo = math.floor((self.f0_range[0] * self.t_obs - guard) / WINDOW_STRIDE)
        hi = math.ceil(
            (self.f0_range[1] * self.t_obs + guard - self.window_bins) / WINDOW_STRIDE
        )
        return max(lo, 1), max(hi, max(lo, 1))


def sample_f(problem: LisaGB, key: Key[Array, ""]) -> Float[Array, ""]:
    """Centre frequency of a window slid by the source prior's f0 marginal; training uses simulator.sample_window."""
    f_centre = sample_sources(key, *map(jnp.asarray, problem.f0_range), 1)[0, 0]
    return window_centre(problem, f_centre)


def window_centre(problem: LisaGB, f_centre: Float[Array, ""]) -> Float[Array, ""]:
    """Centre frequency of the window on the stride grid nearest f_centre, clamped inside the band."""
    lo, hi = problem.window_index_range
    index = jnp.clip(
        jnp.round((f_centre * problem.t_obs - problem.window_bins / 2) / WINDOW_STRIDE),
        lo,
        hi,
    )
    return (index * WINDOW_STRIDE + problem.window_bins / 2) / problem.t_obs


def window_start(problem: LisaGB, f: Float[Array, ""]) -> Int[Array, ""]:
    """First rfft bin of the window centred at f."""
    index = jnp.round((f * problem.t_obs - problem.window_bins / 2) / WINDOW_STRIDE)
    return (index * WINDOW_STRIDE).astype(int)


def f0_window(
    problem: LisaGB, f: Float[Array, ""]
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """The f0 range the window at f draws from: where a source's whole response fits inside it."""
    guard = problem.response_points // 2
    start = window_start(problem, f)
    low = jnp.maximum(start + guard, problem.f0_range[0] * problem.t_obs)
    high = jnp.minimum(
        start + problem.window_bins - guard, problem.f0_range[1] * problem.t_obs
    )
    return low / problem.t_obs, high / problem.t_obs


def window_freqs(problem: LisaGB, f: Float[Array, ""]) -> Float[Array, " F"]:
    return (window_start(problem, f) + jnp.arange(problem.window_bins)) / problem.t_obs


def single_link_noise(
    problem: LisaGB, f: Float[Array, "..."]
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """One link's optical-metrology [m^2/Hz] and test-mass-acceleration [m^2 s^-4/Hz] noise."""
    f = jnp.abs(f)
    optical_metrology = problem.oms_noise**2 * 1e-24 * (1.0 + (0.002 / f) ** 4)
    test_mass_acceleration = (
        problem.acceleration_noise**2
        * 1e-30
        * (1.0 + (0.0004 / f) ** 2)
        * (1.0 + (f / 0.008) ** 4)
    )
    return optical_metrology, test_mass_acceleration


def sky_averaged_sensitivity(
    problem: LisaGB, f: Float[Array, "..."]
) -> Float[Array, "..."]:
    """Sky-averaged strain sensitivity (arXiv:1803.01944 eq. 13), for the chart's amplitude unit only."""
    arm_length = problem.response.arm_length
    f = jnp.abs(f)
    f_star = SPEED_OF_LIGHT / (2.0 * jnp.pi * arm_length)
    p_oms, p_acc = single_link_noise(problem, f)
    return (
        (10.0 / (3.0 * arm_length**2))
        * (p_oms + 4.0 * p_acc / (2.0 * jnp.pi * f) ** 4)
        * (1.0 + 0.6 * (f / f_star) ** 2)
    )


def amplitude_unit(problem: LisaGB, f: Float[Array, ""]) -> Float[Array, ""]:
    """The helicity chart's amplitude unit at f, sqrt(sky-averaged sensitivity / t_obs): about snr 1."""
    return jnp.sqrt(sky_averaged_sensitivity(problem, f) / problem.t_obs)


def noise_psd(problem: LisaGB, f: Float[Array, "..."]) -> Float[Array, "... 3"]:
    """TDI 1.5 A/E/T noise PSD (arXiv:2108.01167), in fractional frequency like jaxgb's get_tdi."""
    arm_length = problem.response.arm_length
    f = jnp.abs(f)
    arm_phase = 2.0 * jnp.pi * arm_length / SPEED_OF_LIGHT * f
    transfer = jnp.cos(arm_phase)
    doppler = (2.0 * jnp.pi * f / SPEED_OF_LIGHT) ** 2
    p_oms, p_acc = single_link_noise(problem, f)
    optical_metrology = p_oms * doppler
    test_mass_acceleration = p_acc * doppler / (2.0 * jnp.pi * f) ** 4
    n_ae = 2.0 * test_mass_acceleration * (
        1.0 + transfer + transfer**2
    ) + 0.5 * optical_metrology * (2.0 + transfer)
    n_t = 2.0 * test_mass_acceleration * (1.0 - transfer) ** 2 + optical_metrology * (
        1.0 - transfer
    )
    n = jnp.stack([n_ae, n_ae, n_t], axis=-1)
    return n * (16.0 * jnp.sin(arm_phase) ** 2)[..., None]


def window_power(problem: LisaGB, f: Float[Array, ""]) -> Float[Array, "F 3"]:
    """Variance of each complex noise bin in the window, noise_psd * t_obs / 2."""
    return noise_psd(problem, window_freqs(problem, f)) * problem.t_obs / 2.0
