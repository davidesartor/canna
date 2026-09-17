from typing import Optional
import math
import warnings

from jaxtyping import Array, Complex, Float, Int, Key
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

import lisaorbits
from jaxgb import jaxgb
from .wdm import from_freq_to_wdm_band
from .priors import ChirpMass, fdot_from_chirp_mass
from . import geometries
from .constants import EARTH_ORBIT_SPEED, SPEED_OF_LIGHT

MIN_RESPONSE_POINTS = 256  # enough time samples to resolve the orbital modulation

# rho^2 = SKY_AVERAGED_SNR_SQUARED * amplitude^2 * t_obs / sky_averaged_sensitivity(f) for a
# monochromatic source averaged over sky, polarisation, inclination and initial phase. The
# frequency dependence is carried by the sensitivity; this constant is the leftover O(1) that
# depends on how the response and the sensitivity are each normalised, so it is measured against
# our own `snr` rather than taken from a paper -- see tests/lisa/test_amplitude_prior.py, which
# re-derives it by Monte Carlo. It is flat to 0.1% below ~1.5 mHz and drifts ~7% by 12 mHz, which
# is the Robson-Cornish-Liu (1 + 0.6 x^2) approximation running out, not the response.
SKY_AVERAGED_SNR_SQUARED = 3.225


def response_span(f0: float, chirp_mass: float, t_obs: float) -> int:
    """Bins the response of a source at f0 covers.

    The window is centred on f0, so half of it has to hold the annual doppler sideband
    plus the whole (upward) chirp drift, and the span is twice that.
    """
    drift = float(fdot_from_chirp_mass(chirp_mass, f0)) * t_obs
    doppler = f0 * EARTH_ORBIT_SPEED / SPEED_OF_LIGHT
    return 2 * math.ceil((doppler + drift) * t_obs)


def max_f0(response_points: int, chirp_mass: float, t_obs: float) -> float:
    """Highest f0 whose `response_span` still fits in a `response_points` window.

    The inverse of `response_span` at fixed chirp mass. Writing the chirp as
    fdot = K Mc^(5/3) f0^(11/3) and the doppler as beta f0, a budget of N bins asks for
    the root of

        beta f0 + K Mc^(5/3) t_obs f0^(11/3) = N / (2 t_obs)

    which in u = f0^(1/3) is a degree-11 trinomial, so it has no solution in radicals and
    is taken numerically here. Each term alone does invert, giving the doppler-only root
    N c / (2 t_obs v) and the chirp-only root (N / 2 K Mc^(5/3) t_obs^2)^(3/11). The true
    root sits below both, and the left side is convex and increasing, so Newton started at
    the smaller of the two descends monotonically onto it.

    Note how differently the two scale: the doppler root is linear in N, the chirp one goes
    as N^(3/11), so once the chirp dominates -- which it does across most of the LISA band --
    quadrupling the window buys only 45% more bandwidth.
    """
    # `needed` rounds the span up to a power of two, so a budget that is not one buys
    # nothing above the power of two below it
    budget = 1 << (response_points.bit_length() - 1)
    beta = EARTH_ORBIT_SPEED / SPEED_OF_LIGHT
    a = float(fdot_from_chirp_mass(chirp_mass, 1.0)) * t_obs
    b = (budget // 2) / t_obs

    f0 = min(b / beta, (b / a) ** (3 / 11))
    for _ in range(64):
        step = (beta * f0 + a * f0 ** (11 / 3) - b) / (
            beta + (11 / 3) * a * f0 ** (8 / 3)
        )
        f0 -= step
        if abs(step) <= 1e-14 * f0:
            break

    # the span lands exactly on the budget at the root, so the last ulp of the solve is
    # what decides which side of the ceil() it falls on. Walk down until it agrees.
    while response_span(f0, chirp_mass, t_obs) > budget:
        f0 = math.nextafter(f0, 0.0)
    return f0


class LisaGB(eqx.Module):
    """Galactic binaries seen by LISA in one sliding frequency window, as an unordered set."""

    n_sources: int = eqx.field(static=True, default=4)
    t_obs: float = eqx.field(static=True, default=2 * 365.25 * 24 * 60 * 60)  # 2yr [s]
    sampling_step: float = eqx.field(static=True, default=0.25)  # [s]
    # the conditioning image is exactly wdm_times x wdm_freq_bands pixels
    wdm_freq_bands: int = eqx.field(static=True, default=256)
    wdm_times: int = eqx.field(static=True, default=32)
    patch_downsample: int = eqx.field(static=True, default=2)

    orbit: lisaorbits.Orbits = eqx.field(
        static=True, default=lisaorbits.EqualArmlengthOrbits()
    )
    # None sizes the response window from t_obs and the priors; giving it instead makes
    # the window the budget and caps f0_range[1] at the band it can hold
    response_points: Optional[int] = eqx.field(static=True, default=None)
    oms_noise: float = eqx.field(static=True, default=15.0)
    acceleration_noise: float = eqx.field(static=True, default=3.0)

    # white dwarf component masses, from which the chirp mass prior is built
    mass_range: tuple[float, float] = eqx.field(static=True, default=(0.15, 1.4))
    # the upper end is a request, not a promise: a given response_points lowers it to
    # whatever the doppler sideband and the chirp drift leave room for
    f0_range: tuple[float, float] = eqx.field(static=True, default=(1e-4, 12.0e-3))
    # the amplitude prior is a matched-filter snr range, not a fixed amplitude box: the same
    # amplitude is a very different source at 0.1 and at 10 mHz, so a fixed box would train on
    # undetectable sources at one end of the band and skip the loud ones at the other. This is
    # how Strub et al. 2022 (arXiv:2204.04467) set their amplitude search box too.
    snr_range: tuple[float, float] = eqx.field(static=True, default=(7.0, 1000.0))

    # built once here: its whitening quadrature must not be re-run under every trace
    chirp_mass_prior: ChirpMass = eqx.field(init=False)
    # f0 and amplitude are log-uniform boxes, the chirp mass is log-normal-ish, the two
    # angle pairs are spheres, the initial phase a circle
    geometry: geometries.Set = eqx.field(init=False)
    # static too: JaxGB is a plain object, not a pytree, so it cannot be traced
    response: jaxgb.JaxGB = eqx.field(init=False, static=True)

    def __post_init__(self):
        self.chirp_mass_prior = ChirpMass(
            m_min=self.mass_range[0], m_max=self.mass_range[1]
        )
        self.geometry = geometries.Set(
            geometries.Product(
                geometries.Bounded(1),
                geometries.Euclidean(1),
                geometries.Bounded(1),
                geometries.Spherical(3),
                geometries.Spherical(3),
                geometries.Spherical(2),
            )
        )

        # the window and the band it can hold determine each other, and which way round
        # depends on which one is actually free. Left to itself the window follows the
        # priors; pinned to what a GPU can afford, the priors follow the window instead
        if self.response_points is None:
            span = response_span(
                self.f0_range[1], self.chirp_mass_range[1], self.t_obs
            )
            self.response_points = 1 << (
                max(span, MIN_RESPONSE_POINTS) - 1
            ).bit_length()
        else:
            if self.response_points < MIN_RESPONSE_POINTS:
                # the floor is what the auto-sizing picks, not a hard bound: a window
                # this narrow still carries the right total power, but it samples the
                # response's own modulation FFT critically rather than generously, so
                # per-bin amplitudes pick up aliasing -- ~10-25% at the top of the band
                # a 16-bin window can hold, against a 1024-bin reference
                warnings.warn(
                    f"response_points={self.response_points} is below the "
                    f"{MIN_RESPONSE_POINTS}-bin floor the auto-sizing would pick, so "
                    f"the orbital modulation is critically sampled and per-bin "
                    f"amplitudes are only approximate. Total power is unaffected.",
                    stacklevel=2,
                )
            ceiling = max_f0(self.response_points, self.chirp_mass_range[1], self.t_obs)
            if ceiling < self.f0_range[1]:
                warnings.warn(
                    f"response_points={self.response_points} holds a "
                    f"Mc={self.chirp_mass_range[1]:.2f}Msun source only up to "
                    f"{ceiling:.4e}Hz over t_obs={self.t_obs:.3e}s, so the f0 prior "
                    f"stops there rather than at {self.f0_range[1]:.4e}Hz. "
                    f"Raise response_points to train on the whole band.",
                    stacklevel=2,
                )
                self.f0_range = (self.f0_range[0], ceiling)
            assert self.f0_range[0] < self.f0_range[1], (
                f"response_points={self.response_points} leaves no band at all: it "
                f"holds a Mc={self.chirp_mass_range[1]:.2f}Msun source only up to "
                f"{ceiling:.4e}Hz over t_obs={self.t_obs:.3e}s, at or below "
                f"f0_range[0]={self.f0_range[0]:.4e}Hz."
            )

        # whichever way round it went, the window has to hold the band it ended up with.
        # the floor is deliberately not applied here: it picks a default, and an
        # explicit budget below it is the caller's call, warned about above
        span = response_span(self.f0_range[1], self.chirp_mass_range[1], self.t_obs)
        needed = 1 << (max(span, 1) - 1).bit_length()
        assert needed <= self.response_points, (
            f"response_points={self.response_points} too narrow for a "
            f"Mc={self.chirp_mass_range[1]:.2f}Msun source at "
            f"f0_max={self.f0_range[1]:.1e}Hz over t_obs={self.t_obs:.3e}s: "
            f"it needs {needed} bins."
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

        assert (
            self.wdm_times % 2 == 0 and self.wdm_times % self.patch_downsample == 0
        ), (
            f"wdm_times={self.wdm_times} must be even and a multiple of "
            f"patch_downsample={self.patch_downsample}."
        )
        assert self.wdm_freq_bands % self.patch_downsample == 0, (
            f"wdm_freq_bands={self.wdm_freq_bands} must be a multiple of "
            f"patch_downsample={self.patch_downsample}."
        )
        # the window has to hold a whole response plus room to slide it around
        assert self.window_bins > 2 * self.response_points, (
            f"window is {self.window_bins} bins "
            f"(wdm_freq_bands={self.wdm_freq_bands} * wdm_times//2), too narrow for a "
            f"response_points={self.response_points} source. "
            f"Raise wdm_freq_bands or wdm_times."
        )
        lo, hi = self.window_index_range
        assert lo <= hi, (
            f"no valid window position: index range is [{lo}, {hi}]. "
            f"The {self.window_bins}-bin window overruns the "
            f"f0_range={self.f0_range} band at t_obs={self.t_obs:.3e}s."
        )

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
    def window_bins(self) -> int:
        return self.wdm_freq_bands * (self.wdm_times // 2)

    @property
    def window_index_range(self) -> tuple[int, int]:
        # window_start is index * wdm_times, so its wdm channel offset stays even
        # and positive; the two ends are where the usable interior just covers f0_range
        guard = self.response_points // 2
        lo = math.floor((self.f0_range[0] * self.t_obs - guard) / self.wdm_times)
        hi = math.ceil(
            (self.f0_range[1] * self.t_obs + guard - self.window_bins) / self.wdm_times
        )
        return max(lo, 1), max(hi, max(lo, 1))

    def sample_f(self, key: Key[Array, ""]) -> Float[Array, ""]:
        """Centre frequency of a log-uniformly slid window, so it tracks the f0 prior."""
        u = jr.uniform(key)
        f_centre = self.f0_range[0] * (self.f0_range[1] / self.f0_range[0]) ** u
        lo, hi = self.window_index_range
        index = jnp.clip(
            jnp.round(f_centre * self.t_obs / self.wdm_times - self.wdm_freq_bands / 4),
            lo,
            hi,
        )
        return (index * self.wdm_times + self.window_bins / 2) / self.t_obs

    def window_start(self, f: Float[Array, ""]) -> Int[Array, ""]:
        """First rfft bin of the window centred at f."""
        index = jnp.round((f * self.t_obs - self.window_bins / 2) / self.wdm_times)
        return index.astype(int) * self.wdm_times

    def f0_window(
        self, f: Float[Array, ""]
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        # a source is only drawn where its whole response fits inside the window
        guard = self.response_points // 2
        start = self.window_start(f)
        low = jnp.maximum(start + guard, self.f0_range[0] * self.t_obs)
        high = jnp.minimum(
            start + self.window_bins - guard, self.f0_range[1] * self.t_obs
        )
        return low / self.t_obs, high / self.t_obs

    def window_freqs(self, f: Float[Array, ""]) -> Float[Array, " F"]:
        return (self.window_start(f) + jnp.arange(self.window_bins)) / self.t_obs

    def physical_to_flow(
        self, p: Float[Array, "... S 8"], f: Float[Array, ""]
    ) -> Float[Array, "... S 11"]:
        f0, mc, amp, sky_lon, sky_lat, psi, iota, phi0 = jnp.split(p, 8, axis=-1)
        log_f = jnp.log(jnp.stack(self.f0_window(f)))
        log_a = jnp.log(jnp.stack(self.a_window(f)))

        # whiten each log block, lift the two angle pairs onto their spheres and
        # the initial phase onto its circle
        f = (2 * jnp.log(f0) - log_f.sum()) / (log_f[1] - log_f[0])
        m = (
            jnp.log(mc) - self.chirp_mass_prior.log_mean
        ) / self.chirp_mass_prior.log_std
        a = (2 * jnp.log(amp) - log_a.sum()) / (log_a[1] - log_a[0])
        return jnp.concat(
            [
                f,
                m,
                a,
                jnp.cos(sky_lat) * jnp.cos(sky_lon),
                jnp.cos(sky_lat) * jnp.sin(sky_lon),
                jnp.sin(sky_lat),
                jnp.cos(iota) * jnp.cos(psi),
                jnp.cos(iota) * jnp.sin(psi),
                jnp.sin(iota),
                jnp.cos(phi0),
                jnp.sin(phi0),
            ],
            axis=-1,
        )

    def flow_to_physical(
        self, x: Float[Array, "... S 11"], f: Float[Array, ""]
    ) -> Float[Array, "... S 8"]:
        log_f = jnp.log(jnp.stack(self.f0_window(f)))
        log_a = jnp.log(jnp.stack(self.a_window(f)))
        f, m, a, sx, sy, sz, ox, oy, oz, cos, sin = jnp.split(x, 11, axis=-1)

        f0 = jnp.exp((f * (log_f[1] - log_f[0]) + log_f.sum()) / 2)
        mc = jnp.exp(m * self.chirp_mass_prior.log_std + self.chirp_mass_prior.log_mean)
        amp = jnp.exp((a * (log_a[1] - log_a[0]) + log_a.sum()) / 2)
        return jnp.concat(
            [
                f0,
                mc,
                amp,
                jnp.mod(jnp.arctan2(sy, sx), 2 * jnp.pi),
                jnp.arctan2(sz, jnp.hypot(sx, sy)),
                jnp.mod(jnp.arctan2(oy, ox), 2 * jnp.pi),
                jnp.arctan2(oz, jnp.hypot(ox, oy)),
                jnp.mod(jnp.arctan2(sin, cos), 2 * jnp.pi),
            ],
            axis=-1,
        )

    # sliding the window moves the whitening, not the manifold
    def exp_map(
        self, x: Float[Array, "S 11"], v: Float[Array, "S 11"]
    ) -> Float[Array, "S 11"]:
        return self.geometry.exp_map(x, v)

    def log_map(
        self, x0: Float[Array, "S 11"], x1: Float[Array, "S 11"]
    ) -> Float[Array, "S 11"]:
        return self.geometry.log_map(x0, x1)

    def sample_physical(
        self, key: Key[Array, ""], f: Float[Array, ""]
    ) -> Float[Array, "S 8"]:
        keys = jr.split(key, 8)
        shape = (self.n_sources, 1)
        log_f = jnp.log(jnp.stack(self.f0_window(f)))
        log_a = jnp.log(jnp.stack(self.a_window(f)))

        f0 = jnp.exp(jr.uniform(keys[0], shape, minval=log_f[0], maxval=log_f[1]))
        mc = eqx.filter_vmap(self.chirp_mass_prior)(jr.split(keys[1], self.n_sources))
        amp = jnp.exp(jr.uniform(keys[2], shape, minval=log_a[0], maxval=log_a[1]))

        # isotropic on the sphere: longitude uniform, sine of latitude uniform
        sky_lon = jr.uniform(keys[3], shape, maxval=2 * jnp.pi)
        sky_lat = jnp.arcsin(jr.uniform(keys[4], shape, minval=-1.0, maxval=1.0))
        psi = jr.uniform(keys[5], shape, maxval=2 * jnp.pi)
        iota = jnp.arcsin(jr.uniform(keys[6], shape, minval=-1.0, maxval=1.0))
        phi0 = jr.uniform(keys[7], shape, maxval=2 * jnp.pi)
        return jnp.concat([f0, mc, amp, sky_lon, sky_lat, psi, iota, phi0], axis=-1)

    def sample_flow(
        self, key: Key[Array, ""], f: Float[Array, ""]
    ) -> Float[Array, "S 11"]:
        return self.physical_to_flow(self.sample_physical(key, f), f)

    def clean_signal(
        self, p: Float[Array, "... S 8"], f: Float[Array, ""]
    ) -> Complex[Array, "... F 3"]:
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

            # place each segment in its positive band on the window's one-sided grid
            start = self.response.get_kmin(sources[:, 0]) - self.window_start(f)
            idxs = start[:, None] + jnp.arange(self.response_points)
            full = jnp.zeros((self.window_bins, 3), dtype=segments.dtype)
            full = full.at[idxs].add(segments)
            return full

        return jnp.vectorize(combine, signature="(s,d)->(f,c)")(p)

    def single_link_noise(
        self, f: Float[Array, "..."]
    ) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        """Displacement noises of one link: optical metrology and test-mass acceleration.

        In m^2/Hz and m^2 s^-4/Hz respectively, before any interferometric response. Both
        `noise_psd` and `sky_averaged_sensitivity` are built from these, so the two cannot
        drift apart when `oms_noise` or `acceleration_noise` change.
        """
        f = jnp.abs(f)
        optical_metrology = self.oms_noise**2 * 1e-24 * (1.0 + (0.002 / f) ** 4)
        test_mass_acceleration = (
            self.acceleration_noise**2
            * 1e-30
            * (1.0 + (0.0004 / f) ** 2)
            * (1.0 + (f / 0.008) ** 4)
        )
        return optical_metrology, test_mass_acceleration

    def sky_averaged_sensitivity(self, f: Float[Array, "..."]) -> Float[Array, "..."]:
        """One-sided sky-averaged strain sensitivity, Robson, Cornish & Liu 2019 eq. 13.

        arXiv:1803.01944. This is the response-divided-out companion to `noise_psd`: the
        latter whitens the data, this one says how loud a source of a given amplitude is,
        and it is only ever used to set the amplitude prior. Its `1 + 0.6 (f/f_*)^2`
        approximation is what makes `SKY_AVERAGED_SNR_SQUARED` drift above ~5 mHz.
        """
        arm_length = self.response.arm_length
        f = jnp.abs(f)
        f_star = SPEED_OF_LIGHT / (2.0 * jnp.pi * arm_length)
        p_oms, p_acc = self.single_link_noise(f)
        return (
            (10.0 / (3.0 * arm_length**2))
            * (p_oms + 4.0 * p_acc / (2.0 * jnp.pi * f) ** 4)
            * (1.0 + 0.6 * (f / f_star) ** 2)
        )

    def a_window(
        self, f: Float[Array, ""]
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        """Amplitude bounds that `snr_range` implies at the window's centre frequency.

        The window is narrow enough that the sensitivity is flat across it, so the bound is
        taken at the centre rather than at each source's own f0 -- which also keeps the flow
        whitening a function of the conditioning window alone.
        """
        scale = jnp.sqrt(
            self.sky_averaged_sensitivity(f)
            / (SKY_AVERAGED_SNR_SQUARED * self.t_obs)
        )
        return self.snr_range[0] * scale, self.snr_range[1] * scale

    def noise_psd(self, f: Float[Array, "..."]) -> Float[Array, "... 3"]:
        # first-generation TDI (1.5) noise PSD, stacked A/E/T on the last axis
        # Babak, Hewitson & Petiteau 2021, arXiv:2108.01167
        #
        # Fractional frequency, which is what jaxgb's get_tdi returns: a link
        # measures a doppler shift, so a displacement noise enters it as
        # (2 pi f / c)^2 in power and an acceleration noise picks up a further
        # 1/(2 pi f)^4. Writing these as fractional lengths instead (dividing by
        # the arm rather than by c/2 pi f) leaves the whitening out by
        # (2 pi f L / c)^2 -- an f^2 tilt across the band, not a constant, so it
        # would silently re-weight low-frequency sources against high-frequency ones.
        arm_length = self.response.arm_length
        f = jnp.abs(f)
        arm_phase = 2.0 * jnp.pi * arm_length / SPEED_OF_LIGHT * f
        transfer = jnp.cos(arm_phase)
        doppler = (2.0 * jnp.pi * f / SPEED_OF_LIGHT) ** 2
        p_oms, p_acc = self.single_link_noise(f)
        optical_metrology = p_oms * doppler
        test_mass_acceleration = p_acc * doppler / (2.0 * jnp.pi * f) ** 4
        n_ae = 2.0 * test_mass_acceleration * (
            1.0 + transfer + transfer**2
        ) + 0.5 * optical_metrology * (2.0 + transfer)
        n_t = 2.0 * test_mass_acceleration * (
            1.0 - transfer
        ) ** 2 + optical_metrology * (1.0 - transfer)

        n = jnp.stack([n_ae, n_ae, n_t], axis=-1)
        tdi15_factor = 16.0 * jnp.sin(arm_phase) ** 2
        return n * tdi15_factor[..., None]

    def sample_observation(
        self,
        key: Key[Array, ""],
        p: Float[Array, "... S 8"],
        f: Float[Array, ""],
    ) -> Complex[Array, "... F 3"]:
        spectra = self.clean_signal(p, f)
        power = self.noise_psd(self.window_freqs(f)) * self.t_obs / 2.0
        real, imag = jr.normal(key, (2, *spectra.shape)) * jax.lax.rsqrt(2.0)
        return spectra + jnp.sqrt(power) * (real + 1j * imag)

    def snr(
        self, p: Float[Array, "... S 8"], f: Float[Array, ""]
    ) -> Float[Array, "..."]:
        """Matched-filter snr, rho = sqrt(<h|h>)."""
        # the noise inner product is <x|y> = 4 Re sum_k x_k y_k^* / (S_k t_obs), so with
        # power = S_k t_obs / 2 the sum below is <h|h>/2 and needs the factor 2 back.
        # equivalently rho^2 = 2 [log_likelihood(h) - log_likelihood(0)]
        spectra = self.clean_signal(p, f)
        power = self.noise_psd(self.window_freqs(f)) * self.t_obs / 2.0
        integrand = 2.0 * jnp.abs(spectra) ** 2 / power
        return jnp.sqrt(jnp.sum(integrand, axis=(-2, -1)))

    def log_likelihood(
        self,
        p: Float[Array, "... S 8"],
        o: Complex[Array, "... F 3"],
        f: Float[Array, ""],
    ) -> Float[Array, "..."]:
        power = self.noise_psd(self.window_freqs(f)) * self.t_obs / 2.0
        residual = o - self.clean_signal(p, f)
        return -jnp.sum(jnp.abs(residual) ** 2 / power, axis=(-2, -1))

    def preprocess(
        self, o: Complex[Array, "... F 3"], f: Float[Array, ""]
    ) -> Float[Array, "... t f 3"]:
        # whiten the datastream by the nominal noise psd
        power = self.noise_psd(self.window_freqs(f)) * self.t_obs / 2.0
        o = o * jax.lax.rsqrt(power)

        # the window already spans exactly wdm_freq_bands channels of nt//2 bins, and
        # window_start is snapped so the true channel offset is even and positive, which
        # is all mmin is read for; the transform runs on the last axis, so the tdi
        # channels ride along as a batch axis
        wdm = from_freq_to_wdm_band(
            jnp.moveaxis(o, -1, -2),
            ntimes=self.wdm_times,
            nfreq_bands=self.wdm_freq_bands,
            mmin=2,
        )

        # compress dynamic range for better training stability
        return jnp.arcsinh(jnp.moveaxis(wdm, -3, -1))
