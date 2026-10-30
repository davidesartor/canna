"""LisaFlow: a one-way MMDiT over source tokens, STFT data tokens and frame registers, with RoPE on a shared frequency axis."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from jaxtyping import Array, Float, Key

from ..networks import MMDiT, FeedForward, Modulation
from ..networks.utils import DType
from .chart import SKY, prior_moments
from .problem import WINDOW_STRIDE, LisaGB

# channels per window: re A, E, T, im A, E, T
STFT_GROUP = 6
# per source: [f0, fdot, l, b, z+ (2), z- (2)]
COORDS = 8
# log10(1 - t) clipped to [-TIME_DECADES, 0]
TIME_DECADES = 6.0
# DiT-style sinusoids of each conditioning scalar: SINUSOID_FREQS periods, geometric over SINUSOID_DECADES down from twice its span
SINUSOID_FREQS, SINUSOID_DECADES = 64, 4.0
# |z| cos k phi and |z| sin k phi of both helicity amplitudes for k = 2..HARMONICS (k = 1 is the coordinate itself)
HARMONICS = 6
HELICITY = (slice(4, 6), slice(6, 8))
# the finest period of the frequency features (f0, a data token's lowest and highest bin), in bins
FREQ_FINEST_PERIOD = 1 / 64
# the training diagnostics: the frame per (log10 true snr bin on LOSS_LOG10_SNR, coordinate group),
# the flow per (log10(1 - t) bin on [-LOSS_TIME_DECADES, 0], coordinate group)
LOSS_SNR_BINS, LOSS_LOG10_SNR = 16, (-1.0, 3.0)
LOSS_TIME_BINS, LOSS_TIME_DECADES = 16, 6.0
LOSS_GROUPS = (slice(0, 1), slice(1, 2), SKY) + HELICITY


def stft_tokens(y: Float[Array, "F 6"], patch: int) -> Float[Array, "N P"]:
    """Each token's `patch` bins as its band's time samples (orthonormal inverse DFT): asinh|z| and gated phase."""
    z = y[:, :3] + 1j * y[:, 3:]
    z = jnp.fft.ifft(rearrange(z, "(n p) c -> n p c", p=patch), axis=1, norm="ortho")
    phase = z * jax.lax.rsqrt(1.0 + jnp.square(jnp.abs(z)))
    flat = lambda a: rearrange(a, "n p c -> n (p c)")
    return jnp.concat(
        [flat(jnp.arcsinh(jnp.abs(z))), flat(phase.real), flat(phase.imag)], axis=-1
    ).astype(y.dtype)


Cache = tuple[Float[Array, "L M+S H K"], Float[Array, "L M+S H K"]]


def reference_frame(
    problem: LisaGB, delta: Float[Array, "S 16"], f: Float[Array, ""]
) -> tuple[Float[Array, "S 8"], Float[Array, "S 8"]]:
    """The frame's mean mu and log std of x1 per f0-ordered slot: the prior's moments, shifted and scaled by delta."""
    prior_mean, prior_std = prior_moments(problem, f)
    delta_mu, delta_s = jnp.split(delta, 2, axis=-1)
    return prior_mean + prior_std * delta_mu, jnp.log(prior_std) + delta_s


class LisaFlow(eqx.Module):
    """A frame per source from the data alone (encode), then the mean m and log std s of z1, x1 in that frame.

    The sources are slots in f0 order: the frame registers carry one slot embedding each.
    """

    dtype: DType = eqx.field(static=True)
    patch: int = eqx.field(static=True)
    rope_periods: tuple[float, ...] = eqx.field(static=True)
    freq_periods: tuple[float, ...] = eqx.field(static=True)
    c_periods: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    t_obs: float = eqx.field(static=True)
    window_bins: int = eqx.field(static=True)
    guard: int = eqx.field(static=True)
    f0_range: tuple[float, float] = eqx.field(static=True)
    x_embed: eqx.nn.Linear
    f0_mlp: FeedForward
    y_embed: eqx.nn.Linear
    y_mlp: FeedForward
    c_mlp: FeedForward
    f_mlp: FeedForward
    backbone: MMDiT
    x_modulation: Modulation
    x_unembed: FeedForward
    s_slot: Float[Array, "N D"] | None
    frame_slot: Float[Array, "S D"]
    frame_modulation: Modulation
    frame_unembed: eqx.nn.Linear

    def __init__(
        self,
        problem: LisaGB,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_blocks: int = 8,
        patch: int = 16,
        sinks: int = 4,
        dtype: DType = jnp.float32,
        attention_dtype: DType | None = None,
        param_dtype: jnp.dtype = jnp.float32,
        *,
        key: Key[Array, ""],
    ):
        n_bins = problem.window_bins
        assert (
            n_bins % patch == 0
        ), f"window of {n_bins} bins is not divisible by {patch}"
        assert (
            hidden_dim // num_heads % 2 == 0
        ), "RoPE pairs a head's dims: head_dim must be even"
        # x64 is on, so a None dtype would mean float64
        assert (
            dtype is not None and param_dtype is not None
        ), "dtype and param_dtype must be explicit"
        self.dtype, self.patch = dtype, patch
        self.t_obs, self.window_bins = problem.t_obs, n_bins
        self.guard, self.f0_range = (
            problem.response_points // 2,
            tuple(problem.f0_range),
        )
        # RoPE periods in Hz, log-spaced from 2 bins to twice the window, one per head-dim pair
        pairs = hidden_dim // num_heads // 2
        self.rope_periods = tuple(
            float(2.0 * n_bins ** (k / max(pairs - 1, 1))) / problem.t_obs
            for k in range(pairs)
        )
        # the frequency features' periods in Hz: octaves from FREQ_FINEST_PERIOD bins to past twice the band's top
        top_bin = problem.f0_range[1] * problem.t_obs
        self.freq_periods = tuple(
            FREQ_FINEST_PERIOD * 2.0**k / problem.t_obs
            for k in range(math.ceil(math.log2(2 * top_bin / FREQ_FINEST_PERIOD)) + 1)
        )
        # the conditioning scalars t and log10(1 - t), each over its span
        spans = (1.0, TIME_DECADES)
        self.c_periods = tuple(
            tuple(
                2.0 * span * 10.0 ** (-SINUSOID_DECADES * k / (SINUSOID_FREQS - 1))
                for k in range(SINUSOID_FREQS)
            )
            for span in spans
        )

        # eqx.nn's dtype is the parameter dtype; the compute dtype is applied in __call__
        kwargs = dict(dtype=param_dtype)
        keys = iter(jr.split(key, 12))
        # z, x, x's helicity harmonics, and the frame's mu and log std
        self.x_embed = eqx.nn.Linear(
            4 * COORDS + 4 * (HARMONICS - 1), hidden_dim, key=next(keys), **kwargs
        )
        # f0, a data token's and the window's lowest and highest frequency: the same features, separate MLPs
        n_freq = 2 * len(self.freq_periods)
        self.f0_mlp = FeedForward(
            n_freq, hidden_dim, hidden_dim, key=next(keys), **kwargs
        )
        self.y_embed = eqx.nn.Linear(
            3 * patch * 3, hidden_dim, key=next(keys), **kwargs
        )
        self.y_mlp = FeedForward(
            2 * n_freq, hidden_dim, hidden_dim, key=next(keys), **kwargs
        )
        # one MLP per conditioning vector: c over (t, log10(1 - t)), the data stream's c_f over the window
        self.c_mlp = FeedForward(
            2 * SINUSOID_FREQS * len(spans),
            hidden_dim,
            hidden_dim,
            key=next(keys),
            **kwargs,
        )
        self.f_mlp = FeedForward(
            2 * n_freq, hidden_dim, hidden_dim, key=next(keys), **kwargs
        )

        # no biases inside the blocks: every parameter there but the qk gain is a matrix Muon can take
        self.backbone = MMDiT(
            hidden_dim,
            num_heads,
            num_blocks,
            2,
            qk_gain=True,
            sinks=sinks > 0,
            registers=problem.n_sources,
            attention_dtype=attention_dtype,
            one_way=True,
            key=next(keys),
            use_bias=False,
            **kwargs,
        )
        self.x_modulation = Modulation(hidden_dim, key=next(keys), **kwargs)
        # the mean and log std of z1 per coordinate
        self.x_unembed = FeedForward(
            hidden_dim, hidden_dim, 2 * COORDS, key=next(keys), **kwargs
        )
        # the frame's (delta_mu, delta_s) per slot: no bias, so the zero-init modulation starts it at the prior
        self.frame_slot = 0.02 * jr.normal(
            next(keys), (problem.n_sources, hidden_dim), param_dtype
        )
        self.frame_modulation = Modulation(hidden_dim, key=next(keys), **kwargs)
        self.frame_unembed = eqx.nn.Linear(
            hidden_dim, 2 * COORDS, use_bias=False, key=next(keys), **kwargs
        )

        # the opt-in sinks draw from their own key, past split(key, 12)'s, so the modules above stay as they were
        # attention sinks: per-draw tokens that start from c, each with its own embedding
        self.s_slot = (
            0.02 * jr.normal(jr.fold_in(key, 12), (sinks, hidden_dim), param_dtype)
            if sinks
            else None
        )

    def frequency_features(self, freq: Float[Array, "..."]) -> Float[Array, "... P"]:
        """cos and sin of a frequency in Hz over freq_periods, turned in the input's precision, then cast."""
        turns = jnp.mod(freq[..., None] / jnp.asarray(self.freq_periods), 1.0)
        turns = turns.astype(jnp.float32)
        return jnp.concat(
            [jnp.cos(2 * jnp.pi * turns), jnp.sin(2 * jnp.pi * turns)], axis=-1
        )

    def window(
        self, f: Float[Array, ""]
    ) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
        """The window's first bin, and f0's lowest and highest bin from there."""
        start = (
            jnp.round((f * self.t_obs - self.window_bins / 2) / WINDOW_STRIDE)
            * WINDOW_STRIDE
        )
        low = jnp.maximum(self.guard, self.f0_range[0] * self.t_obs - start)
        high = jnp.minimum(
            self.window_bins - self.guard, self.f0_range[1] * self.t_obs - start
        )
        return start, low, high

    def source_bins(
        self, x_f: Float[Array, "..."], f: Float[Array, ""]
    ) -> Float[Array, "..."]:
        """The bin of f0, from the window start, for the flow coordinate x_f (log f0 mapped onto [-1, 1])."""
        start, low, high = self.window(f)
        # f0 = c exp(x_f h): c the geometric centre, h half the log span; c - start cancels start exactly
        half_log_span = 0.5 * jnp.log1p((high - low) / (start + low))
        centre = jnp.sqrt((start + low) * (start + high))
        centre_offset = (start * (low + high) + low * high) / (centre + start)
        return centre_offset + centre * jnp.expm1(x_f * half_log_span)

    def condition(self, f: Float[Array, ""]) -> Float[Array, " D"]:
        """c_f: the window's f0 range, its lowest and highest frequency in Hz."""
        start, low, high = self.window(f)
        edges = jnp.stack([start + low, start + high]) / self.t_obs
        return self.f_mlp(rearrange(self.frequency_features(edges), "e p -> (e p)"))

    def rope_angles(self, freq: Float[Array, "N"]) -> Float[Array, "N P"]:
        """RoPE angles of frequencies in Hz, from 0 Hz: turned in the input's precision (float64), then cast."""
        turns = jnp.mod(freq[:, None] / jnp.asarray(self.rope_periods), 1.0)
        return 2 * jnp.pi * turns.astype(jnp.float32)

    def cast_backbone(self) -> MMDiT:
        """The backbone in the compute dtype; the stored params keep param_dtype."""
        params, static = eqx.partition(self.backbone, eqx.is_inexact_array)
        return eqx.combine(jax.tree.map(lambda p: p.astype(self.dtype), params), static)

    def encode(
        self, y: Float[Array, "F 6"], f: Float[Array, ""]
    ) -> tuple[Float[Array, "S 16"], Cache]:
        """The frame's (delta_mu, delta_s) per slot, and every block's data and register keys and values.

        A function of (y, f) alone, so one per observation.
        """
        assert (
            y.shape[-1] == STFT_GROUP
        ), f"expected {STFT_GROUP} data channels, got {y.shape[-1]}"
        f32 = jnp.float32
        c_f = self.condition(f)
        tokens_y = jax.vmap(self.y_embed)(stft_tokens(y.astype(f32), self.patch))

        # each data token's lowest and highest frequency in Hz for its embedding, their mean for RoPE
        start, _, _ = self.window(f)
        lowest = self.patch * jnp.arange(tokens_y.shape[0], dtype=f32)
        band = (
            jnp.stack([start + lowest, start + lowest + self.patch - 1], -1)
            / self.t_obs
        )
        features = self.frequency_features(band)
        tokens_y = tokens_y + jax.vmap(self.y_mlp)(
            rearrange(features, "n e p -> n (e p)")
        )
        angles = self.rope_angles(band.mean(-1))

        frames = c_f + self.frame_slot
        cache, frames = self.cast_backbone().encode(
            tokens_y.astype(self.dtype),
            c_f.astype(self.dtype),
            (jnp.cos(angles), jnp.sin(angles)),
            frames.astype(self.dtype),
        )
        h, gate = self.frame_modulation(frames.astype(f32), c_f)
        return jax.vmap(self.frame_unembed)(h * gate), cache

    def __call__(
        self,
        z: Float[Array, "S X"],
        t: Float[Array, ""],
        frame: tuple[Float[Array, "S X"], Float[Array, "S X"]],
        f: Float[Array, ""],
        cache: Cache,
    ) -> Float[Array, "S O"]:
        """The mean m and log std s of z1 per coordinate, z in the frame (mu, log std) the flow does not train."""
        f32 = jnp.float32
        mu, log_sigma = jax.lax.stop_gradient(frame)
        x = mu + jnp.exp(log_sigma) * z
        # t on its own and as log10(1 - t), resolving the last decades before the data
        late = jnp.log10(jnp.maximum(1 - t, 10.0**-TIME_DECADES))
        scalars = jnp.stack([t, late + TIME_DECADES])
        turns_c = jnp.mod(scalars[:, None] / jnp.asarray(self.c_periods), 1.0).astype(
            f32
        )
        c = self.c_mlp(
            rearrange(
                jnp.concat(
                    [jnp.cos(2 * jnp.pi * turns_c), jnp.sin(2 * jnp.pi * turns_c)],
                    axis=-1,
                ),
                "s k -> (s k)",
            )
        ) + self.condition(f)

        # each helicity amplitude's higher harmonics at its radius beside the coordinates, so they vanish with it
        # (hypot and arctan2 have no gradient at z = 0: fine while nothing differentiates the flow in x)
        radius = jnp.stack(
            [jnp.hypot(x[:, b.start], x[:, b.start + 1]) for b in HELICITY], -1
        )
        phi = jnp.stack(
            [jnp.arctan2(x[:, b.start + 1], x[:, b.start]) for b in HELICITY], -1
        )
        k_phi = phi[:, :, None] * jnp.arange(2, HARMONICS + 1)
        flat = lambda a: rearrange(radius[:, :, None] * a, "s c k -> s (c k)")
        features = jnp.concat(
            [z, x, flat(jnp.cos(k_phi)), flat(jnp.sin(k_phi)), mu, log_sigma], axis=-1
        )
        tokens_x = jax.vmap(self.x_embed)(features.astype(f32))

        # f0 in Hz, before any cast, for its embedding and RoPE
        start, _, _ = self.window(f)
        f0 = (start + self.source_bins(x[:, 0], f)) / self.t_obs
        tokens_x = tokens_x + jax.vmap(self.f0_mlp)(self.frequency_features(f0))
        angles = self.rope_angles(f0)

        tokens_s = None if self.s_slot is None else c + self.s_slot
        h = self.cast_backbone().decode(
            tokens_x.astype(self.dtype),
            c.astype(self.dtype),
            cache,
            (jnp.cos(angles), jnp.sin(angles)),
            None if tokens_s is None else tokens_s.astype(self.dtype),
        )
        h, _ = self.x_modulation(h.astype(f32), c)
        return jax.vmap(self.x_unembed)(h)
