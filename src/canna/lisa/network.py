"""LisaFlow: a one-way MMDiT over source tokens and STFT data tokens, with RoPE on a shared frequency axis."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange, repeat
from jaxtyping import Array, Float, Key

from ..networks import MMDiT, FeedForward, Modulation, PiecewiseLinear
from ..networks.utils import DType
from .chart import SKY
from .problem import WINDOW_STRIDE, LisaGB

# PLE bins: STFT amplitudes reach asinh(~1000), f0 gets 4x the other coordinates (F26)
STFT_AMPLITUDE_BINS, STFT_PHASE_BINS, DATA_REACH = 128, 64, 1000.0
# channels per window: re A, E, T, im A, E, T
STFT_GROUP = 6
COORD_BINS, F0_BINS = 64, 256
# log10(1 - t) clipped to [-TIME_DECADES, 0]
TIME_DECADES = 6.0
# DiT-style sinusoids of each conditioning scalar: SINUSOID_FREQS periods, geometric over SINUSOID_DECADES down from twice its span
SINUSOID_FREQS, SINUSOID_DECADES = 64, 4.0
# |z| cos k phi and |z| sin k phi of both helicity amplitudes for k = 2..HARMONICS (k = 1 is the coordinate itself)
HARMONICS = 6
HELICITY = (slice(4, 6), slice(6, 8))
# the helicity radius asinh|z| reaches asinh(snr_range[1]) at sin iota = +-1
Z_REACH = math.asinh(1000.0)
# the finest period of f0's Fourier features, in bins
F0_FINEST_PERIOD = 1 / 64
# log f0 on [-1, 1], log Mc and the sky standardised, z± within Z_REACH
COORD_RANGES = ((-1.0, 1.0),) + ((-4.0, 4.0),) * 3 + ((-Z_REACH, Z_REACH),) * 4
# learned loss weights: one log-variance per (log10(1 - s) bin on [-LOSS_TIME_DECADES, 0], coordinate group)
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


class LisaFlow(eqx.Module):
    """x1 (or, with prediction="velocity", the velocity) of the (loudest-first) sources, from a data stream that sees only (y, f).

    prediction="residual": (x1 - xt) / σ and log σ² per coordinate, σ the flow's own scale of x1 - xt.
    """

    dtype: DType = eqx.field(static=True)
    patch: int = eqx.field(static=True)
    prediction: str = eqx.field(static=True)
    self_condition: bool = eqx.field(static=True)
    rope_periods: tuple[float, ...] = eqx.field(static=True)
    f0_periods: tuple[float, ...] = eqx.field(static=True)
    y_periods: tuple[float, ...] = eqx.field(static=True)
    c_periods: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    t_obs: float = eqx.field(static=True)
    window_bins: int = eqx.field(static=True)
    guard: int = eqx.field(static=True)
    f0_range: tuple[float, float] = eqx.field(static=True)
    x_embed: PiecewiseLinear
    f0_mlp: FeedForward
    y_embed: PiecewiseLinear
    y_mlp: FeedForward
    x_slot: Float[Array, "S D"]
    c_mlp: FeedForward
    f_mlp: FeedForward
    backbone: MMDiT
    x_modulation: Modulation
    x_unembed: FeedForward
    hat_embed: PiecewiseLinear | None
    loss_log_var: Float[Array, "B G"] | None
    s_slot: Float[Array, "N D"] | None

    def __init__(
        self,
        problem: LisaGB,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_blocks: int = 8,
        patch: int = 16,
        prediction: str = "x",
        self_condition: bool = True,
        loss_weights: bool = False,
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
        assert prediction in ("velocity", "x", "residual"), f"unknown prediction {prediction!r}"
        assert hidden_dim // num_heads % 2 == 0, "RoPE pairs a head's dims: head_dim must be even"
        # x64 is on, so a None dtype would mean float64
        assert dtype is not None and param_dtype is not None, "dtype and param_dtype must be explicit"
        self.dtype, self.patch = dtype, patch
        self.prediction = prediction
        self.self_condition = self_condition
        self.t_obs, self.window_bins = problem.t_obs, n_bins
        self.guard, self.f0_range = (
            problem.response_points // 2,
            tuple(problem.f0_range),
        )
        # RoPE periods log-spaced from 2 bins to twice the window, one per head-dim pair
        pairs = hidden_dim // num_heads // 2
        self.rope_periods = tuple(
            float(2.0 * n_bins ** (k / max(pairs - 1, 1))) for k in range(pairs)
        )
        # f0's Fourier periods: octaves from F0_FINEST_PERIOD bins to past twice the band's top bin
        top_bin = problem.f0_range[1] * problem.t_obs
        self.f0_periods = tuple(
            F0_FINEST_PERIOD * 2.0**k
            for k in range(math.ceil(math.log2(2 * top_bin / F0_FINEST_PERIOD)) + 1)
        )
        # a data token's Fourier periods: octaves from two tokens to past twice the band's top bin
        self.y_periods = tuple(
            2.0 * patch * 2.0**k
            for k in range(math.ceil(math.log2(top_bin / patch)) + 1)
        )
        # the conditioning scalars t, log10(1 - t) and log f, each over its span
        spans = (1.0, TIME_DECADES, math.log(self.f0_range[1] / self.f0_range[0]))
        self.c_periods = tuple(
            tuple(
                2.0 * span * 10.0 ** (-SINUSOID_DECADES * k / (SINUSOID_FREQS - 1))
                for k in range(SINUSOID_FREQS)
            )
            for span in spans
        )

        # eqx.nn's dtype is the parameter dtype; the compute dtype is applied in __call__
        kwargs = dict(dtype=param_dtype)
        keys = iter(jr.split(key, 10))
        edges = lambda lo, hi, bins: tuple(
            lo + (hi - lo) * i / bins for i in range(bins + 1)
        )

        coord_edges = tuple(
            edges(lo, hi, F0_BINS if i == 0 else COORD_BINS)
            for i, (lo, hi) in enumerate(COORD_RANGES)
        )
        harmonic_edges = (edges(-Z_REACH, Z_REACH, COORD_BINS),) * (4 * (HARMONICS - 1))
        self.x_embed = PiecewiseLinear(
            coord_edges + harmonic_edges, hidden_dim, key=next(keys), **kwargs
        )
        self.f0_mlp = FeedForward(
            2 * len(self.f0_periods), hidden_dim, hidden_dim, key=next(keys), **kwargs
        )
        n = patch * 3
        data_edges = (edges(0.0, math.asinh(DATA_REACH), STFT_AMPLITUDE_BINS),) * n + (
            edges(-1.0, 1.0, STFT_PHASE_BINS),
        ) * (2 * n)
        self.y_embed = PiecewiseLinear(data_edges, hidden_dim, key=next(keys), **kwargs)
        self.y_mlp = FeedForward(
            2 * len(self.y_periods), hidden_dim, hidden_dim, key=next(keys), **kwargs
        )
        # the sources are ordered, so each slot gets its own embedding
        self.x_slot = 0.02 * jr.normal(
            next(keys), (problem.n_sources, hidden_dim), param_dtype
        )

        # one MLP per conditioning vector: c over (t, log10(1 - t), log f), the data stream's over log f
        self.c_mlp = FeedForward(
            2 * SINUSOID_FREQS * len(spans),
            hidden_dim,
            hidden_dim,
            key=next(keys),
            **kwargs,
        )
        self.f_mlp = FeedForward(
            2 * SINUSOID_FREQS, hidden_dim, hidden_dim, key=next(keys), **kwargs
        )

        # no biases inside the blocks: every parameter there but the qk gain is a matrix Muon can take
        self.backbone = MMDiT(
            hidden_dim,
            num_heads,
            num_blocks,
            2,
            qk_gain=True,
            sinks=sinks > 0,
            attention_dtype=attention_dtype,
            one_way=True,
            key=next(keys),
            use_bias=False,
            **kwargs,
        )
        self.x_modulation = Modulation(hidden_dim, key=next(keys), **kwargs)
        # a residual flow also emits each coordinate's log σ²
        self.x_unembed = FeedForward(
            hidden_dim,
            hidden_dim,
            len(COORD_RANGES) * (2 if prediction == "residual" else 1),
            key=next(keys),
            **kwargs,
        )

        # the opt-in modules draw from their own key, so the ones above stay as they were
        self.hat_embed = (
            PiecewiseLinear(coord_edges, hidden_dim, key=jr.fold_in(key, 1), **kwargs)
            if self_condition
            else None
        )
        # λ starts at a fresh flow's log error per bin: x-prediction's |x1 - out|² / (1 - s)² at the bin's centre
        centre_decades = LOSS_TIME_DECADES * (
            (jnp.arange(LOSS_TIME_BINS) + 0.5) / LOSS_TIME_BINS - 1
        )
        log_var = (
            -2 * math.log(10) * centre_decades
            if prediction == "x"
            else jnp.zeros(LOSS_TIME_BINS)
        )
        # a residual flow weighs its error by its own σ, so it has no λ
        self.loss_log_var = (
            repeat(log_var, "t -> t g", g=len(LOSS_GROUPS)).astype(param_dtype)
            if loss_weights and prediction != "residual"
            else None
        )
        # attention sinks: per-draw tokens that start from c, each with its own embedding
        self.s_slot = (
            0.02 * jr.normal(jr.fold_in(key, 2), (sinks, hidden_dim), param_dtype)
            if sinks
            else None
        )

    def source_bins(
        self, x_f: Float[Array, "..."], f: Float[Array, ""]
    ) -> Float[Array, "..."]:
        """The bin of f0, from the window start, for the flow coordinate x_f (log f0 mapped onto [-1, 1])."""
        start = (
            jnp.round((f * self.t_obs - self.window_bins / 2) / WINDOW_STRIDE)
            * WINDOW_STRIDE
        )
        low = jnp.maximum(self.guard, self.f0_range[0] * self.t_obs - start)
        high = jnp.minimum(
            self.window_bins - self.guard, self.f0_range[1] * self.t_obs - start
        )
        # f0 = c exp(x_f h): c the geometric centre, h half the log span; c - start cancels start exactly
        half_log_span = 0.5 * jnp.log1p((high - low) / (start + low))
        centre = jnp.sqrt((start + low) * (start + high))
        centre_offset = (start * (low + high) + low * high) / (centre + start)
        return centre_offset + centre * jnp.expm1(x_f * half_log_span)

    def encode(
        self, y: Float[Array, "F 6"], f: Float[Array, ""]
    ) -> tuple[Float[Array, "L M H K"], Float[Array, "L M H K"]]:
        """Every block's data keys and values: a function of (y, f) alone, so one per observation."""
        assert (
            y.shape[-1] == STFT_GROUP
        ), f"expected {STFT_GROUP} data channels, got {y.shape[-1]}"
        f32 = jnp.float32
        turns_f = jnp.mod(
            (jnp.log(f) - math.log(self.f0_range[0])) / jnp.asarray(self.c_periods[2]),
            1.0,
        ).astype(f32)
        c_f = self.f_mlp(
            jnp.concat([jnp.cos(2 * jnp.pi * turns_f), jnp.sin(2 * jnp.pi * turns_f)])
        )
        tokens_y = jax.vmap(self.y_embed)(stft_tokens(y.astype(f32), self.patch))

        # each data token's centre bin: absolute for its embedding, from the window start for RoPE
        bins = (
            self.patch * jnp.arange(tokens_y.shape[0], dtype=f32) + (self.patch - 1) / 2
        )
        start = (
            jnp.round((f * self.t_obs - self.window_bins / 2) / WINDOW_STRIDE)
            * WINDOW_STRIDE
        )
        turns = jnp.mod(
            (start + bins)[:, None] / jnp.asarray(self.y_periods), 1.0
        ).astype(f32)
        tokens_y = tokens_y + jax.vmap(self.y_mlp)(
            jnp.concat(
                [jnp.cos(2 * jnp.pi * turns), jnp.sin(2 * jnp.pi * turns)], axis=-1
            )
        )
        # RoPE turns in float64, then cast: float32 angles of far bins are off by ~1e-3 rad
        angles = 2 * jnp.pi * jnp.mod(
            bins[:, None] / jnp.asarray(self.rope_periods), 1.0
        ).astype(f32)

        # mixed precision: the stored params keep param_dtype, the backbone runs in dtype
        params, static = eqx.partition(self.backbone, eqx.is_inexact_array)
        backbone = eqx.combine(
            jax.tree.map(lambda p: p.astype(self.dtype), params), static
        )
        return backbone.encode(
            tokens_y.astype(self.dtype),
            c_f.astype(self.dtype),
            (jnp.cos(angles), jnp.sin(angles)),
        )

    def __call__(
        self,
        x: Float[Array, "S X"],
        t: Float[Array, ""],
        y: Float[Array, "F 6"],
        f: Float[Array, ""],
        x_hat: Float[Array, "S X"] | None = None,
        cache: tuple[Float[Array, "L M H K"], Float[Array, "L M H K"]] | None = None,
    ) -> Float[Array, "S O"]:
        """cache: `encode(y, f)`, when the caller already has it."""
        assert (
            x.shape[0] == self.x_slot.shape[0]
        ), f"built for {self.x_slot.shape[0]} source slots, got {x.shape[0]} sources"
        cache = self.encode(y, f) if cache is None else cache
        f32 = jnp.float32
        # t on its own and as log10(1 - t), resolving the last decades before the data
        late = jnp.log10(jnp.maximum(1 - t, 10.0**-TIME_DECADES))
        scalars = jnp.stack(
            [t, late + TIME_DECADES, jnp.log(f) - math.log(self.f0_range[0])]
        )
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
        )

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
        features = jnp.concat([x, flat(jnp.cos(k_phi)), flat(jnp.sin(k_phi))], axis=-1)
        tokens_x = jax.vmap(self.x_embed)(features.astype(f32)) + self.x_slot

        # f0's bin, before any cast: absolute for its embedding, from the window start for RoPE
        bins = self.source_bins(x[:, 0], f)
        start = (
            jnp.round((f * self.t_obs - self.window_bins / 2) / WINDOW_STRIDE)
            * WINDOW_STRIDE
        )
        turns = jnp.mod(
            (start + bins)[:, None] / jnp.asarray(self.f0_periods), 1.0
        ).astype(f32)
        tokens_x = tokens_x + jax.vmap(self.f0_mlp)(
            jnp.concat(
                [jnp.cos(2 * jnp.pi * turns), jnp.sin(2 * jnp.pi * turns)], axis=-1
            )
        )
        if self.self_condition and x_hat is not None:
            lo, hi = jnp.asarray(COORD_RANGES, f32).T
            tokens_x = tokens_x + jax.vmap(self.hat_embed)(
                jnp.clip(x_hat.astype(f32), lo, hi)
            )
        angles = 2 * jnp.pi * jnp.mod(
            bins[:, None] / jnp.asarray(self.rope_periods), 1.0
        ).astype(f32)

        params, static = eqx.partition(self.backbone, eqx.is_inexact_array)
        backbone = eqx.combine(
            jax.tree.map(lambda p: p.astype(self.dtype), params), static
        )
        tokens_s = None if self.s_slot is None else c + self.s_slot
        h = backbone.decode(
            tokens_x.astype(self.dtype),
            c.astype(self.dtype),
            cache,
            (jnp.cos(angles), jnp.sin(angles)),
            None if tokens_s is None else tokens_s.astype(self.dtype),
        )
        h, _ = self.x_modulation(h.astype(f32), c)
        return jax.vmap(self.x_unembed)(h)
