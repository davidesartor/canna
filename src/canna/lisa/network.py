import math

from jaxtyping import Array, Float, Key
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from einops import rearrange

from ..networks import MMDiT, FeedForward, Modulation, PiecewiseLinear, SinusoidalEmbed
from ..networks.utils import DType
from .problem import WindowGeometry

# PLE bins. The data are whitened to unit noise per real and imaginary part, and reach
# ~700 sigma for the loudest sources on XS; their edges are uniform in arcsinh out to
# DATA_REACH, so a bin is 0.24 sigma wide near the noise and ~70 sigma wide at 300 sigma.
# Quantile edges would leave everything from ~80 sigma up in one bin. f0 gets four times
# the bins of the other coordinates: it is the one the posterior resolves most finely
# against its prior range (F26).
DATA_BINS, DATA_REACH = 64, 1000.0
COORD_BINS, F0_BINS, WINDOW_BINS = 64, 256, 64
# the flow coordinates' ranges: log f0 and log A are mapped onto [-1, 1], log Mc is
# standardised, and the sky, orientation and phase are unit vectors
COORD_RANGES = ((-1.0, 1.0), (-4.0, 4.0)) + ((-1.0, 1.0),) * 9


def uniform_edges(low: float, high: float, bins: int) -> tuple[float, ...]:
    return tuple(low + (high - low) * i / bins for i in range(bins + 1))


def arcsinh_edges(reach: float, bins: int) -> tuple[float, ...]:
    """Edges symmetric about 0 and uniform in arcsinh, out to +-reach."""
    return tuple(math.sinh(u) for u in uniform_edges(-math.asinh(reach), math.asinh(reach), bins))


class LisaFlow(eqx.Module):
    """Velocity field over a set of binaries, conditioned on a whitened frequency-domain window.

    The data are the window's rfft bins, whitened, with the real and imaginary parts of A, E
    and T stacked as six channels. `patch` neighbouring bins make one token. Every scalar
    the network takes in -- each data value, each source coordinate, the window frequency --
    goes through a piecewise linear encoding (`PiecewiseLinear`) and one linear map.

    Precision: the embeddings, the conditioning and the velocity head run in float32 and
    only the MMDiT backbone in `dtype`. The arcsinh(WDM) input of the earlier networks was
    cast to bf16 before its first layer, which rounded the loudest pixels by 0.5-3.6 noise
    sigmas rms; here the cast comes after the encoding.

    The sources come loudest first (LisaGB.loudest_first), so each source slot gets a learned
    embedding of its own, as each y token gets one for its position.

    Two heads read the source stream: the velocity, and the reconstructed parameters x1 (the
    auxiliary target: at t = 0 it is the posterior mean of each slot's parameters).
    """

    dtype: DType = eqx.field(static=True)
    patch: int = eqx.field(static=True)
    # rotary position embedding in frequency (optional): the window's geometry, to place each
    # source and each data token in bins, and the rotation periods, in bins
    window: WindowGeometry | None = eqx.field(static=True)
    rope_periods: tuple[float, ...] | None = eqx.field(static=True)
    x_embed: PiecewiseLinear
    y_embed: PiecewiseLinear
    y_pos: Float[Array, "N D"]
    x_slot: Float[Array, "S D"] | None
    t_embed: SinusoidalEmbed
    t_mlp: FeedForward
    f_embed: PiecewiseLinear
    f_mlp: FeedForward
    backbone: MMDiT
    x_modulation: Modulation
    x_unembed: FeedForward

    def __init__(
        self,
        x_shape: tuple[int, int],
        y_shape: tuple[int, int],
        hidden_dim: int,
        num_heads: int,
        num_blocks: int,
        f_range: tuple[float, float],
        patch: int = 2,
        expand: int = 2,
        slots: bool = True,
        rope: bool = False,
        window: WindowGeometry | None = None,
        rope_min_period: float = 2.0,
        dtype: DType = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        *,
        key: Key[Array, ""],
        **kwargs,
    ):
        (n_sources, x_dim), (*_, n_bins, y_dim) = x_shape, y_shape
        assert x_dim == len(COORD_RANGES), f"expected {len(COORD_RANGES)} flow coordinates"
        assert n_bins % patch == 0, f"window of {n_bins} bins is not divisible by {patch}"
        self.dtype, self.patch = dtype, patch
        # RoPE: one angle per token and per pair of a head's dimensions, periods log-spaced from
        # rope_min_period bins to twice the window, so attention resolves f0 - f_token from a
        # fraction of a bin up to the whole window
        assert not rope or window is not None, "RoPE needs the window geometry"
        self.window = window if rope else None
        pairs = hidden_dim // num_heads // 2
        self.rope_periods = (
            tuple(
                float(rope_min_period * (2.0 * window.window_bins / rope_min_period) ** (k / max(pairs - 1, 1)))
                for k in range(pairs)
            )
            if rope
            else None
        )

        # eqx.nn's dtype is the parameter dtype; the compute dtype is applied in __call__
        kwargs = dict(kwargs, dtype=param_dtype)
        keys = iter(jr.split(key, 11))

        coord_edges = tuple(
            uniform_edges(lo, hi, F0_BINS if i == 0 else COORD_BINS)
            for i, (lo, hi) in enumerate(COORD_RANGES)
        )
        self.x_embed = PiecewiseLinear(coord_edges, hidden_dim, key=next(keys), **kwargs)
        data_edges = (arcsinh_edges(DATA_REACH, DATA_BINS),) * (patch * y_dim)
        self.y_embed = PiecewiseLinear(data_edges, hidden_dim, key=next(keys), **kwargs)
        self.y_pos = 0.02 * jr.normal(
            next(keys), (n_bins // patch, hidden_dim), param_dtype
        )
        # a learned embedding per source slot, for ordered sources; without it the source
        # stream is permutation equivariant, as an unordered set needs
        key_slot = next(keys)
        self.x_slot = (
            0.02 * jr.normal(key_slot, (n_sources, hidden_dim), param_dtype) if slots else None
        )

        self.t_embed = SinusoidalEmbed(hidden_dim, key=next(keys), **kwargs)
        self.t_mlp = FeedForward(hidden_dim, hidden_dim, hidden_dim, key=next(keys), **kwargs)
        log_f = (uniform_edges(math.log(f_range[0]), math.log(f_range[1]), WINDOW_BINS),)
        self.f_embed = PiecewiseLinear(log_f, hidden_dim, key=next(keys), **kwargs)
        self.f_mlp = FeedForward(hidden_dim, hidden_dim, hidden_dim, key=next(keys), **kwargs)

        # no biases inside the blocks: every parameter there is a matrix Muon can take
        self.backbone = MMDiT(
            hidden_dim, num_heads, num_blocks, expand, key=next(keys), use_bias=False, **kwargs
        )

        self.x_modulation = Modulation(hidden_dim, key=next(keys), **kwargs)
        self.x_unembed = FeedForward(
            hidden_dim, hidden_dim, 2 * x_dim, key=next(keys), **kwargs
        )

    def __call__(
        self,
        x: Float[Array, "S X"],
        t: Float[Array, ""],
        y: Float[Array, "F C"],
        f: Float[Array, ""],
    ) -> tuple[Float[Array, "S X"], Float[Array, "S X"]]:
        assert self.x_slot is None or x.shape[0] == self.x_slot.shape[0], (
            f"built for {self.x_slot.shape[0]} source slots, got {x.shape[0]} sources"
        )
        f32 = jnp.float32
        x_in = x  # the input precision, for the sources' positions
        c = self.t_mlp(self.t_embed(t.astype(f32)))
        c = c + self.f_mlp(self.f_embed(jnp.log(f.astype(f32))[None]))
        x = jax.vmap(self.x_embed)(x.astype(f32))
        if self.x_slot is not None:
            x = x + self.x_slot
        y = rearrange(y.astype(f32), "(n p) c -> n (p c)", p=self.patch)
        y = jax.vmap(self.y_embed)(y) + self.y_pos

        # mixed precision: the stored params keep param_dtype, the backbone runs in dtype
        params, static = eqx.partition(self.backbone, eqx.is_inexact_array)
        backbone = eqx.combine(jax.tree.map(lambda p: p.astype(self.dtype), params), static)
        rope = None
        if self.rope_periods is not None:
            # positions in bins from the window start, computed before any cast: each source at
            # its current f0, each data token at its centre bin
            x_bins = self.window.source_bins(x_in[:, 0], f).astype(f32)
            y_bins = self.window.token_bins(y.shape[0], self.patch)
            bins = jnp.concat([x_bins, y_bins])
            angles = 2 * jnp.pi * bins[:, None] / jnp.asarray(self.rope_periods, f32)[None, :]
            rope = (jnp.cos(angles), jnp.sin(angles))
        x, _ = backbone(x.astype(self.dtype), y.astype(self.dtype), c.astype(self.dtype), rope)

        x, _ = self.x_modulation(x.astype(f32), c)
        dx, x1 = jnp.split(jax.vmap(self.x_unembed)(x), 2, axis=-1)
        return dx, x1
