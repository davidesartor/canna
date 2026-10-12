import math

from jaxtyping import Array, Float, Key
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from einops import rearrange

from ..networks import MMDiT, FeedForward, Modulation, PiecewiseLinear, SinusoidalEmbed
from ..networks.utils import DType

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

    The x stream gets no positional embedding: that is what keeps it permutation invariant
    over sources. The y tokens get a learned one, one vector per token position.
    """

    dtype: DType = eqx.field(static=True)
    patch: int = eqx.field(static=True)
    x_embed: PiecewiseLinear
    y_embed: PiecewiseLinear
    y_pos: Float[Array, "N D"]
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
        dtype: DType = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        *,
        key: Key[Array, ""],
        **kwargs,
    ):
        (*_, x_dim), (*_, n_bins, y_dim) = x_shape, y_shape
        assert x_dim == len(COORD_RANGES), f"expected {len(COORD_RANGES)} flow coordinates"
        assert n_bins % patch == 0, f"window of {n_bins} bins is not divisible by {patch}"
        self.dtype, self.patch = dtype, patch

        # eqx.nn's dtype is the parameter dtype; the compute dtype is applied in __call__
        kwargs = dict(kwargs, dtype=param_dtype)
        keys = iter(jr.split(key, 10))

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
        self.x_unembed = FeedForward(hidden_dim, hidden_dim, x_dim, key=next(keys), **kwargs)

    def __call__(
        self,
        x: Float[Array, "S X"],
        t: Float[Array, ""],
        y: Float[Array, "F C"],
        f: Float[Array, ""],
    ) -> Float[Array, "S X"]:
        f32 = jnp.float32
        c = self.t_mlp(self.t_embed(t.astype(f32)))
        c = c + self.f_mlp(self.f_embed(jnp.log(f.astype(f32))[None]))
        x = jax.vmap(self.x_embed)(x.astype(f32))
        y = rearrange(y.astype(f32), "(n p) c -> n (p c)", p=self.patch)
        y = jax.vmap(self.y_embed)(y) + self.y_pos

        # mixed precision: the stored params keep param_dtype, the backbone runs in dtype
        params, static = eqx.partition(self.backbone, eqx.is_inexact_array)
        backbone = eqx.combine(jax.tree.map(lambda p: p.astype(self.dtype), params), static)
        x, _ = backbone(x.astype(self.dtype), y.astype(self.dtype), c.astype(self.dtype))

        x, _ = self.x_modulation(x.astype(f32), c)
        return jax.vmap(self.x_unembed)(x)
