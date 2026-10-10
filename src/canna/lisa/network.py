from jaxtyping import Array, Float, Key
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from einops import rearrange

from ..networks import (
    MMDiT,
    FeedForward,
    Modulation,
    Patchify,
    PositionalEmbed,
    SinusoidalEmbed,
    UnPatchify,
)
from ..networks.utils import DType
from .problem import WindowGeometry


class LisaFlow(eqx.Module):
    """Velocity field over a set of binaries conditioned on a whitened WDM window."""

    dtype: DType = eqx.field(static=True)
    x_embed: FeedForward
    y_patchify: Patchify
    y_pos_h: PositionalEmbed
    y_pos_w: PositionalEmbed
    t_embed: SinusoidalEmbed
    t_mlp: FeedForward
    f_embed: SinusoidalEmbed
    f_mlp: FeedForward
    backbone: MMDiT
    x_modulation: Modulation
    x_unembed: FeedForward
    y_unembed: UnPatchify

    def __init__(
        self,
        x_shape: tuple[int, int],
        y_shape: tuple[int, int, int],
        hidden_dim: int,
        num_heads: int,
        num_blocks: int,
        patch_stages: int = 1,
        expand: int = 2,
        dtype: DType = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        *,
        key: Key[Array, ""],
        **kwargs,
    ):
        (*_, x_dim), (*_, height, width, y_dim) = x_shape, y_shape

        patch = 2**patch_stages
        assert (
            height % patch == 0 and width % patch == 0
        ), f"conditioning image {height}x{width} is not divisible by {patch}"
        self.dtype = dtype

        # eqx.nn's dtype is the parameter dtype; the compute dtype is applied in __call__
        kwargs = dict(kwargs, dtype=param_dtype)
        keys = iter(jr.split(key, 12))

        # the x stream gets no positional embedding: that is what keeps it permutation
        # invariant over sources
        self.x_embed = FeedForward(
            x_dim, hidden_dim, hidden_dim, key=next(keys), **kwargs
        )
        self.y_patchify = Patchify(
            y_dim, hidden_dim, patch_stages, key=next(keys), **kwargs
        )
        self.y_pos_h = PositionalEmbed(
            hidden_dim, height // patch, axis=-3, key=next(keys), **kwargs
        )
        self.y_pos_w = PositionalEmbed(
            hidden_dim, width // patch, axis=-2, key=next(keys), **kwargs
        )
        self.t_embed = SinusoidalEmbed(hidden_dim, key=next(keys), **kwargs)
        self.t_mlp = FeedForward(
            hidden_dim, hidden_dim, hidden_dim, key=next(keys), **kwargs
        )
        self.f_embed = SinusoidalEmbed(hidden_dim, key=next(keys), **kwargs)
        self.f_mlp = FeedForward(
            hidden_dim, hidden_dim, hidden_dim, key=next(keys), **kwargs
        )

        self.backbone = MMDiT(
            hidden_dim, num_heads, num_blocks, expand, key=next(keys), **kwargs
        )

        self.x_modulation = Modulation(hidden_dim, key=next(keys), **kwargs)
        self.x_unembed = FeedForward(
            hidden_dim, hidden_dim, 2 * x_dim, key=next(keys), **kwargs
        )
        self.y_unembed = UnPatchify(
            y_dim, hidden_dim, patch_stages, key=next(keys), **kwargs
        )

    def __call__(
        self,
        x: Float[Array, "S X"],
        t: Float[Array, ""],
        y: Float[Array, "H W C"],
        f: Float[Array, ""],
    ) -> tuple[
        Float[Array, "S X"],
        Float[Array, "S X"],
        Float[Array, "H W C"],
    ]:
        # mixed precision: the stored params keep param_dtype, this forward runs in dtype
        params, static = eqx.partition(self, eqx.is_inexact_array)
        net = eqx.combine(jax.tree.map(lambda p: p.astype(self.dtype), params), static)
        x_in = x  # float32, for the position features of a PositionedLisaFlow
        x, y = x.astype(self.dtype), y.astype(self.dtype)

        c = net.t_mlp(net.t_embed(t).astype(self.dtype))
        c = c + net.f_mlp(net.f_embed(f).astype(self.dtype))
        y_width = y.shape[-2]
        x = jax.vmap(net.x_embed)(x)
        y = net.y_pos_w(net.y_pos_h(net.y_patchify(y)))
        x, y = net.add_positions(x, y, x_in, y_width, f)

        h, w = y.shape[-3], y.shape[-2]
        x, y = net.backbone(x, rearrange(y, "h w d -> (h w) d"), c)

        x, _ = net.x_modulation(x, c)
        dx, x = jnp.split(jax.vmap(net.x_unembed)(x), 2, axis=-1)

        y = rearrange(y, "(h w) d -> h w d", h=h, w=w)
        return dx, x, net.y_unembed(y)

    def add_positions(self, x, y, x_in, y_width, f):
        """Hook for PositionedLisaFlow; the plain network has no position features."""
        return x, y


class PositionedLisaFlow(LisaFlow):
    """LisaFlow with each source and each column of conditioning tokens given its position.

    The positions are in bins from the window start, through one Fourier basis for both
    streams: sin and cos of 2 pi b / P for `positions` periods P from 2 bins to twice the
    window (WindowGeometry.features). They enter through plain linear maps, so the residual
    streams carry the sinusoids themselves and a query can score a key by a weighted sum of
    cos(2 pi (b_source - b_token) / P), which peaks where a source's tokens are.

    It exists because the plain network places f0 only to ~2e-3 of its coordinate's range,
    whatever the window (F26): 0.04 bins on XS, 3 bins on B. There the source enters as a
    raw scalar and the tokens as sinusoids no finer than the window, both in bf16, where the
    token index is not even exact above 256. Here the features are computed in float32 from
    the float32 inputs, before the cast. A subclass, so that LisaFlow's fields -- and every
    checkpoint saved from it -- stay as they were.
    """

    positions: int = eqx.field(static=True)
    window: WindowGeometry = eqx.field(static=True)
    x_pos: eqx.nn.Linear
    y_pos: eqx.nn.Linear

    def __init__(
        self,
        x_shape: tuple[int, int],
        y_shape: tuple[int, int, int],
        hidden_dim: int,
        num_heads: int,
        num_blocks: int,
        patch_stages: int = 1,
        expand: int = 2,
        *,
        positions: int,
        window: WindowGeometry,
        dtype: DType = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        key: Key[Array, ""],
        **kwargs,
    ):
        super().__init__(
            x_shape,
            y_shape,
            hidden_dim,
            num_heads,
            num_blocks,
            patch_stages,
            expand,
            dtype,
            param_dtype,
            key=key,
            **kwargs,
        )
        assert positions > 0, "a PositionedLisaFlow needs at least one period"
        self.positions, self.window = positions, window
        key_x, key_y = jr.split(jr.fold_in(key, 1))
        kwargs = dict(kwargs, dtype=param_dtype)
        self.x_pos = eqx.nn.Linear(2 * positions, hidden_dim, key=key_x, **kwargs)
        self.y_pos = eqx.nn.Linear(2 * positions, hidden_dim, key=key_y, **kwargs)

    def add_positions(self, x, y, x_in, y_width, f):
        # x_in is the float32 input: the f0 coordinate is its first column
        patch = 2 ** len(self.y_patchify.linears)
        x_bins = self.window.source_bins(x_in[:, 0].astype(jnp.float32), f)
        y_bins = self.window.token_bins(y_width // patch, patch)
        x_pos = jax.vmap(self.x_pos)(self.window.features(x_bins, self.positions))
        y_pos = jax.vmap(self.y_pos)(self.window.features(y_bins, self.positions))
        # the token grid is (h w): the frequency columns are the same in every time row
        return x + x_pos.astype(x.dtype), y + y_pos.astype(y.dtype)[None]
