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
        x, y = x.astype(self.dtype), y.astype(self.dtype)

        c = net.t_mlp(net.t_embed(t).astype(self.dtype))
        c = c + net.f_mlp(net.f_embed(f).astype(self.dtype))
        x = jax.vmap(net.x_embed)(x)
        y = net.y_pos_w(net.y_pos_h(net.y_patchify(y)))

        h, w = y.shape[-3], y.shape[-2]
        x, y = net.backbone(x, rearrange(y, "h w d -> (h w) d"), c)

        x, _ = net.x_modulation(x, c)
        dx, x = jnp.split(jax.vmap(net.x_unembed)(x), 2, axis=-1)

        y = rearrange(y, "(h w) d -> h w d", h=h, w=w)
        return dx, x, net.y_unembed(y)
