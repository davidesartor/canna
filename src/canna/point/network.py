from jaxtyping import Array, Float, Key
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from ..networks import MLP, FeedForward, Modulation, SinusoidalEmbed
from ..networks.utils import DType


class PointFlow(eqx.Module):
    """Velocity field over a flat vector conditioned on a vector observation."""

    dtype: DType = eqx.field(static=True)
    x_embed: FeedForward
    y_embed: FeedForward
    t_embed: SinusoidalEmbed
    c_embed: FeedForward
    backbone: MLP
    x_modulation: Modulation
    x_unembed: FeedForward

    def __init__(
        self,
        x_shape: tuple[int],
        y_shape: tuple[int],
        hidden_dim: int,
        num_blocks: int,
        expand: int = 2,
        dtype: DType = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        *,
        key: Key[Array, ""],
        **kwargs,
    ):
        (x_dim,), (y_dim,) = x_shape, y_shape
        self.dtype = dtype

        # eqx.nn's dtype is the parameter dtype; the compute dtype is applied in __call__
        kwargs = dict(kwargs, dtype=param_dtype)
        keys = iter(jr.split(key, 7))

        self.x_embed = FeedForward(
            x_dim, hidden_dim, hidden_dim, key=next(keys), **kwargs
        )
        self.y_embed = FeedForward(
            y_dim, hidden_dim, hidden_dim, key=next(keys), **kwargs
        )
        self.t_embed = SinusoidalEmbed(hidden_dim, key=next(keys), **kwargs)
        self.c_embed = FeedForward(
            hidden_dim, hidden_dim, hidden_dim, key=next(keys), **kwargs
        )

        self.backbone = MLP(hidden_dim, num_blocks, expand, key=next(keys), **kwargs)

        self.x_modulation = Modulation(hidden_dim, key=next(keys), **kwargs)
        self.x_unembed = FeedForward(
            hidden_dim, hidden_dim, x_dim, key=next(keys), **kwargs
        )

    def __call__(
        self,
        x: Float[Array, " X"],
        t: Float[Array, ""],
        y: Float[Array, " Y"],
    ) -> Float[Array, " X"]:
        # mixed precision: the stored params keep param_dtype, this forward runs in dtype
        params, static = eqx.partition(self, eqx.is_inexact_array)
        net = eqx.combine(jax.tree.map(lambda p: p.astype(self.dtype), params), static)
        x, y = x.astype(self.dtype), y.astype(self.dtype)

        c = (net.y_embed(y) + net.t_embed(t)).astype(self.dtype)
        c = net.c_embed(c)

        x = net.backbone(net.x_embed(x), c)

        x, _ = net.x_modulation(x, c)
        return net.x_unembed(x)
