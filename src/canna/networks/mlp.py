from jaxtyping import Float, Array
import jax
import jax.numpy as jnp
from flax import nnx

from .utils import FeedForward, Modulation, SinusoidalEmbed


class MLPBlock(nnx.Module):
    def __init__(self, dim: int, expand: int, **kwargs):
        self.mod = Modulation(dim, **kwargs)
        self.mlp = FeedForward(dim, expand * dim, dim, **kwargs)

    def __call__(
        self, x: Float[Array, "... D"], c: Float[Array, "... D"]
    ) -> Float[Array, "... D"]:
        h, gate = self.mod(x, c)
        return x + self.mlp(h) * gate


class MLPFlow(nnx.Module):
    def __init__(
        self,
        x_shape: tuple[int],
        y_shape: tuple[int],
        hidden_dim: int,
        num_blocks: int,
        expand: int = 2,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        (x_dim,), (y_dim,) = x_shape, y_shape

        self.x_embed = FeedForward(x_dim, hidden_dim, hidden_dim, rngs=rngs, **kwargs)
        self.y_embed = FeedForward(y_dim, hidden_dim, hidden_dim, rngs=rngs, **kwargs)
        self.t_embed = SinusoidalEmbed(hidden_dim, rngs=rngs, **kwargs)
        self.c_embed = FeedForward(
            hidden_dim, hidden_dim, hidden_dim, rngs=rngs, **kwargs
        )

        @nnx.scan(in_axes=nnx.Carry, length=num_blocks)
        def make_block(rngs: nnx.Rngs) -> tuple[nnx.Rngs, MLPBlock]:
            block = MLPBlock(hidden_dim, expand, rngs=rngs, **kwargs)
            return rngs, block

        _, self.blocks = make_block(rngs)

        self.x_modulation = Modulation(hidden_dim, rngs=rngs, **kwargs)
        self.x_unembed = FeedForward(
            hidden_dim, hidden_dim, 2 * x_dim, rngs=rngs, **kwargs
        )
        self.y_unembed = FeedForward(hidden_dim, hidden_dim, y_dim, rngs=rngs, **kwargs)

    def __call__(
        self,
        x: Float[Array, "... X"],
        y: Float[Array, "... Y"],
        t: Float[Array, "..."],
    ) -> tuple[
        Float[Array, "... X"],
        Float[Array, "... X"],
        Float[Array, "... Y"],
    ]:
        y = self.y_embed(y)
        t = self.t_embed(t)
        c = self.c_embed(t + y)
        x = self.x_embed(x)

        # lax.scan over split state: nnx.scan blocks autodiff w.r.t. the input
        graphdef, state = nnx.split(self.blocks)

        @jax.checkpoint
        def scan_blocks(
            x: Float[Array, "... D"], block: nnx.State
        ) -> tuple[Float[Array, "... D"], None]:
            return nnx.merge(graphdef, block)(x, c), None

        x, _ = jax.lax.scan(scan_blocks, x, state)

        x, _ = self.x_modulation(x, c)
        x = self.x_unembed(x)
        dx, x = jnp.split(x, 2, axis=-1)

        y = self.y_unembed(y)
        return dx, x, y
