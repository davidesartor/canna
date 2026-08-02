from jaxtyping import Array, Float, Key
import jax
import jax.random as jr
import equinox as eqx

from .utils import FeedForward, Modulation


class MLPBlock(eqx.Module):
    mod: Modulation
    mlp: FeedForward

    def __init__(self, dim: int, expand: int, *, key: Key[Array, ""], **kwargs):
        key_mod, key_mlp = jr.split(key)
        self.mod = Modulation(dim, key=key_mod, **kwargs)
        self.mlp = FeedForward(dim, expand * dim, dim, key=key_mlp, **kwargs)

    def __call__(
        self, x: Float[Array, " D"], c: Float[Array, " D"]
    ) -> Float[Array, " D"]:
        h, gate = self.mod(x, c)
        return x + self.mlp(h) * gate


class MLP(eqx.Module):
    """One stream of gated residual blocks, modulated by c."""

    blocks: MLPBlock

    def __init__(
        self,
        hidden_dim: int,
        num_blocks: int,
        expand: int = 2,
        *,
        key: Key[Array, ""],
        **kwargs,
    ):
        make_block = lambda key: MLPBlock(hidden_dim, expand, key=key, **kwargs)
        self.blocks = eqx.filter_vmap(make_block)(jr.split(key, num_blocks))

    def __call__(
        self, x: Float[Array, " D"], c: Float[Array, " D"]
    ) -> Float[Array, " D"]:
        # one compiled block body, scanned over the stacked block params
        params, static = eqx.partition(self.blocks, eqx.is_array)

        @jax.checkpoint
        def scan_blocks(
            x: Float[Array, " D"], block: MLPBlock
        ) -> tuple[Float[Array, " D"], None]:
            return eqx.combine(block, static)(x, c), None

        x, _ = jax.lax.scan(scan_blocks, x, params)
        return x
