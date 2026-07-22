from jaxtyping import Float, Array
import jax.numpy as jnp
from flax import nnx


class FeedForward(nnx.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, **kwargs):
        self.linear1 = nnx.Linear(in_dim, 2 * hidden_dim, **kwargs)
        self.linear2 = nnx.Linear(hidden_dim, out_dim, **kwargs)

    def __call__(self, x: Float[Array, "... I"]) -> Float[Array, "... O"]:
        x = self.linear1(x)
        x, g = jnp.split(x, 2, axis=-1)
        x = x * nnx.silu(g)
        return self.linear2(x)


class Modulation(nnx.Module):
    def __init__(self, dim: int, **kwargs):
        self.linear = nnx.Linear(
            in_features=dim,
            out_features=3 * dim,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            **kwargs,
        )

    def __call__(
        self, x: Float[Array, "... D"], c: Float[Array, "... D"]
    ) -> tuple[Float[Array, "... D"], Float[Array, "... D"]]:
        c = nnx.silu(c)
        c = self.linear(c)
        shift, scale, gate = jnp.split(c, 3, axis=-1)
        x = nnx.standardize(x, axis=-1)
        x = x * (1 + scale) + shift
        return x, gate


class SinusoidalEmbed(nnx.Module):
    def __init__(self, dim: int, period: float = 2 * jnp.pi, **kwargs):
        self.dim = dim
        self.period = period
        self.embed = FeedForward(2 * dim, dim, dim, **kwargs)

    def __call__(self, t: Float[Array, "..."]) -> Float[Array, "... D"]:
        assert jnp.issubdtype(t.dtype, jnp.floating), "t must be floating point"
        log_f = -jnp.linspace(0, jnp.log(self.period), self.dim, dtype=t.dtype)
        angles = 2 * jnp.pi * jnp.exp(log_f) * t[..., None]
        x = jnp.concat([jnp.sin(angles), jnp.cos(angles)], axis=-1)
        return self.embed(x)
