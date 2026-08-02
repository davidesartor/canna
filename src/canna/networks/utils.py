from jaxtyping import Array, Float, Key
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from einops import rearrange

DType = jnp.dtype | None


class FeedForward(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        key: Key[Array, ""],
        **kwargs,
    ):
        key1, key2 = jr.split(key)
        self.linear1 = eqx.nn.Linear(in_dim, 2 * hidden_dim, key=key1, **kwargs)
        self.linear2 = eqx.nn.Linear(hidden_dim, out_dim, key=key2, **kwargs)

    def __call__(self, x: Float[Array, " I"]) -> Float[Array, " O"]:
        x, gate = jnp.split(self.linear1(x), 2)
        return self.linear2(x * jax.nn.silu(gate))


class Modulation(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, dim: int, *, key: Key[Array, ""], **kwargs):
        linear = eqx.nn.Linear(dim, 3 * dim, key=key, **kwargs)
        # zero-init, so every gated residual branch starts as the exact identity
        self.linear = jax.tree.map(jnp.zeros_like, linear)

    def __call__(
        self, x: Float[Array, "... D"], c: Float[Array, " D"]
    ) -> tuple[Float[Array, "... D"], Float[Array, " D"]]:
        shift, scale, gate = jnp.split(self.linear(jax.nn.silu(c)), 3)
        return jax.nn.standardize(x, axis=-1) * (1 + scale) + shift, gate


class SinusoidalEmbed(eqx.Module):
    dim: int = eqx.field(static=True)
    period: float = eqx.field(static=True)
    embed: FeedForward

    def __init__(
        self,
        dim: int,
        period: float = 2 * jnp.pi,
        *,
        key: Key[Array, ""],
        **kwargs,
    ):
        self.dim = dim
        self.period = period
        self.embed = FeedForward(2 * dim, dim, dim, key=key, **kwargs)

    def __call__(self, t: Float[Array, ""]) -> Float[Array, " D"]:
        assert jnp.issubdtype(t.dtype, jnp.floating), "t must be floating point"
        log_f = -jnp.linspace(0, jnp.log(self.period), self.dim, dtype=t.dtype)
        angles = 2 * jnp.pi * jnp.exp(log_f) * t
        return self.embed(jnp.concat([jnp.sin(angles), jnp.cos(angles)]))


class PositionalEmbed(eqx.Module):
    axis: int = eqx.field(static=True)
    max_len: int = eqx.field(static=True)
    embed: SinusoidalEmbed

    def __init__(
        self,
        dim: int,
        max_len: int,
        axis: int = -2,
        *,
        key: Key[Array, ""],
        **kwargs,
    ):
        self.axis = axis
        self.max_len = max_len
        self.embed = SinusoidalEmbed(dim, key=key, **kwargs)

    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        assert x.shape[self.axis] <= self.max_len, "sequence longer than max_len"
        pos = jnp.arange(x.shape[self.axis], dtype=x.dtype) / self.max_len
        shape = [1] * x.ndim
        shape[self.axis], shape[-1] = pos.shape[0], self.embed.dim
        return x + jax.vmap(self.embed)(pos).reshape(shape)


class Fold2d(eqx.Module):
    def __call__(self, z: Float[Array, "H W C"]) -> Float[Array, "h w D"]:
        return rearrange(z, "(h p) (w q) c -> h w (p q c)", p=2, q=2)


class Unfold2d(eqx.Module):
    def __call__(self, z: Float[Array, "h w D"]) -> Float[Array, "H W C"]:
        return rearrange(z, "h w (p q c) -> (h p) (w q) c", p=2, q=2)


class Patchify(eqx.Module):
    fold: Fold2d
    linears: tuple[eqx.nn.Linear, ...]
    norms: tuple[eqx.nn.RMSNorm, ...]

    def __init__(
        self,
        channels: int,
        dim: int,
        stages: int = 4,
        *,
        key: Key[Array, ""],
        **kwargs,
    ):
        assert stages >= 1, "stages must be at least 1"
        narrowest = 4 ** (stages - 1)
        assert dim % narrowest == 0, f"hidden_dim must be divisible by {narrowest}"
        dims = [channels, *(dim // 4**stage for stage in reversed(range(stages)))]

        keys = iter(jr.split(key, stages))
        self.fold = Fold2d()
        self.linears = tuple(
            eqx.nn.Linear(4 * i, o, key=next(keys), **kwargs)
            for i, o in zip(dims, dims[1:])
        )
        self.norms = tuple(
            eqx.nn.RMSNorm(o, use_bias=False, **kwargs) for o in dims[1:-1]
        )

    def __call__(self, y: Float[Array, "H W C"]) -> Float[Array, "h w D"]:
        for stage, linear in enumerate(self.linears):
            y = jax.vmap(jax.vmap(linear))(self.fold(y))
            # the last stage projects straight to the model dim, unnormalized
            if stage < len(self.norms):
                y = jax.nn.silu(jax.vmap(jax.vmap(self.norms[stage]))(y))
        return y


class UnPatchify(eqx.Module):
    unfold: Unfold2d
    linears: tuple[eqx.nn.Linear, ...]
    norms: tuple[eqx.nn.RMSNorm, ...]

    def __init__(
        self,
        channels: int,
        dim: int,
        stages: int = 4,
        *,
        key: Key[Array, ""],
        **kwargs,
    ):
        assert stages >= 1, "stages must be at least 1"
        narrowest = 4 ** (stages - 1)
        assert dim % narrowest == 0, f"hidden_dim must be divisible by {narrowest}"
        dims = [*(dim // 4**stage for stage in range(stages)), channels]

        keys = iter(jr.split(key, stages))
        self.unfold = Unfold2d()
        self.linears = tuple(
            eqx.nn.Linear(i, 4 * o, key=next(keys), **kwargs)
            for i, o in zip(dims, dims[1:])
        )
        self.norms = tuple(
            eqx.nn.RMSNorm(i, use_bias=False, **kwargs) for i in dims[1:-1]
        )

    def __call__(self, y: Float[Array, "h w D"]) -> Float[Array, "H W C"]:
        for stage, linear in enumerate(self.linears):
            # the first stage takes the model dim straight from the backbone
            if stage > 0:
                y = jax.nn.silu(jax.vmap(jax.vmap(self.norms[stage - 1]))(y))
            y = self.unfold(jax.vmap(jax.vmap(linear))(y))
        return y
