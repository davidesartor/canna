from jaxtyping import Array, Float, Key
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from einops import rearrange

DType = jnp.dtype | None


def stacked_linear(
    copies: int, in_dim: int, out_dim: int, *, key: Key[Array, ""], **kwargs
) -> eqx.nn.Linear:
    """`copies` independent linear maps in one module, stacked on a leading axis.

    A fused map (q, k and v, say, as one 3*D-wide matrix) would be orthogonalised by Muon as
    a single matrix; stacked, Muon sees the leading axis as a batch axis and treats each
    map on its own. Apply with `apply_stacked`.
    """
    make = lambda key: eqx.nn.Linear(in_dim, out_dim, key=key, **kwargs)
    return eqx.filter_vmap(make)(jr.split(key, copies))


def apply_stacked(linear: eqx.nn.Linear, x: Float[Array, " I"]) -> Float[Array, "K O"]:
    """Every map of a `stacked_linear` applied to the same input."""
    return eqx.filter_vmap(lambda linear: linear(x))(linear)


class FeedForward(eqx.Module):
    """SwiGLU: linear2(value * silu(gate)), with value and gate as two stacked maps."""

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
        self.linear1 = stacked_linear(2, in_dim, hidden_dim, key=key1, **kwargs)
        self.linear2 = eqx.nn.Linear(hidden_dim, out_dim, key=key2, **kwargs)

    def __call__(self, x: Float[Array, " I"]) -> Float[Array, " O"]:
        x, gate = apply_stacked(self.linear1, x)
        return self.linear2(x * jax.nn.silu(gate))


class Modulation(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, dim: int, *, key: Key[Array, ""], **kwargs):
        # shift, scale and gate as three stacked maps. Zero-init, so every gated residual
        # branch starts as the exact identity
        linear = stacked_linear(3, dim, dim, key=key, **kwargs)
        self.linear = jax.tree.map(jnp.zeros_like, linear)

    def __call__(
        self, x: Float[Array, "... D"], c: Float[Array, " D"]
    ) -> tuple[Float[Array, "... D"], Float[Array, " D"]]:
        shift, scale, gate = apply_stacked(self.linear, jax.nn.silu(c))
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
