from jaxtyping import Array, Float, Key
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from einops import rearrange

from .utils import FeedForward, Modulation


class MultiStreamAttention(eqx.Module):
    num_heads: int = eqx.field(static=True)
    qkv_proj_x: eqx.nn.Linear
    qkv_proj_y: eqx.nn.Linear
    out_proj_x: eqx.nn.Linear
    out_proj_y: eqx.nn.Linear

    def __init__(
        self,
        dim: int,
        num_heads: int,
        use_bias: bool = False,
        *,
        key: Key[Array, ""],
        **kwargs,
    ):
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        keys = iter(jr.split(key, 4))
        self.num_heads = num_heads
        self.qkv_proj_x = eqx.nn.Linear(
            dim, dim * 3, use_bias, key=next(keys), **kwargs
        )
        self.qkv_proj_y = eqx.nn.Linear(
            dim, dim * 3, use_bias, key=next(keys), **kwargs
        )
        self.out_proj_x = eqx.nn.Linear(dim, dim, use_bias, key=next(keys), **kwargs)
        self.out_proj_y = eqx.nn.Linear(dim, dim, use_bias, key=next(keys), **kwargs)

    def __call__(
        self, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> tuple[Float[Array, "N D"], Float[Array, "M D"]]:
        n = x.shape[-2]
        x = jax.vmap(self.qkv_proj_x)(x)
        y = jax.vmap(self.qkv_proj_y)(y)

        qkv = rearrange(jnp.concat([x, y]), "n (h d) -> n h d", h=self.num_heads)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # qk norm is purely functional: per-head rms, no learned scale
        rms = lambda z: jax.lax.rsqrt(
            jnp.mean(jnp.square(z), axis=-1, keepdims=True) + 1e-6
        )
        q, k = q * rms(q), k * rms(k)

        # dot_product_attention wants a leading batch axis
        h = jax.nn.dot_product_attention(
            q[None], k[None], v[None], implementation="xla"
        )[0]
        h = rearrange(h, "n h d -> n (h d)")

        x, y = jnp.split(h, [n])
        return jax.vmap(self.out_proj_x)(x), jax.vmap(self.out_proj_y)(y)


class MMDiTBlock(eqx.Module):
    mod_x_attn: Modulation
    mod_y_attn: Modulation
    mod_x_mlp: Modulation
    mod_y_mlp: Modulation
    attention: MultiStreamAttention
    mlp_x: FeedForward
    mlp_y: FeedForward

    def __init__(
        self,
        dim: int,
        num_heads: int,
        expand: int,
        *,
        key: Key[Array, ""],
        **kwargs,
    ):
        keys = iter(jr.split(key, 7))
        self.mod_x_attn = Modulation(dim, key=next(keys), **kwargs)
        self.mod_y_attn = Modulation(dim, key=next(keys), **kwargs)
        self.mod_x_mlp = Modulation(dim, key=next(keys), **kwargs)
        self.mod_y_mlp = Modulation(dim, key=next(keys), **kwargs)
        self.attention = MultiStreamAttention(dim, num_heads, key=next(keys), **kwargs)
        self.mlp_x = FeedForward(dim, expand * dim, dim, key=next(keys), **kwargs)
        self.mlp_y = FeedForward(dim, expand * dim, dim, key=next(keys), **kwargs)

    def __call__(
        self,
        x: Float[Array, "N D"],
        y: Float[Array, "M D"],
        c: Float[Array, " D"],
    ) -> tuple[Float[Array, "N D"], Float[Array, "M D"]]:
        hx, gate_x = self.mod_x_attn(x, c)
        hy, gate_y = self.mod_y_attn(y, c)
        hx, hy = self.attention(hx, hy)
        x = x + hx * gate_x
        y = y + hy * gate_y

        hx, gate_x = self.mod_x_mlp(x, c)
        hy, gate_y = self.mod_y_mlp(y, c)
        x = x + jax.vmap(self.mlp_x)(hx) * gate_x
        y = y + jax.vmap(self.mlp_y)(hy) * gate_y
        return x, y


class MMDiT(eqx.Module):
    """Two token streams attending jointly through a stack of blocks, both gated by c."""

    blocks: MMDiTBlock

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_blocks: int,
        expand: int = 2,
        *,
        key: Key[Array, ""],
        **kwargs,
    ):
        make_block = lambda key: MMDiTBlock(
            hidden_dim, num_heads, expand, key=key, **kwargs
        )
        self.blocks = eqx.filter_vmap(make_block)(jr.split(key, num_blocks))

    def __call__(
        self,
        x: Float[Array, "N D"],
        y: Float[Array, "M D"],
        c: Float[Array, " D"],
    ) -> tuple[Float[Array, "N D"], Float[Array, "M D"]]:
        # one compiled block body, scanned over the stacked block params
        params, static = eqx.partition(self.blocks, eqx.is_array)

        @jax.checkpoint
        def scan_blocks(
            carry: tuple[Float[Array, "N D"], Float[Array, "M D"]],
            block: MMDiTBlock,
        ) -> tuple[tuple[Float[Array, "N D"], Float[Array, "M D"]], None]:
            x, y = carry
            return eqx.combine(block, static)(x, y, c), None

        (x, y), _ = jax.lax.scan(scan_blocks, (x, y), params)
        return x, y
