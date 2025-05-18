import jax.numpy as jnp
from jaxtyping import Float, Array
from flax import linen as nn
from flax.linen.initializers import zeros


class MMDiTBlock(nn.Module):
    num_heads: int
    """
    implementation inspired by SD 3.5
    https://openreview.net/forum?id=FPnUhsQJ5B
    """

    @nn.compact
    def __call__(
        self,
        x: Float[Array, "*B N D"],
        c: Float[Array, "*B T D"],
        y: Float[Array, "*B D"],
    ):
        assert x.shape[-1] == c.shape[-1] == y.shape[-1]
        *_, N, D = x.shape

        # get modulation parameters
        y = nn.silu(y)
        modulation_x = nn.Dense(6 * D, kernel_init=zeros)(y)
        alpha_x, beta_x, gamma_x, delta_x, epsilon_x, zeta_x = jnp.split(
            modulation_x, 6, axis=-1
        )
        modulation_c = nn.Dense(6 * D, kernel_init=zeros)(y)
        alpha_c, beta_c, gamma_c, delta_c, epsilon_c, zeta_c = jnp.split(
            modulation_c, 6, axis=-1
        )

        # cross attention block
        hx = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        hx = hx * (1 + alpha_x[..., None, :]) + beta_x[..., None, :]
        hc = nn.LayerNorm(use_bias=False, use_scale=False)(c)
        hc = hc * (1 + alpha_c[..., None, :]) + beta_c[..., None, :]

        hx = nn.Dense(D, use_bias=False)(hx)
        hc = nn.Dense(D, use_bias=False)(hc)

        h = jnp.concatenate([hx, hc], axis=-2)
        h = nn.SelfAttention(self.num_heads, use_bias=False)(h)
        hx, hc = jnp.split(h, [N], axis=-2)

        hx = nn.Dense(D, use_bias=False)(hx)
        hc = nn.Dense(D, use_bias=False)(hc)

        x = x + hx * gamma_x[..., None, :]
        c = c + hc * gamma_c[..., None, :]

        # feed forward blocks
        hx = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        hx = hx * (1 + delta_x[..., None, :]) + epsilon_x[..., None, :]
        hc = nn.LayerNorm(use_bias=False, use_scale=False)(c)
        hc = hc * (1 + delta_c[..., None, :]) + epsilon_c[..., None, :]

        hx = nn.Dense(4 * D, use_bias=False)(hx)
        hx = nn.silu(hx)
        hx = nn.Dense(D, use_bias=False)(hx)

        hc = nn.Dense(4 * D, use_bias=False)(hc)
        hc = nn.silu(hc)
        hc = nn.Dense(D, use_bias=False)(hc)

        x = x + hx * zeta_x[..., None, :]
        c = c + hc * zeta_c[..., None, :]

        return x, c


class SinusoidalEmbed(nn.Module):
    dim: int
    period: float = 2 * jnp.pi
    n_freqs: int = 256

    @nn.compact
    def __call__(self, t: Float[Array, "*B"]):
        freqs = jnp.exp(-jnp.log(self.period) * jnp.linspace(0, 1, self.n_freqs))
        angles = 2 * jnp.pi * freqs * t[..., None]
        x = jnp.concat([jnp.sin(angles), jnp.cos(angles)], axis=-1)
        x = nn.Dense(self.dim)(x)
        x = nn.silu(x)
        x = nn.Dense(self.dim)(x)
        return x


class GWMMDiT(nn.Module):
    dim: int
    num_heads: int
    num_blocks: int

    @nn.compact
    def __call__(
        self,
        x: Float[Array, "*B N P"],
        t: Float[Array, "*B 1"],
        c: Float[Array, "*B T C"],
    ):

        *_, N, P = x.shape
        *_, T, C = c.shape

        # main sequence embedding
        x = nn.Dense(self.dim)(x)
        x = x + SinusoidalEmbed(self.dim)(jnp.arange(N) / N)

        # conditioning embedding
        c = nn.Dense(self.dim)(c)
        c = c + SinusoidalEmbed(self.dim)(jnp.arange(T) / T)

        # timestep embedding
        y = SinusoidalEmbed(self.dim)(t)
        y = nn.Dense(self.dim)(y)
        y = nn.silu(y)
        y = nn.Dense(self.dim)(y)

        # cross attention blocks
        for _ in range(self.num_blocks):
            x, c = MMDiTBlock(self.num_heads)(x, c, y)

        # final projection
        modulation = nn.Dense(2 * self.dim, kernel_init=zeros)(y)
        alpha, beta = jnp.split(modulation, 2, axis=-1)
        x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x = x * (1 + alpha[..., None, :]) + beta[..., None, :]
        x = nn.Dense(P)(x)
        return x
