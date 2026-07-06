from jaxtyping import Float, Array, Scalar, Key
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from einops import rearrange

ACTIVATION = jax.nn.silu


def rms_norm(x: Float[Array, "... D"], eps: float = 1e-8):
    ms = jnp.mean(x**2, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(ms + eps)


def adaptive_norm(
    x: Float[Array, "... N D"],
    scale: Float[Array, "... 1 D"],
    shift: Float[Array, "... 1 D"],
    eps: float = 1e-8,
):
    x = x - x.mean(axis=-1, keepdims=True)
    x = x / (x.std(axis=-1, keepdims=True) + eps)
    x = x * (1 + scale) + shift
    return x


def FeedForward(input_dim: int, hidden_dim: int, output_dim: int, *, key: Key):
    return eqx.nn.MLP(
        in_size=input_dim,
        width_size=hidden_dim,
        out_size=output_dim,
        activation=ACTIVATION,
        depth=1,
        key=key,
    )


class Modulation(eqx.Module):
    dim: int
    linear: eqx.nn.Linear

    def __init__(self, dim: int, *, key: Key):
        self.dim = dim
        linear = eqx.nn.Linear(in_features=dim, out_features=3 * dim, key=key)
        # zero out weight and bias to start with identity modulation
        self.linear = jax.tree.map(lambda w: jnp.zeros_like(w), linear)

    def __call__(
        self, c: Float[Array, "D"]
    ) -> tuple[Float[Array, "D"], Float[Array, "D"], Float[Array, "D"]]:
        assert c.shape[-1] == self.dim, f"Expected c dim {self.dim}, got {c.shape[-1]}"
        modulation = self.linear(ACTIVATION(c))
        shift, scale, gate = jnp.split(modulation, 3, axis=-1)
        return shift, scale, gate


class SinusoidalEmbed(eqx.Module):
    dim: int
    period: float
    embed: eqx.nn.MLP

    def __init__(self, dim: int, period: float = 2 * jnp.pi, *, key: Key):
        self.dim = dim
        self.period = period
        self.embed = FeedForward(2 * dim, dim, dim, key=key)

    def __call__(self, t: Scalar) -> Float[Array, "D"]:
        freqs = jnp.exp(
            -jnp.log(self.period) * jnp.linspace(0, 1, self.dim, dtype=t.dtype)
        )
        angles = 2 * jnp.pi * freqs * t[..., None]
        x = jnp.concat([jnp.sin(angles), jnp.cos(angles)], axis=-1)
        x = self.embed(x)
        return x


class MultiStreamAttention(eqx.Module):
    dim: int
    num_heads: int
    use_bias: bool
    qkv_proj_x: eqx.nn.Linear
    qkv_proj_y: eqx.nn.Linear
    out_proj_x: eqx.nn.Linear
    out_proj_y: eqx.nn.Linear

    def __init__(self, dim: int, num_heads: int, use_bias: bool = False, *, key: Key):
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.use_bias = use_bias

        k_qkv_x, k_qkv_y, k_out_x, k_out_y = jr.split(key, 4)
        self.qkv_proj_x = eqx.nn.Linear(dim, dim * 3, use_bias=use_bias, key=k_qkv_x)
        self.qkv_proj_y = eqx.nn.Linear(dim, dim * 3, use_bias=use_bias, key=k_qkv_y)
        self.out_proj_x = eqx.nn.Linear(dim, dim, use_bias=use_bias, key=k_out_x)
        self.out_proj_y = eqx.nn.Linear(dim, dim, use_bias=use_bias, key=k_out_y)

    def __call__(self, x: Float[Array, "N D"], y: Float[Array, "M D"]):
        assert x.shape[-1] == self.dim, f"Expected x dim {self.dim}, got {x.shape[-1]}"
        assert y.shape[-1] == self.dim, f"Expected y dim {self.dim}, got {y.shape[-1]}"

        x = eqx.filter_vmap(self.qkv_proj_x)(x)
        y = eqx.filter_vmap(self.qkv_proj_y)(y)
        h = jnp.concat([x, y], axis=-2)
        qkv = rearrange(h, "... N (H D) -> ... N H D", H=self.num_heads)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q, k = rms_norm(q), rms_norm(k)
        h = jax.nn.dot_product_attention(q, k, v)
        h = rearrange(h, "... N H D -> ... N (H D)")
        x, y = jnp.split(h, [len(x)], axis=-2)
        x = eqx.filter_vmap(self.out_proj_x)(x)
        y = eqx.filter_vmap(self.out_proj_y)(y)
        return x, y


class MMDiTBlock(eqx.Module):
    dim: int
    num_heads: int
    expand: int
    modulations: Modulation  # stacked: 4 modulations along leading axis
    attention: MultiStreamAttention
    mlp_x: eqx.nn.MLP
    mlp_y: eqx.nn.MLP

    def __init__(self, dim: int, num_heads: int, expand: int = 2, *, key: Key):
        self.dim = dim
        self.num_heads = num_heads
        self.expand = expand

        k_mods, k_attn, k_mlp_x, k_mlp_y = jr.split(key, 4)
        modulation_init = lambda k: Modulation(dim, key=k)
        self.modulations = eqx.filter_vmap(modulation_init)(jr.split(k_mods, 4))
        self.attention = MultiStreamAttention(dim, num_heads, key=k_attn)
        self.mlp_x = FeedForward(dim, expand * dim, dim, key=k_mlp_x)
        self.mlp_y = FeedForward(dim, expand * dim, dim, key=k_mlp_y)

    def __call__(
        self,
        x: Float[Array, "N D"],
        y: Float[Array, "M D"],
        c: Float[Array, "D"],
    ):
        assert x.shape[-1] == self.dim, f"Expected x dim {self.dim}, got {x.shape[-1]}"
        assert y.shape[-1] == self.dim, f"Expected y dim {self.dim}, got {y.shape[-1]}"
        assert c.shape[-1] == self.dim, f"Expected c dim {self.dim}, got {c.shape[-1]}"

        # Compute all 4 modulations at once for efficiency.
        modulate = eqx.filter_vmap(lambda m: m(c), out_axes=-2)
        shifts, scales, gates = modulate(self.modulations)  # (..., 4, D).
        shift_x1, shift_y1, shift_x2, shift_y2 = jnp.split(shifts, 4, axis=-2)
        scale_x1, scale_y1, scale_x2, scale_y2 = jnp.split(scales, 4, axis=-2)
        gate_x1, gate_y1, gate_x2, gate_y2 = jnp.split(gates, 4, axis=-2)

        # cross attention block
        hx = adaptive_norm(x, scale_x1, shift_x1)
        hy = adaptive_norm(y, scale_y1, shift_y1)
        hx, hy = self.attention(hx, hy)
        x = x + hx * gate_x1
        y = y + hy * gate_y1

        # feed forward blocks
        hx = adaptive_norm(x, scale_x2, shift_x2)
        hx = eqx.filter_vmap(self.mlp_x)(hx)
        x = x + hx * gate_x2

        hy = adaptive_norm(y, scale_y2, shift_y2)
        hy = eqx.filter_vmap(self.mlp_y)(hy)
        y = y + hy * gate_y2
        return x, y


class MMDiT(eqx.Module):
    x_dim: int
    y_dim: int
    hidden_dim: int
    num_heads: int
    num_blocks: int
    x_pos_embed: SinusoidalEmbed
    y_pos_embed: SinusoidalEmbed
    c_pos_embed: SinusoidalEmbed
    x_embed: eqx.nn.MLP
    y_embed: eqx.nn.MLP
    c_embed: eqx.nn.MLP
    blocks: MMDiTBlock
    out_modulation: Modulation
    out_unembed: eqx.nn.MLP

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_blocks: int,
        *,
        key: Key,
    ):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        k_pos, k_embeds, k_blocks, k_out = jr.split(key, 4)

        # position embeddings
        k_x_pos, k_y_pos, k_c_pos = jr.split(k_pos, 3)
        self.x_pos_embed = SinusoidalEmbed(hidden_dim, key=k_x_pos)
        self.y_pos_embed = SinusoidalEmbed(hidden_dim, key=k_y_pos)
        self.c_pos_embed = SinusoidalEmbed(hidden_dim, key=k_c_pos)

        # input embeddings
        k_x_embed, k_y_embed, k_c_embed = jr.split(k_embeds, 3)
        self.x_embed = FeedForward(x_dim, hidden_dim, hidden_dim, key=k_x_embed)
        self.y_embed = FeedForward(y_dim, hidden_dim, hidden_dim, key=k_y_embed)
        self.c_embed = FeedForward(hidden_dim, hidden_dim, hidden_dim, key=k_c_embed)

        # cross attention blocks
        init_block = lambda k: MMDiTBlock(hidden_dim, num_heads, key=k)
        self.blocks = eqx.filter_vmap(init_block)(jr.split(k_blocks, num_blocks))

        # output layers: projects to 2*x_dim, split into (dx, x_mle)
        k_out_mod, k_out_unembed = jr.split(k_out, 2)
        self.out_modulation = Modulation(hidden_dim, key=k_out_mod)
        self.out_unembed = FeedForward(
            hidden_dim, hidden_dim, 2 * x_dim, key=k_out_unembed
        )

    def __call__(
        self,
        x: Float[Array, "N D"],
        y: Float[Array, "M C"],
        t: Scalar,
    ) -> tuple[Float[Array, "N D"], Float[Array, "N D"]]:
        assert (
            x.shape[-1] == self.x_dim
        ), f"Expected x dim {self.x_dim}, got {x.shape[-1]}"
        assert (
            y.shape[-1] == self.y_dim
        ), f"Expected y dim {self.y_dim}, got {y.shape[-1]}"

        # embedding
        x_pos = jnp.linspace(0, 1, len(x), dtype=x.dtype)
        x = eqx.filter_vmap(self.x_embed)(x) + eqx.filter_vmap(self.x_pos_embed)(x_pos)
        y_pos = jnp.linspace(0, 1, len(y), dtype=y.dtype)
        y = eqx.filter_vmap(self.y_embed)(y) + eqx.filter_vmap(self.y_pos_embed)(y_pos)
        c = self.c_embed(self.c_pos_embed(t))

        # apply cross attention blocks with scan
        # need to partition cause MLP activations are static fields
        blocks_params, blocks_treedef = eqx.partition(self.blocks, eqx.is_array)

        def scan_step(xy, params):
            block = eqx.combine(params, blocks_treedef)
            return block(*xy, c), None

        (x, y), _ = jax.lax.scan(scan_step, (x, y), blocks_params)

        # unembedding
        shift, scale, _ = self.out_modulation(c)
        x = adaptive_norm(x, scale, shift)
        x = eqx.filter_vmap(self.out_unembed)(x)
        dx, x = jnp.split(x, 2, axis=-1)
        return dx, x

    def push(
        self,
        x: Float[Array, "N D"],
        y: Float[Array, "M C"],
        ode_steps: int = 16,
        exponential_map=jnp.sum,
    ) -> Float[Array, "N D"]:
        def runge_kutta_4_step(i, x):
            dt = 1 / ode_steps
            t = (i * dt).astype(x.dtype)
            k1, _ = self(x, y, t)
            k2, _ = self(x + k1 * dt / 2, y, t + dt / 2)
            k3, _ = self(x + k2 * dt / 2, y, t + dt / 2)
            k4, _ = self(x + k3 * dt, y, t + dt)
            dx = (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
            x = exponential_map(x, dx)
            return x

        x = jax.lax.fori_loop(0, ode_steps, runge_kutta_4_step, x)
        return x
