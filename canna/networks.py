from jaxtyping import Float, Array
import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange


class FeedForward(nnx.Module):
    """SwiGLU MLP (in -> hidden -> out): SiLU-gated hidden."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, **kwargs):
        self.linear1 = nnx.Linear(in_dim, 2 * hidden_dim, **kwargs)
        self.linear2 = nnx.Linear(hidden_dim, out_dim, **kwargs)

    def __call__(self, x: Float[Array, "... I"]) -> Float[Array, "... O"]:
        x = self.linear1(x)
        x, g = jnp.split(x, 2, axis=-1)
        x = x * nnx.silu(g)
        return self.linear2(x)


class SinusoidalEmbed(nnx.Module):
    """Sinusoidal (sin/cos) frequency features of a scalar, projected through a small MLP."""

    def __init__(self, dim: int, period: float = 2 * jnp.pi, **kwargs):
        self.dim = dim
        self.period = period
        self.embed = FeedForward(2 * dim, dim, dim, **kwargs)

    def __call__(self, t: Float[Array, "..."]) -> Float[Array, "... D"]:
        log_f = -jnp.linspace(0, jnp.log(self.period), self.dim, dtype=t.dtype)
        angles = 2 * jnp.pi * jnp.exp(log_f) * t[..., None]
        x = jnp.concat([jnp.sin(angles), jnp.cos(angles)], axis=-1)
        return self.embed(x)


class PositionalEmbed(nnx.Module):
    """Add a sinusoidal position embedding along `axis`."""

    def __init__(self, dim: int, axis: int = -2, **kwargs):
        self.axis = axis
        self.embed = SinusoidalEmbed(dim, **kwargs)

    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        pos = jnp.linspace(0, 1, x.shape[self.axis], dtype=x.dtype)
        # broadcast (position, dim) back onto `axis` and the channel axis
        shape = [1] * x.ndim
        shape[self.axis], shape[-1] = pos.shape[0], self.embed.dim
        return x + self.embed(pos).reshape(shape)


class Fold2d(nnx.Module):
    """Fold each 2x2 spatial patch into the channel dim."""

    def __call__(self, z: Float[Array, "... H W C"]) -> Float[Array, "... h w D"]:
        return rearrange(z, "... (h p) (w q) c -> ... h w (p q c)", p=2, q=2)


class Patchify(nnx.Module):
    """Patch-embed a (H, W, C) image into a (h, w, D) token grid."""

    def __init__(self, channels: int, dim: int, **kwargs):
        assert dim % 64 == 0, "hidden_dim must be divisible by 64"
        # a 2x2 stride-2 conv <=> fold, then a per-pixel linear
        self.layers = nnx.Sequential(
            Fold2d(),
            nnx.Linear(4 * channels, dim // 64, **kwargs),
            nnx.RMSNorm(dim // 64, **kwargs),
            nnx.silu,
            Fold2d(),
            nnx.Linear(dim // 16, dim // 16, **kwargs),
            nnx.RMSNorm(dim // 16, **kwargs),
            nnx.silu,
            Fold2d(),
            nnx.Linear(dim // 4, dim // 4, **kwargs),
            nnx.RMSNorm(dim // 4, **kwargs),
            nnx.silu,
            Fold2d(),
            nnx.Linear(dim, dim, **kwargs),
        )

    def __call__(self, y: Float[Array, "... H W C"]) -> Float[Array, "... h w D"]:
        return self.layers(y)


class Unfold2d(nnx.Module):
    """Unfold the channel dim back into a 2x2 spatial patch."""

    def __call__(self, z: Float[Array, "... h w D"]) -> Float[Array, "... H W C"]:
        return rearrange(z, "... h w (p q c) -> ... (h p) (w q) c", p=2, q=2)


class UnPatchify(nnx.Module):
    """Patch-unembed a (h, w, D) token grid back into a (H, W, C) image."""

    def __init__(self, channels: int, dim: int, **kwargs):
        assert dim % 64 == 0, "hidden_dim must be divisible by 64"
        # a transposed 2x2 stride-2 conv <=> a per-pixel linear, then unfold
        self.layers = nnx.Sequential(
            nnx.Linear(dim, dim, **kwargs),
            Unfold2d(),
            nnx.RMSNorm(dim // 4, **kwargs),
            nnx.silu,
            nnx.Linear(dim // 4, dim // 4, **kwargs),
            Unfold2d(),
            nnx.RMSNorm(dim // 16, **kwargs),
            nnx.silu,
            nnx.Linear(dim // 16, dim // 16, **kwargs),
            Unfold2d(),
            nnx.RMSNorm(dim // 64, **kwargs),
            nnx.silu,
            nnx.Linear(dim // 64, 4 * channels, **kwargs),
            Unfold2d(),
        )

    def __call__(self, y: Float[Array, "... h w D"]) -> Float[Array, "... H W C"]:
        return self.layers(y)


class Modulation(nnx.Module):
    """adaLN: standardize x, shift/scale/gate it from conditioning c; zero-init = identity + zero gate at start."""

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
        # compute the modulation
        c = nnx.silu(c)
        c = self.linear(c)
        shift, scale, gate = jnp.split(c, 3, axis=-1)

        # apply adaptive layer norm
        x = nnx.standardize(x, axis=-1)
        x = x * (1 + scale) + shift
        return x, gate


class MultiStreamAttention(nnx.Module):
    """Joint qk-normed attention over the concatenated (x, y) token streams."""

    def __init__(self, dim: int, num_heads: int, use_bias: bool = False, **kwargs):
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.qkv_proj_x = nnx.Linear(dim, dim * 3, use_bias=use_bias, **kwargs)
        self.qkv_proj_y = nnx.Linear(dim, dim * 3, use_bias=use_bias, **kwargs)
        self.out_proj_x = nnx.Linear(dim, dim, use_bias=use_bias, **kwargs)
        self.out_proj_y = nnx.Linear(dim, dim, use_bias=use_bias, **kwargs)
        self.q_norm = nnx.RMSNorm(dim // num_heads, **kwargs)
        self.k_norm = nnx.RMSNorm(dim // num_heads, **kwargs)

    def __call__(
        self, x: Float[Array, "... N D"], y: Float[Array, "... M D"]
    ) -> tuple[Float[Array, "... N D"], Float[Array, "... M D"]]:
        # per-stream qkv projection, then concat into one sequence
        x = self.qkv_proj_x(x)
        y = self.qkv_proj_y(y)
        h = jnp.concat([x, y], axis=-2)

        # split heads, qk-normed attention
        qkv = rearrange(h, "... N (H D) -> ... N H D", H=self.num_heads)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q, k = self.q_norm(q), self.k_norm(k)
        h = jax.nn.dot_product_attention(q, k, v, implementation="xla")
        h = rearrange(h, "... N H D -> ... N (H D)")

        # split streams back apart, per-stream output projection
        x, y = jnp.split(h, [x.shape[-2]], axis=-2)
        x = self.out_proj_x(x)
        y = self.out_proj_y(y)
        return x, y


class MMDiTBlock(nnx.Module):
    """One MMDiT block: joint (x, y) attention + per-stream MLPs, all adaLN-modulated by conditioning c."""

    def __init__(self, dim: int, num_heads: int, expand: int, **kwargs):
        # one modulation per (stream, sublayer): x/y attention, x/y mlp
        self.mod_x_attn = Modulation(dim, **kwargs)
        self.mod_y_attn = Modulation(dim, **kwargs)
        self.mod_x_mlp = Modulation(dim, **kwargs)
        self.mod_y_mlp = Modulation(dim, **kwargs)
        self.attention = MultiStreamAttention(dim, num_heads, **kwargs)
        self.mlp_x = FeedForward(dim, expand * dim, dim, **kwargs)
        self.mlp_y = FeedForward(dim, expand * dim, dim, **kwargs)

    def __call__(
        self,
        x: Float[Array, "... N D"],
        y: Float[Array, "... M D"],
        c: Float[Array, "... 1 D"],
    ) -> tuple[Float[Array, "... N D"], Float[Array, "... M D"]]:
        # multi-stream cross attention
        hx, gate_x = self.mod_x_attn(x, c)
        hy, gate_y = self.mod_y_attn(y, c)
        hx, hy = self.attention(hx, hy)
        x = x + hx * gate_x
        y = y + hy * gate_y

        # independent feed-forward blocks
        hx, gate_x = self.mod_x_mlp(x, c)
        hy, gate_y = self.mod_y_mlp(y, c)
        x = x + self.mlp_x(hx) * gate_x
        y = y + self.mlp_y(hy) * gate_y
        return x, y


class MMDiT(nnx.Module):
    """Multi-modal DiT over param tokens x and WDM patches y; outputs (dx, x_mle, y_recon)."""

    def __init__(
        self,
        x_dim: int,
        y_channels: int,
        hidden_dim: int,
        num_heads: int,
        num_blocks: int,
        expand: int = 2,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        # embeddings
        self.x_embed = nnx.Sequential(
            FeedForward(x_dim, hidden_dim, hidden_dim, rngs=rngs, **kwargs),
            # NOTE: no positional embedding => permutation-invariant
            # PositionalEmbed(hidden_dim, axis=-2, rngs=rngs, **kwargs)
        )
        self.y_embed = nnx.Sequential(
            Patchify(y_channels, hidden_dim, rngs=rngs, **kwargs),
            PositionalEmbed(hidden_dim, axis=-3, rngs=rngs, **kwargs),
            PositionalEmbed(hidden_dim, axis=-2, rngs=rngs, **kwargs),
        )
        self.c_embed = nnx.Sequential(
            SinusoidalEmbed(hidden_dim, rngs=rngs, **kwargs),
            FeedForward(hidden_dim, hidden_dim, hidden_dim, rngs=rngs, **kwargs),
        )

        # blocks stacked along a leading axis so __call__ can scan over them
        @nnx.scan(in_axes=nnx.Carry, length=num_blocks)
        def make_block(rngs: nnx.Rngs) -> tuple[nnx.Rngs, MMDiTBlock]:
            block = MMDiTBlock(hidden_dim, num_heads, expand, rngs=rngs, **kwargs)
            return rngs, block

        _, self.blocks = make_block(rngs)

        # output heads
        self.x_modulation = Modulation(hidden_dim, rngs=rngs, **kwargs)
        self.x_unembed = FeedForward(
            hidden_dim, hidden_dim, 2 * x_dim, rngs=rngs, **kwargs
        )
        self.y_unembed = UnPatchify(y_channels, hidden_dim, rngs=rngs, **kwargs)

    def __call__(
        self,
        x: Float[Array, "... N F"],
        y: Float[Array, "... H W C"],
        t: Float[Array, "..."],
    ) -> tuple[
        Float[Array, "... N F"],
        Float[Array, "... N F"],
        Float[Array, "... H W C"],
    ]:
        # embeddings
        c = self.c_embed(t)  # (...) -> (... D)
        c = c[..., None, :]  # (... D) -> (... 1 D)
        x = self.x_embed(x)  # (... N F) -> (... N D)
        y = self.y_embed(y)  # (... H W C) -> (... h w D)
        *_, H, W, D = y.shape  # save y shape for unembedding
        y = rearrange(y, "... h w d -> ... (h w) d")

        # scan the stacked blocks, use remat to avoid OOM
        @nnx.scan(in_axes=(0, nnx.Carry, None), out_axes=nnx.Carry)
        @nnx.remat
        def scan_blocks(
            block: MMDiTBlock,
            carry: tuple[Float[Array, "... N D"], Float[Array, "... M D"]],
            c: Float[Array, "... 1 D"],
        ) -> tuple[Float[Array, "... N D"], Float[Array, "... M D"]]:
            x, y = carry
            return block(x, y, c)

        x, y = scan_blocks(self.blocks, (x, y), c)

        # x output head
        x, _ = self.x_modulation(x, c)
        x = self.x_unembed(x)  # (... N D) -> (... N 2F)
        dx, x = jnp.split(x, 2, axis=-1)

        # y output head
        y = rearrange(y, "... (h w) d -> ... h w d", h=H, w=W)
        y = self.y_unembed(y)  # (... h w D) -> (... H W C)
        return dx, x, y
