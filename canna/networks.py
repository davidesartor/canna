import os
from jaxtyping import Float, Array, Scalar
import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange

ACTIVATION = jax.nn.silu

# remat level (read once at import): higher = more sites rematted = less memory,
# more recompute. 0 none | 1 block+patch | 2 +attention | 3 +mlp
REMAT_LEVEL = int(os.environ.get("REMAT_LEVEL", "2"))

# y reconstruction head mode (read once at import): memory diagnostic knob.
# full = run the Unpatchify head (default) | off = skip it, return a zeros WDM image.
# The head's forward activations dominate peak memory, so `off` is the real lever.
RECON_HEAD = os.environ.get("RECON_HEAD", "full").lower()
assert RECON_HEAD in ("full", "off"), RECON_HEAD


def maybe_remat(level: int):
    """Rematerialize the decorated method when REMAT_LEVEL >= level (higher = more aggressive)."""
    return lambda method: nnx.remat(method) if REMAT_LEVEL >= level else method


def rms_norm(x: Float[Array, "... D"], eps: float = 1e-8):
    ms = jnp.mean(x**2, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(ms + eps)


def adaptive_norm(
    x: Float[Array, "... N D"],
    scale: Float[Array, "... D"],
    shift: Float[Array, "... D"],
    eps: float = 1e-8,
):
    x = x - x.mean(axis=-1, keepdims=True)
    x = x / (x.std(axis=-1, keepdims=True) + eps)
    x = x * (1 + scale) + shift
    return x


class FeedForward(nnx.Module):
    """Single-hidden-layer MLP (in -> width -> out) with SiLU, batched over leading dims."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        *,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.fc_in = nnx.Linear(
            in_size, width_size, param_dtype=dtype, dtype=dtype, rngs=rngs
        )
        self.fc_out = nnx.Linear(
            width_size, out_size, param_dtype=dtype, dtype=dtype, rngs=rngs
        )

    @maybe_remat(level=3)
    def __call__(self, x: Float[Array, "... I"]) -> Float[Array, "... O"]:
        return self.fc_out(ACTIVATION(self.fc_in(x)))


class Modulation(nnx.Module):
    """adaLN modulation: maps conditioning to (shift, scale, gate); zero-init -> identity at start."""

    def __init__(self, dim: int, *, rngs: nnx.Rngs, dtype=jnp.float32):
        self.linear = nnx.Linear(
            dim,
            3 * dim,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            param_dtype=dtype,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self, c: Float[Array, "D"]
    ) -> tuple[Float[Array, "D"], Float[Array, "D"], Float[Array, "D"]]:
        shift, scale, gate = jnp.split(self.linear(ACTIVATION(c)), 3, axis=-1)
        return shift, scale, gate


class SinusoidalEmbed(nnx.Module):
    def __init__(
        self, dim: int, period: float = 2 * jnp.pi, *, rngs: nnx.Rngs, dtype=jnp.float32
    ):
        self.dim = dim
        self.period = period
        self.embed = FeedForward(2 * dim, dim, dim, rngs=rngs, dtype=dtype)

    def __call__(self, t: Float[Array, "..."]) -> Float[Array, "... D"]:
        freqs = jnp.exp(
            -jnp.log(self.period) * jnp.linspace(0, 1, self.dim, dtype=t.dtype)
        )
        angles = 2 * jnp.pi * freqs * t[..., None]
        x = jnp.concat([jnp.sin(angles), jnp.cos(angles)], axis=-1)
        return self.embed(x)


class Patchify(nnx.Module):
    """Fold a (T, F, C) WDM image into a (t, f, D) token grid; channel-last throughout.

    Each stage is a 2x2 stride-2 conv expressed as fold-patch-into-channels + per-pixel Linear.
    """

    def __init__(self, channels: int, dim: int, *, rngs: nnx.Rngs, dtype=jnp.float32):
        assert dim % 64 == 0, "hidden_dim must be divisible by 64"
        lin = lambda i, o: nnx.Linear(i, o, param_dtype=dtype, dtype=dtype, rngs=rngs)
        ln = lambda c: nnx.LayerNorm(c, param_dtype=dtype, dtype=dtype, rngs=rngs)
        # crop to a multiple of 16 so four 2x2 folds divide evenly (drops the odd Nyquist bin)
        crop = lambda z: z[: z.shape[0] // 16 * 16, : z.shape[1] // 16 * 16]
        # a 2x2 stride-2 conv = fold each spatial patch into the channel dim, then a per-pixel linear
        fold = lambda z: rearrange(z, "(h p) (w q) c -> h w (p q c)", p=2, q=2)
        self.layers = nnx.Sequential(
            crop,
            fold, lin(4 * channels, dim // 64), ln(dim // 64), ACTIVATION,
            fold, lin(dim // 16, dim // 16), ln(dim // 16), ACTIVATION,
            fold, lin(dim // 4, dim // 4), ln(dim // 4), ACTIVATION,
            fold, lin(dim, dim),
        )

    @maybe_remat(level=1)
    def __call__(self, y: Float[Array, "T F C"]) -> Float[Array, "t f D"]:
        return self.layers(y)  # (T, F, C) -> (t, f, D), channel-last throughout


class Unpatchify(nnx.Module):
    """Unfold a (t, f, D) token grid back up to a (T, F, C) WDM image; channel-last throughout.

    Each stage is a transposed 2x2 stride-2 conv expressed as per-pixel Linear + unfold-channels-into-patch.
    """

    def __init__(self, channels: int, dim: int, *, rngs: nnx.Rngs, dtype=jnp.float32):
        assert dim % 64 == 0, "hidden_dim must be divisible by 64"
        lin = lambda i, o: nnx.Linear(i, o, param_dtype=dtype, dtype=dtype, rngs=rngs)
        ln = lambda c: nnx.LayerNorm(c, param_dtype=dtype, dtype=dtype, rngs=rngs)
        # a transposed 2x2 stride-2 conv = a per-pixel linear, then unfold the channel dim into a 2x2 patch
        unfold = lambda z: rearrange(z, "h w (p q c) -> (h p) (w q) c", p=2, q=2)
        # channels decay 4x per stage: the full-res activation carries dim//64 channels
        self.layers = nnx.Sequential(
            lin(dim, dim), unfold, ln(dim // 4), ACTIVATION,
            lin(dim // 4, dim // 4), unfold, ln(dim // 16), ACTIVATION,
            lin(dim // 16, dim // 16), unfold, ln(dim // 64), ACTIVATION,
            lin(dim // 64, 4 * channels), unfold,
        )

    @maybe_remat(level=1)
    def __call__(self, y: Float[Array, "t f D"]) -> Float[Array, "T F C"]:
        return self.layers(y)  # (t, f, D) -> (T, F, C), channel-last throughout


class MultiStreamAttention(nnx.Module):
    """Joint qk-normed attention over the concatenated (x, y) token streams."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        use_bias: bool = False,
        *,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        lin = lambda i, o: nnx.Linear(
            i, o, use_bias=use_bias, param_dtype=dtype, dtype=dtype, rngs=rngs
        )
        self.qkv_proj_x = lin(dim, dim * 3)
        self.qkv_proj_y = lin(dim, dim * 3)
        self.out_proj_x = lin(dim, dim)
        self.out_proj_y = lin(dim, dim)

    @maybe_remat(level=2)
    def __call__(self, x: Float[Array, "N D"], y: Float[Array, "M D"]):
        # per-stream qkv projection, then concat into one sequence
        x = self.qkv_proj_x(x)
        y = self.qkv_proj_y(y)
        h = jnp.concat([x, y], axis=-2)

        # split heads, qk-normed attention
        qkv = rearrange(h, "... N (H D) -> ... N H D", H=self.num_heads)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q, k = rms_norm(q), rms_norm(k)
        h = jax.nn.dot_product_attention(q, k, v, implementation="xla")
        h = rearrange(h, "... N H D -> ... N (H D)")

        # split streams back apart, per-stream output projection
        x, y = jnp.split(h, [x.shape[-2]], axis=-2)
        x = self.out_proj_x(x)
        y = self.out_proj_y(y)
        return x, y


class MMDiTBlock(nnx.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        expand: int = 2,
        *,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        # modulations indexed [x_attn, y_attn, x_mlp, y_mlp]
        self.modulations = nnx.data(
            [Modulation(dim, rngs=rngs, dtype=dtype) for _ in range(4)]
        )
        self.attention = MultiStreamAttention(dim, num_heads, rngs=rngs, dtype=dtype)
        self.mlp_x = FeedForward(dim, dim, expand * dim, rngs=rngs, dtype=dtype)
        self.mlp_y = FeedForward(dim, dim, expand * dim, rngs=rngs, dtype=dtype)

    @maybe_remat(level=1)
    def __call__(
        self,
        x: Float[Array, "N D"],
        y: Float[Array, "M D"],
        c: Float[Array, "D"],
    ):
        (
            (sh_xa, sc_xa, g_xa),
            (sh_ya, sc_ya, g_ya),
            (sh_xm, sc_xm, g_xm),
            (sh_ym, sc_ym, g_ym),
        ) = (m(c) for m in self.modulations)

        # joint attention
        hx = adaptive_norm(x, sc_xa, sh_xa)
        hy = adaptive_norm(y, sc_ya, sh_ya)
        hx, hy = self.attention(hx, hy)
        x = x + hx * g_xa
        y = y + hy * g_ya

        # per-stream feed-forward
        x = x + self.mlp_x(adaptive_norm(x, sc_xm, sh_xm)) * g_xm
        y = y + self.mlp_y(adaptive_norm(y, sc_ym, sh_ym)) * g_ym
        return x, y


class MMDiT(nnx.Module):
    def __init__(
        self,
        x_dim: int,
        y_channels: int,
        hidden_dim: int,
        num_heads: int,
        num_blocks: int,
        *,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.x_dim = x_dim
        self.y_channels = y_channels

        # position embeddings (y uses 2D axial time/freq grid)
        self.x_pos_embed = SinusoidalEmbed(hidden_dim, rngs=rngs, dtype=dtype)
        self.c_pos_embed = SinusoidalEmbed(hidden_dim, rngs=rngs, dtype=dtype)
        self.y_pos_embed_t = SinusoidalEmbed(hidden_dim, rngs=rngs, dtype=dtype)
        self.y_pos_embed_f = SinusoidalEmbed(hidden_dim, rngs=rngs, dtype=dtype)

        # input embeddings (x, conditioning, and patchified y)
        self.x_embed = FeedForward(
            x_dim, hidden_dim, hidden_dim, rngs=rngs, dtype=dtype
        )
        self.c_embed = FeedForward(
            hidden_dim, hidden_dim, hidden_dim, rngs=rngs, dtype=dtype
        )
        self.y_patchify = Patchify(y_channels, hidden_dim, rngs=rngs, dtype=dtype)

        # transformer blocks
        self.blocks = nnx.data(
            [
                MMDiTBlock(hidden_dim, num_heads, rngs=rngs, dtype=dtype)
                for _ in range(num_blocks)
            ]
        )

        # output heads (x and y)
        self.x_out_modulation = Modulation(hidden_dim, rngs=rngs, dtype=dtype)
        self.x_out_unembed = FeedForward(
            hidden_dim, 2 * x_dim, hidden_dim, rngs=rngs, dtype=dtype
        )
        self.y_out_modulation = Modulation(hidden_dim, rngs=rngs, dtype=dtype)
        self.y_unpatchify = Unpatchify(y_channels, hidden_dim, rngs=rngs, dtype=dtype)

    def __call__(
        self,
        x: Float[Array, "N D"],
        y: Float[Array, "T F C"],
        t: Scalar,
    ) -> tuple[Float[Array, "N D"], Float[Array, "N D"], Float[Array, "T F C"]]:
        # embed x and conditioning
        x_pos = jnp.linspace(0, 1, len(x), dtype=x.dtype)
        x = self.x_embed(x) + self.x_pos_embed(x_pos)
        c = self.c_embed(self.c_pos_embed(t))

        # patchify y into a token grid with 2D time/freq position embeddings
        y_grid = self.y_patchify(y)  # (n_t, n_f, D)
        n_t, n_f, _ = y_grid.shape
        y = rearrange(y_grid, "t f d -> (t f) d")
        t_pos = jnp.linspace(0, 1, n_t, dtype=y.dtype)
        f_pos = jnp.linspace(0, 1, n_f, dtype=y.dtype)
        tt, ff = jnp.meshgrid(t_pos, f_pos, indexing="ij")
        y = y + self.y_pos_embed_t(tt.reshape(-1))
        y = y + self.y_pos_embed_f(ff.reshape(-1))

        # transformer blocks
        for block in self.blocks:
            x, y = block(x, y, c)

        # x output head -> (dx, x_mle)
        shift, scale, _ = self.x_out_modulation(c)
        x = self.x_out_unembed(adaptive_norm(x, scale, shift))
        dx, x_mle = jnp.split(x, 2, axis=-1)

        # y output head -> denoised WDM (RECON_HEAD gates the head for memory diagnostics)
        if RECON_HEAD == "off":
            y_recon = jnp.zeros((n_t * 16, n_f * 16, self.y_channels), dtype=y.dtype)
            return dx, x_mle, y_recon
        shift_y, scale_y, _ = self.y_out_modulation(c)
        y = adaptive_norm(y, scale_y, shift_y)
        y = rearrange(y, "(t f) d -> t f d", t=n_t)
        y_recon = self.y_unpatchify(y)
        return dx, x_mle, y_recon
