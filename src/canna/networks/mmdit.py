from jaxtyping import Float, Array
import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange

from .utils import FeedForward, Modulation, SinusoidalEmbed


class PositionalEmbed(nnx.Module):
    def __init__(self, dim: int, max_len: int, axis: int = -2, **kwargs):
        self.axis = axis
        self.max_len = max_len
        self.embed = SinusoidalEmbed(dim, **kwargs)

    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        assert x.shape[self.axis] <= self.max_len, "sequence longer than max_len"
        pos = jnp.arange(x.shape[self.axis], dtype=x.dtype) / self.max_len
        shape = [1] * x.ndim
        shape[self.axis], shape[-1] = pos.shape[0], self.embed.dim
        return x + self.embed(pos).reshape(shape)


class Fold2d(nnx.Module):
    def __call__(self, z: Float[Array, "... H W C"]) -> Float[Array, "... h w D"]:
        return rearrange(z, "... (h p) (w q) c -> ... h w (p q c)", p=2, q=2)


class Patchify(nnx.Module):
    def __init__(self, channels: int, dim: int, stages: int = 4, **kwargs):
        assert stages >= 1, "stages must be at least 1"
        narrowest = 4 ** (stages - 1)
        assert dim % narrowest == 0, f"hidden_dim must be divisible by {narrowest}"
        dims = [channels, *(dim // 4**stage for stage in reversed(range(stages)))]

        down = lambda i, o: (
            Fold2d(),
            nnx.Linear(4 * i, o, **kwargs),
            nnx.RMSNorm(o, **kwargs),
            nnx.silu,
        )

        layers = []
        for i, o in zip(dims, dims[1:-1]):
            layers.extend(down(i, o))
        layers.append(Fold2d())
        layers.append(nnx.Linear(4 * dims[-2], dim, **kwargs))
        self.layers = nnx.Sequential(*layers)

    def __call__(self, y: Float[Array, "... H W C"]) -> Float[Array, "... h w D"]:
        return self.layers(y)


class Unfold2d(nnx.Module):
    def __call__(self, z: Float[Array, "... h w D"]) -> Float[Array, "... H W C"]:
        return rearrange(z, "... h w (p q c) -> ... (h p) (w q) c", p=2, q=2)


class UnPatchify(nnx.Module):
    def __init__(self, channels: int, dim: int, stages: int = 4, **kwargs):
        assert stages >= 1, "stages must be at least 1"
        narrowest = 4 ** (stages - 1)
        assert dim % narrowest == 0, f"hidden_dim must be divisible by {narrowest}"
        dims = [*(dim // 4**stage for stage in range(stages)), channels]

        up = lambda i, o: (
            nnx.RMSNorm(i, **kwargs),
            nnx.silu,
            nnx.Linear(i, 4 * o, **kwargs),
            Unfold2d(),
        )
        layers = []
        layers.append(nnx.Linear(dim, 4 * dims[1], **kwargs))
        layers.append(Unfold2d())
        for i, o in zip(dims[1:], dims[2:]):
            layers.extend(up(i, o))
        self.layers = nnx.Sequential(*layers)

    def __call__(self, y: Float[Array, "... h w D"]) -> Float[Array, "... H W C"]:
        return self.layers(y)


class MultiStreamAttention(nnx.Module):
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
        x = self.qkv_proj_x(x)
        y = self.qkv_proj_y(y)
        h = jnp.concat([x, y], axis=-2)

        qkv = rearrange(h, "... N (H D) -> ... N H D", H=self.num_heads)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q, k = self.q_norm(q), self.k_norm(k)

        *batch, N, H, D = q.shape
        flatten = lambda z: z.reshape(-1, N, H, D)
        h = jax.nn.dot_product_attention(
            flatten(q), flatten(k), flatten(v), implementation="xla"
        )
        h = h.reshape(*batch, N, H, D)
        h = rearrange(h, "... N H D -> ... N (H D)")

        x, y = jnp.split(h, [x.shape[-2]], axis=-2)
        x = self.out_proj_x(x)
        y = self.out_proj_y(y)
        return x, y


class MMDiTBlock(nnx.Module):
    def __init__(self, dim: int, num_heads: int, expand: int, **kwargs):
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
        hx, gate_x = self.mod_x_attn(x, c)
        hy, gate_y = self.mod_y_attn(y, c)
        hx, hy = self.attention(hx, hy)
        x = x + hx * gate_x
        y = y + hy * gate_y

        hx, gate_x = self.mod_x_mlp(x, c)
        hy, gate_y = self.mod_y_mlp(y, c)
        x = x + self.mlp_x(hx) * gate_x
        y = y + self.mlp_y(hy) * gate_y
        return x, y


class MMDiTFlow(nnx.Module):
    def __init__(
        self,
        x_shape: tuple[int, int],
        y_shape: tuple[int, int, int],
        hidden_dim: int,
        num_heads: int,
        num_blocks: int,
        expand: int = 2,
        patch_stages: int = 4,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        _, x_dim = x_shape
        height, width, y_dim = y_shape
        patch = 2**patch_stages
        assert height % patch == 0 and width % patch == 0, f"H, W must divide {patch}"

        self.x_embed = nnx.Sequential(
            FeedForward(x_dim, hidden_dim, hidden_dim, rngs=rngs, **kwargs),
            # skip positional embedding for x stream -> permutation-invariant
            # PositionalEmbed(hidden_dim, axis=-2, rngs=rngs, **kwargs),
        )
        self.y_embed = nnx.Sequential(
            Patchify(y_dim, hidden_dim, patch_stages, rngs=rngs, **kwargs),
            PositionalEmbed(hidden_dim, height // patch, axis=-3, rngs=rngs, **kwargs),
            PositionalEmbed(hidden_dim, width // patch, axis=-2, rngs=rngs, **kwargs),
        )
        self.c_embed = nnx.Sequential(
            SinusoidalEmbed(hidden_dim, rngs=rngs, **kwargs),
            FeedForward(hidden_dim, hidden_dim, hidden_dim, rngs=rngs, **kwargs),
        )

        @nnx.scan(in_axes=nnx.Carry, length=num_blocks)
        def make_block(rngs: nnx.Rngs) -> tuple[nnx.Rngs, MMDiTBlock]:
            block = MMDiTBlock(hidden_dim, num_heads, expand, rngs=rngs, **kwargs)
            return rngs, block

        _, self.blocks = make_block(rngs)

        self.x_modulation = Modulation(hidden_dim, rngs=rngs, **kwargs)
        self.x_unembed = FeedForward(
            hidden_dim, hidden_dim, 2 * x_dim, rngs=rngs, **kwargs
        )
        self.y_unembed = UnPatchify(
            y_dim, hidden_dim, patch_stages, rngs=rngs, **kwargs
        )

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
        c = self.c_embed(t)
        c = c[..., None, :]
        x = self.x_embed(x)
        y = self.y_embed(y)
        *_, H, W, D = y.shape
        y = rearrange(y, "... h w d -> ... (h w) d")

        # lax.scan over split state: nnx.scan blocks autodiff w.r.t. the input
        graphdef, state = nnx.split(self.blocks)

        @jax.checkpoint
        def scan_blocks(
            carry: tuple[Float[Array, "... N D"], Float[Array, "... M D"]],
            block: nnx.State,
        ) -> tuple[tuple[Float[Array, "... N D"], Float[Array, "... M D"]], None]:
            x, y = carry
            return nnx.merge(graphdef, block)(x, y, c), None

        (x, y), _ = jax.lax.scan(scan_blocks, (x, y), state)

        x, _ = self.x_modulation(x, c)
        x = self.x_unembed(x)
        dx, x = jnp.split(x, 2, axis=-1)

        y = rearrange(y, "... (h w) d -> ... h w d", h=H, w=W)
        y = self.y_unembed(y)
        return dx, x, y
