from functools import partial
from jaxtyping import Array, Float, Key
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from einops import rearrange

from .utils import FeedForward, Modulation, apply_stacked, stacked_linear


def rotate(
    z: Float[Array, "T H D"], cos: Float[Array, "T P"], sin: Float[Array, "T P"]
):
    """Rotary position embedding: rotate each (first half, second half) pair of a head's dims
    by its token's angle, in float32, and cast back. P = D / 2 angles per token, shared by
    every head, so q . k depends on the difference of the two tokens' positions. The values
    are rotated too and the output turned back by the query's angle, as GTA (arXiv:2310.10375),
    so what a query reads also arrives relative to its own position."""
    z32 = z.astype(jnp.float32)
    a, b = jnp.split(z32, 2, axis=-1)
    cos, sin = cos[:, None, :], sin[:, None, :]
    return jnp.concat([a * cos - b * sin, a * sin + b * cos], axis=-1).astype(z.dtype)


class MultiStreamAttention(eqx.Module):
    num_heads: int = eqx.field(static=True)
    attention_dtype: jnp.dtype | None = eqx.field(static=True)
    qkv_proj_x: eqx.nn.Linear
    qkv_proj_y: eqx.nn.Linear
    out_proj_x: eqx.nn.Linear
    out_proj_y: eqx.nn.Linear | None
    qk_gain: Float[Array, " H"] | None
    qkv_proj_s: eqx.nn.Linear | None
    out_proj_s: eqx.nn.Linear | None

    def __init__(
        self,
        dim: int,
        num_heads: int,
        use_bias: bool = False,
        qk_gain: bool = False,
        sinks: bool = False,
        attention_dtype: jnp.dtype | None = None,
        last: bool = False,
        *,
        key: Key[Array, ""],
        **kwargs,
    ):
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        keys = iter(jr.split(key, 4))
        self.num_heads = num_heads
        # the scores and the weighted sum run in attention_dtype (None: the input's), the softmax in float32
        self.attention_dtype = attention_dtype
        # q, k and v as three stacked maps, so Muon orthogonalises each on its own
        self.qkv_proj_x = stacked_linear(
            3, dim, dim, use_bias=use_bias, key=next(keys), **kwargs
        )
        self.qkv_proj_y = stacked_linear(
            3, dim, dim, use_bias=use_bias, key=next(keys), **kwargs
        )
        self.out_proj_x = eqx.nn.Linear(dim, dim, use_bias, key=next(keys), **kwargs)
        self.out_proj_y = eqx.nn.Linear(dim, dim, use_bias, key=next(keys), **kwargs)
        # a learned per-head logit scale on top of the rms norm, starting as none
        self.qk_gain = (
            jnp.ones((num_heads,), kwargs.get("dtype", jnp.float32))
            if qk_gain
            else None
        )
        # the sink stream draws from its own key, so the projections above stay as they were
        sink_keys = jr.split(jr.fold_in(key, 1), 2)
        self.qkv_proj_s, self.out_proj_s = (
            (
                stacked_linear(
                    3, dim, dim, use_bias=use_bias, key=sink_keys[0], **kwargs
                ),
                eqx.nn.Linear(dim, dim, use_bias, key=sink_keys[1], **kwargs),
            )
            if sinks
            else (None, None)
        )
        # the one-way stack's last block: nothing reads its y or sink outputs
        if last:
            self.out_proj_y = self.out_proj_s = None

    def heads(
        self, linear: eqx.nn.Linear, z: Float[Array, "T D"], rope=None
    ) -> tuple[Float[Array, "T H K"], Float[Array, "T H K"], Float[Array, "T H K"]]:
        """One stream's q, k and v: rms-normalised q and k (q times the per-head gain), all three rotated by rope."""
        qkv = jax.vmap(partial(apply_stacked, linear), out_axes=1)(z)
        q, k, v = rearrange(qkv, "s n (h d) -> s n h d", h=self.num_heads)

        rms = lambda z: jax.lax.rsqrt(
            jnp.mean(jnp.square(z), axis=-1, keepdims=True) + 1e-6
        )
        q, k = q * rms(q), k * rms(k)
        if self.qk_gain is not None:
            q = q * self.qk_gain[:, None].astype(q.dtype)
        if rope is not None:
            q, k, v = rotate(q, *rope), rotate(k, *rope), rotate(v, *rope)
        return q, k, v

    def attend(
        self,
        q: Float[Array, "T H K"],
        k: Float[Array, "U H K"],
        v: Float[Array, "U H K"],
        rope=None,
    ) -> Float[Array, "T D"]:
        """rope: the queries' (cos, sin), to turn the rotated values' sum back."""
        out_dtype = v.dtype
        q, k, v = (a.astype(self.attention_dtype or out_dtype) for a in (q, k, v))
        # cuDNN flash attention tiles the softmax instead of materialising the N x N
        # logits, and that matrix -- batch * heads * tokens^2, in f32 whatever the
        # compute dtype -- is what caps the batch size. It only accepts fp16/bf16, on an
        # Ampere or newer gpu, and implementation=None still falls back to xla, so choose explicitly.
        flash = (
            q.dtype in (jnp.bfloat16, jnp.float16)
            and jax.default_backend() == "gpu"
            and float(jax.devices()[0].compute_capability) >= 8.0
        )

        # dot_product_attention wants a leading batch axis
        h = jax.nn.dot_product_attention(
            q[None], k[None], v[None], implementation="cudnn" if flash else "xla"
        )[0]
        if rope is not None:
            cos, sin = rope
            h = rotate(h, cos, -sin)
        return rearrange(h.astype(out_dtype), "n h d -> n (h d)")

    def __call__(
        self, x: Float[Array, "N D"], y: Float[Array, "M D"], rope=None
    ) -> tuple[Float[Array, "N D"], Float[Array, "M D"]]:
        """rope: optional (cos, sin), each [N + M, head_dim / 2], for the x then the y tokens."""
        n = x.shape[-2]
        rope_x, rope_y = (
            (None, None)
            if rope is None
            else tuple(zip(*(jnp.split(a, [n]) for a in rope)))
        )
        qkv_x = self.heads(self.qkv_proj_x, x, rope_x)
        qkv_y = self.heads(self.qkv_proj_y, y, rope_y)
        q, k, v = (jnp.concat([a, b]) for a, b in zip(qkv_x, qkv_y))
        x, y = jnp.split(self.attend(q, k, v, rope), [n])
        return jax.vmap(self.out_proj_x)(x), jax.vmap(self.out_proj_y)(y)


class MMDiTBlock(eqx.Module):
    mod_x_attn: Modulation
    mod_y_attn: Modulation
    mod_x_mlp: Modulation
    mod_y_mlp: Modulation | None
    attention: MultiStreamAttention
    mlp_x: FeedForward
    mlp_y: FeedForward | None
    mod_s_attn: Modulation | None
    mod_s_mlp: Modulation | None
    mlp_s: FeedForward | None

    def __init__(
        self,
        dim: int,
        num_heads: int,
        expand: int,
        qk_gain: bool = False,
        sinks: bool = False,
        attention_dtype: jnp.dtype | None = None,
        last: bool = False,
        *,
        key: Key[Array, ""],
        **kwargs,
    ):
        keys = iter(jr.split(key, 7))
        self.mod_x_attn = Modulation(dim, key=next(keys), **kwargs)
        self.mod_y_attn = Modulation(dim, key=next(keys), **kwargs)
        self.mod_x_mlp = Modulation(dim, key=next(keys), **kwargs)
        self.mod_y_mlp = Modulation(dim, key=next(keys), **kwargs)
        self.attention = MultiStreamAttention(
            dim,
            num_heads,
            qk_gain=qk_gain,
            sinks=sinks,
            attention_dtype=attention_dtype,
            last=last,
            key=next(keys),
            **kwargs,
        )
        self.mlp_x = FeedForward(dim, expand * dim, dim, key=next(keys), **kwargs)
        self.mlp_y = FeedForward(dim, expand * dim, dim, key=next(keys), **kwargs)
        sink_keys = iter(jr.split(jr.fold_in(key, 1), 3))
        self.mod_s_attn, self.mod_s_mlp, self.mlp_s = (
            (
                Modulation(dim, key=next(sink_keys), **kwargs),
                Modulation(dim, key=next(sink_keys), **kwargs),
                FeedForward(dim, expand * dim, dim, key=next(sink_keys), **kwargs),
            )
            if sinks
            else (None, None, None)
        )
        # the one-way stack's last block: its y and sink updates would reach nothing
        if last:
            self.mod_y_mlp = self.mlp_y = self.mod_s_mlp = self.mlp_s = None

    def __call__(
        self,
        x: Float[Array, "N D"],
        y: Float[Array, "M D"],
        c: Float[Array, " D"],
        rope=None,
    ) -> tuple[Float[Array, "N D"], Float[Array, "M D"]]:
        hx, gate_x = self.mod_x_attn(x, c)
        hy, gate_y = self.mod_y_attn(y, c)
        hx, hy = self.attention(hx, hy, rope)
        x = x + hx * gate_x
        y = y + hy * gate_y

        hx, gate_x = self.mod_x_mlp(x, c)
        hy, gate_y = self.mod_y_mlp(y, c)
        x = x + jax.vmap(self.mlp_x)(hx) * gate_x
        y = y + jax.vmap(self.mlp_y)(hy) * gate_y
        return x, y

    def encode(
        self, y: Float[Array, "M D"], c: Float[Array, " D"], rope=None
    ) -> tuple[
        Float[Array, "M D"], tuple[Float[Array, "M H K"], Float[Array, "M H K"]]
    ]:
        """The y stream alone, attending to itself only: its updated tokens (as is in a last block) and this block's y keys and values."""
        attention = self.attention
        hy, gate = self.mod_y_attn(y, c)
        q, k, v = attention.heads(attention.qkv_proj_y, hy, rope)
        dtype = attention.attention_dtype or v.dtype
        kv = (k.astype(dtype), v.astype(dtype))
        if self.mlp_y is None:
            return y, kv
        y = y + jax.vmap(attention.out_proj_y)(attention.attend(q, k, v, rope)) * gate
        hy, gate = self.mod_y_mlp(y, c)
        return y + jax.vmap(self.mlp_y)(hy) * gate, kv

    def decode(
        self,
        x: Float[Array, "N D"],
        c: Float[Array, " D"],
        kv: tuple[Float[Array, "M H K"], Float[Array, "M H K"]],
        rope=None,
        s: Float[Array, "S D"] | None = None,
    ) -> tuple[Float[Array, "N D"], Float[Array, "S D"] | None]:
        """The x stream and the sinks s, attending jointly to both and to the y keys and values `encode` left."""
        attention = self.attention
        n = x.shape[-2]
        hx, gate = self.mod_x_attn(x, c)
        q, k, v = attention.heads(attention.qkv_proj_x, hx, rope)
        if s is not None:
            # the sinks carry no position: they join x's tokens unrotated
            hs, gate_s = self.mod_s_attn(s, c)
            qkv_s = attention.heads(attention.qkv_proj_s, hs)
            q, k, v = (jnp.concat([a, b]) for a, b in zip((q, k, v), qkv_s))
        # one softmax over own and cached keys, never concatenated: under a vmap over draws
        # the shared cache would otherwise be copied once per draw
        out_dtype = v.dtype
        q, k, v = (a.astype(attention.attention_dtype or out_dtype) for a in (q, k, v))
        scale = q.shape[-1] ** -0.5
        logits = jnp.concat(
            [
                jnp.einsum("nhk,mhk->hnm", q, k, preferred_element_type=jnp.float32),
                jnp.einsum(
                    "nhk,mhk->hnm", q, kv[0], preferred_element_type=jnp.float32
                ),
            ],
            axis=-1,
        )
        weights = jax.nn.softmax(logits * scale, axis=-1).astype(v.dtype)
        own = k.shape[0]
        positioned = jnp.einsum("hnm,mhk->nhk", weights[..., :n], v[:n]) + jnp.einsum(
            "hnm,mhk->nhk", weights[..., own:], kv[1]
        )
        # GTA: x's and the cached y's values arrive rotated, so each x query turns their sum back
        # by its own angle; the sinks' values carry no position and are read as they are
        if rope is not None:
            cos, sin = rope
            positioned = positioned.at[:n].set(rotate(positioned[:n], cos, -sin))
        h = positioned + jnp.einsum("hnm,mhk->nhk", weights[..., n:own], v[n:])
        h, h_s = jnp.split(rearrange(h.astype(out_dtype), "n h d -> n (h d)"), [n])
        x = x + jax.vmap(attention.out_proj_x)(h) * gate
        if s is not None and self.mlp_s is not None:
            s = s + jax.vmap(attention.out_proj_s)(h_s) * gate_s
            hs, gate_s = self.mod_s_mlp(s, c)
            s = s + jax.vmap(self.mlp_s)(hs) * gate_s
        hx, gate = self.mod_x_mlp(x, c)
        return x + jax.vmap(self.mlp_x)(hx) * gate, s


class MMDiT(eqx.Module):
    """Two token streams attending jointly through a stack of blocks, both gated by c.

    With one_way, encode/decode run them instead, y never attending to x, so y's keys and values can be cached.
    With sinks, decode also runs a third stream of sink tokens s, attending jointly with x to x, s and the cached y.
    """

    blocks: MMDiTBlock
    last: MMDiTBlock | None

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_blocks: int,
        expand: int = 2,
        qk_gain: bool = False,
        sinks: bool = False,
        attention_dtype: jnp.dtype | None = None,
        one_way: bool = False,
        *,
        key: Key[Array, ""],
        **kwargs,
    ):
        make_block = lambda key, last=False: MMDiTBlock(
            hidden_dim,
            num_heads,
            expand,
            qk_gain,
            sinks,
            attention_dtype,
            last,
            key=key,
            **kwargs,
        )
        keys = jr.split(key, num_blocks)
        # one-way, the last block's y and sink updates reach nothing: it is built without them
        if one_way:
            assert num_blocks >= 1, "a one-way MMDiT needs a block"
            self.blocks = eqx.filter_vmap(make_block)(keys[:-1])
            self.last = make_block(keys[-1], last=True)
        else:
            self.blocks = eqx.filter_vmap(make_block)(keys)
            self.last = None

    def __call__(
        self,
        x: Float[Array, "N D"],
        y: Float[Array, "M D"],
        c: Float[Array, " D"],
        rope=None,
    ) -> tuple[Float[Array, "N D"], Float[Array, "M D"]]:
        assert self.last is None, "built one-way: run encode and decode"
        # one compiled block body, scanned over the stacked block params
        params, static = eqx.partition(self.blocks, eqx.is_array)

        @jax.checkpoint
        def scan_blocks(
            carry: tuple[Float[Array, "N D"], Float[Array, "M D"]],
            block: MMDiTBlock,
        ) -> tuple[tuple[Float[Array, "N D"], Float[Array, "M D"]], None]:
            x, y = carry
            return eqx.combine(block, static)(x, y, c, rope), None

        (x, y), _ = jax.lax.scan(scan_blocks, (x, y), params)
        return x, y

    def encode(
        self, y: Float[Array, "M D"], c: Float[Array, " D"], rope=None
    ) -> tuple[Float[Array, "L M H K"], Float[Array, "L M H K"]]:
        """Every block's y keys and values, for `decode`."""
        assert self.last is not None, "built joint: one_way=True to encode and decode"
        params, static = eqx.partition(self.blocks, eqx.is_array)

        @jax.checkpoint
        def scan_blocks(y: Float[Array, "M D"], block: MMDiTBlock):
            return eqx.combine(block, static).encode(y, c, rope)

        y, (k, v) = jax.lax.scan(scan_blocks, y, params)
        _, (k_last, v_last) = self.last.encode(y, c, rope)
        return jnp.concat([k, k_last[None]]), jnp.concat([v, v_last[None]])

    def decode(
        self,
        x: Float[Array, "N D"],
        c: Float[Array, " D"],
        cache: tuple[Float[Array, "L M H K"], Float[Array, "L M H K"]],
        rope=None,
        s: Float[Array, "S D"] | None = None,
    ) -> Float[Array, "N D"]:
        """The x stream (and the sinks s) through every block, against the y keys and values from `encode`."""
        assert self.last is not None, "built joint: one_way=True to encode and decode"
        assert s is None or self.last.mod_s_attn is not None, "built without a sink stream"
        params, static = eqx.partition(self.blocks, eqx.is_array)
        last, last_static = eqx.partition(self.last, eqx.is_array)

        @jax.checkpoint
        def scan_blocks(carry, inputs):
            block, kv = inputs
            x, s = carry
            return eqx.combine(block, static).decode(x, c, kv, rope, s), None

        @jax.checkpoint
        def last_block(block, x, s, kv):
            return eqx.combine(block, last_static).decode(x, c, kv, rope, s)[0]

        cached, cached_last = (
            jax.tree.map(lambda a: a[:-1], cache),
            jax.tree.map(lambda a: a[-1], cache),
        )
        (x, s), _ = jax.lax.scan(scan_blocks, (x, s), (params, cached))
        return last_block(last, x, s, cached_last)
