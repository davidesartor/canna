from functools import partial
from itertools import accumulate
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


def gta_attention(
    q: Float[Array, "N H K"],
    positioned: list[tuple[Float[Array, "_ H K"], Float[Array, "_ H K"]]],
    free: list[tuple[Float[Array, "_ H K"], Float[Array, "_ H K"]]],
    rope=None,
) -> Float[Array, "N H K"]:
    """One softmax of q over every (k, v) group, never concatenated: under a vmap over draws a shared
    cache would otherwise be copied once per draw. GTA: the positioned values arrive rotated, so the
    first queries (as many as rope has rows) turn their sum back by their own angle; the free values
    (sinks, registers) carry no position and are read as they are."""
    groups = positioned + free
    logits = jnp.concat(
        [
            jnp.einsum("nhk,mhk->hnm", q, k, preferred_element_type=jnp.float32)
            for k, _ in groups
        ],
        axis=-1,
    )
    weights = jax.nn.softmax(logits * q.shape[-1] ** -0.5, axis=-1).astype(q.dtype)
    weights = jnp.split(
        weights, list(accumulate(k.shape[0] for k, _ in groups))[:-1], axis=-1
    )
    read = lambda ws, vs: sum(
        jnp.einsum("hnm,mhk->nhk", w, v) for w, (_, v) in zip(ws, vs)
    )
    h = read(weights[: len(positioned)], positioned)
    if rope is not None:
        cos, sin = rope
        n = cos.shape[0]
        h = h.at[:n].set(rotate(h[:n], cos, -sin))
    return h + read(weights[len(positioned) :], free)


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
    qkv_proj_r: eqx.nn.Linear | None
    out_proj_r: eqx.nn.Linear | None

    def __init__(
        self,
        dim: int,
        num_heads: int,
        use_bias: bool = False,
        qk_gain: bool = False,
        sinks: bool = False,
        registers: bool = False,
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
        register_keys = jr.split(jr.fold_in(key, 2), 2)
        self.qkv_proj_r, self.out_proj_r = (
            (
                stacked_linear(
                    3, dim, dim, use_bias=use_bias, key=register_keys[0], **kwargs
                ),
                eqx.nn.Linear(dim, dim, use_bias, key=register_keys[1], **kwargs),
            )
            if registers
            else (None, None)
        )
        # the one-way stack's last block: nothing reads its y or sink outputs (its registers' it does)
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
    registers: int = eqx.field(static=True)
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
    mod_r_attn: Modulation | None
    mod_r_mlp: Modulation | None
    mlp_r: FeedForward | None

    def __init__(
        self,
        dim: int,
        num_heads: int,
        expand: int,
        qk_gain: bool = False,
        sinks: bool = False,
        registers: int = 0,
        attention_dtype: jnp.dtype | None = None,
        last: bool = False,
        *,
        key: Key[Array, ""],
        **kwargs,
    ):
        keys = iter(jr.split(key, 7))
        self.registers = registers
        self.mod_x_attn = Modulation(dim, key=next(keys), **kwargs)
        self.mod_y_attn = Modulation(dim, key=next(keys), **kwargs)
        self.mod_x_mlp = Modulation(dim, key=next(keys), **kwargs)
        self.mod_y_mlp = Modulation(dim, key=next(keys), **kwargs)
        self.attention = MultiStreamAttention(
            dim,
            num_heads,
            qk_gain=qk_gain,
            sinks=sinks,
            registers=registers > 0,
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
        register_keys = iter(jr.split(jr.fold_in(key, 2), 3))
        self.mod_r_attn, self.mod_r_mlp, self.mlp_r = (
            (
                Modulation(dim, key=next(register_keys), **kwargs),
                Modulation(dim, key=next(register_keys), **kwargs),
                FeedForward(dim, expand * dim, dim, key=next(register_keys), **kwargs),
            )
            if registers
            else (None, None, None)
        )
        # the one-way stack's last block: its y and sink updates would reach nothing (its registers' do)
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
        self,
        y: Float[Array, "M D"],
        c: Float[Array, " D"],
        rope=None,
        r: Float[Array, "R D"] | None = None,
    ) -> tuple[
        Float[Array, "M D"],
        Float[Array, "R D"] | None,
        tuple[Float[Array, "M+R H K"], Float[Array, "M+R H K"]],
    ]:
        """The y stream and the registers r, attending to each other only: their updated tokens (y as is in a
        last block) and this block's keys and values, y's then r's."""
        attention = self.attention
        m = y.shape[-2]
        hy, gate = self.mod_y_attn(y, c)
        q, k, v = attention.heads(attention.qkv_proj_y, hy, rope)
        out_dtype = v.dtype
        cast = lambda *a: tuple(
            b.astype(attention.attention_dtype or out_dtype) for b in a
        )
        if r is None:
            kv = cast(k, v)
            h = attention.attend(q, k, v, rope)
        else:
            # the registers carry no position: they join y's tokens unrotated
            hr, gate_r = self.mod_r_attn(r, c)
            q_r, k_r, v_r = attention.heads(attention.qkv_proj_r, hr)
            q, k, v, q_r, k_r, v_r = cast(q, k, v, q_r, k_r, v_r)
            kv = (jnp.concat([k, k_r]), jnp.concat([v, v_r]))
            h = gta_attention(jnp.concat([q, q_r]), [(k, v)], [(k_r, v_r)], rope)
            h, h_r = jnp.split(rearrange(h.astype(out_dtype), "n h d -> n (h d)"), [m])
            r = r + jax.vmap(attention.out_proj_r)(h_r) * gate_r
            hr, gate_r = self.mod_r_mlp(r, c)
            r = r + jax.vmap(self.mlp_r)(hr) * gate_r
        if self.mlp_y is None:
            return y, r, kv
        y = y + jax.vmap(attention.out_proj_y)(h) * gate
        hy, gate = self.mod_y_mlp(y, c)
        return y + jax.vmap(self.mlp_y)(hy) * gate, r, kv

    def decode(
        self,
        x: Float[Array, "N D"],
        c: Float[Array, " D"],
        kv: tuple[Float[Array, "M+R H K"], Float[Array, "M+R H K"]],
        rope=None,
        s: Float[Array, "S D"] | None = None,
    ) -> tuple[Float[Array, "N D"], Float[Array, "S D"] | None]:
        """The x stream and the sinks s, attending jointly to both and to the y and register keys and values `encode` left."""
        attention = self.attention
        n = x.shape[-2]
        hx, gate = self.mod_x_attn(x, c)
        q, k, v = attention.heads(attention.qkv_proj_x, hx, rope)
        out_dtype = v.dtype
        cast = lambda *a: tuple(
            b.astype(attention.attention_dtype or out_dtype) for b in a
        )
        q, k, v = cast(q, k, v)
        m = kv[0].shape[0] - self.registers
        positioned = [(k, v), (kv[0][:m], kv[1][:m])]
        free = [(kv[0][m:], kv[1][m:])]
        if s is not None:
            # the sinks carry no position: they join x's tokens unrotated
            hs, gate_s = self.mod_s_attn(s, c)
            q_s, k_s, v_s = cast(*attention.heads(attention.qkv_proj_s, hs))
            q = jnp.concat([q, q_s])
            free.append((k_s, v_s))
        h = gta_attention(q, positioned, free, rope)
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
    With registers, encode also runs that many register tokens r jointly with y, unrotated; decode reads their cached
    keys and values beside y's.
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
        registers: int = 0,
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
            registers,
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
        assert self.blocks.registers == 0, "registers live in encode: build one-way"
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
        self,
        y: Float[Array, "M D"],
        c: Float[Array, " D"],
        rope=None,
        r: Float[Array, "R D"] | None = None,
    ) -> tuple[
        tuple[Float[Array, "L M+R H K"], Float[Array, "L M+R H K"]],
        Float[Array, "R D"] | None,
    ]:
        """Every block's y and register keys and values, for `decode`, and the registers r after the last block."""
        assert self.last is not None, "built joint: one_way=True to encode and decode"
        assert (
            r is None if self.last.registers == 0 else r.shape[0] == self.last.registers
        ), f"built with {self.last.registers} registers"
        params, static = eqx.partition(self.blocks, eqx.is_array)

        @jax.checkpoint
        def scan_blocks(carry, block: MMDiTBlock):
            y, r = carry
            y, r, kv = eqx.combine(block, static).encode(y, c, rope, r)
            return (y, r), kv

        (y, r), (k, v) = jax.lax.scan(scan_blocks, (y, r), params)
        _, r, (k_last, v_last) = self.last.encode(y, c, rope, r)
        return (jnp.concat([k, k_last[None]]), jnp.concat([v, v_last[None]])), r

    def decode(
        self,
        x: Float[Array, "N D"],
        c: Float[Array, " D"],
        cache: tuple[Float[Array, "L M+R H K"], Float[Array, "L M+R H K"]],
        rope=None,
        s: Float[Array, "S D"] | None = None,
    ) -> Float[Array, "N D"]:
        """The x stream (and the sinks s) through every block, against the y and register keys and values from `encode`."""
        assert self.last is not None, "built joint: one_way=True to encode and decode"
        assert (
            s is None or self.last.mod_s_attn is not None
        ), "built without a sink stream"
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
