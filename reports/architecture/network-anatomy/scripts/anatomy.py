"""LisaFlow's anatomy per band x size: stage shapes, parameter leaves and FLOPs per part, all abstract.

XLA's cost analysis of each part lowered alone gives the per-block split; a jaxpr walk that multiplies
scans by their length gives the matmul FLOPs of the whole encode, decode, sampler and training step.
A CPU job (anatomy.sbatch beside this file); from the repo root:
    JAX_PLATFORMS=cpu uv run --frozen python reports/architecture/network-anatomy/scripts/anatomy.py [config ...]
"""

import argparse
import json
import math
import subprocess
import time
import warnings

warnings.filterwarnings("ignore")
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from canna.lisa import config
from canna.lisa.network import LisaFlow, reference_frame, stft_tokens
from canna.lisa.optim import is_muon
from canna.lisa.problem import LisaGB
from canna.lisa.sample import ODE_STEPS, sample_posterior
from canna.lisa.train import losses, train_sample

REPORT = Path(__file__).parents[1]
OUT = REPORT / "data" / "anatomy.jsonl"
CONFIGS = [f"{b}-{s}" for b in ("narrow", "full") for s in ("XS", "S", "B", "L", "XL")]


def is_abstract(a) -> bool:
    return isinstance(a, jax.ShapeDtypeStruct)


def sds(*shape: int, dtype=jnp.float32) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(tuple(shape), dtype)


def traced(fn, *args):
    """fn as a function of args' abstract leaves (modules included), and those leaves."""
    dynamic, static = eqx.partition(args, is_abstract)
    return (lambda d: fn(*eqx.combine(d, static))), dynamic


def shape_of(fn, *args):
    return jax.eval_shape(*traced(fn, *args))


def xla_flops(fn, *args) -> float:
    """XLA's flop count of fn(*args) on the lowered HLO (a loop body counts once)."""
    g, d = traced(fn, *args)
    cost = jax.jit(g).lower(d).cost_analysis()
    cost = cost[0] if isinstance(cost, list) else cost
    return float(cost["flops"])


def inner_jaxprs(params: dict):
    for value in params.values():
        for item in value if isinstance(value, (tuple, list)) else (value,):
            if hasattr(item, "eqns"):
                yield item
            elif hasattr(getattr(item, "jaxpr", None), "eqns"):
                yield item.jaxpr


def walk(jaxpr, branch=max) -> int:
    """2·m·n·k over every dot_general, a scan's body times its length, a cond by `branch` of its branches."""
    total = 0
    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        if name == "dot_general":
            (contract, _), _ = eqn.params["dimension_numbers"]
            k = math.prod(eqn.invars[0].aval.shape[i] for i in contract)
            total += 2 * k * math.prod(eqn.outvars[0].aval.shape)
            continue
        assert name != "while", "a while loop has no static trip count"
        inner = [walk(j, branch) for j in inner_jaxprs(eqn.params)]
        if name == "cond":
            total += branch(inner)
        else:
            total += sum(inner) * (eqn.params["length"] if name == "scan" else 1)
    return total


def matmul_flops(fn, *args, branch=max) -> int:
    g, d = traced(fn, *args)
    return walk(jax.make_jaxpr(g)(d).jaxpr, branch)


def first_block(blocks):
    """One block of the stacked (scanned) blocks, abstract."""
    return jax.tree.map(
        lambda a: sds(*a.shape[1:], dtype=a.dtype) if is_abstract(a) else a, blocks
    )


def block_parts(block, D: int, H: int, M: int, S: int, Z: int) -> dict:
    """XLA flops of each part of one block, per stream; the attention core is the block total minus the rest.

    M data tokens, S sources (also the register count), Z sinks.
    """
    K = D // H
    att = block.attention
    c = sds(D)
    rope = lambda T: (sds(T, K // 2), sds(T, K // 2))
    per_token = lambda m, n: (
        0.0 if m is None else xla_flops(lambda m, z: jax.vmap(m)(z), m, sds(n, D))
    )
    mod = lambda m, n: (
        0.0 if m is None else xla_flops(lambda m, z, c: m(z, c), m, sds(n, D), c)
    )
    heads = lambda proj, n, r: xla_flops(
        lambda a, p, z, r: a.heads(p, z, r), att, proj, sds(n, D), r
    )
    parts = {
        "data": dict(
            modulation=mod(block.mod_y_attn, M) + mod(block.mod_y_mlp, M),
            qkv=heads(att.qkv_proj_y, M, rope(M)),
            out=per_token(att.out_proj_y, M),
            mlp=per_token(block.mlp_y, M),
        ),
        "registers": dict(
            modulation=mod(block.mod_r_attn, S) + mod(block.mod_r_mlp, S),
            qkv=heads(att.qkv_proj_r, S, None),
            out=per_token(att.out_proj_r, S),
            mlp=per_token(block.mlp_r, S),
        ),
        "sources": dict(
            modulation=mod(block.mod_x_attn, S) + mod(block.mod_x_mlp, S),
            qkv=heads(att.qkv_proj_x, S, rope(S)),
            out=per_token(att.out_proj_x, S),
            mlp=per_token(block.mlp_x, S),
        ),
        "sinks": dict(
            modulation=mod(block.mod_s_attn, Z) + mod(block.mod_s_mlp, Z),
            qkv=heads(att.qkv_proj_s, Z, None),
            out=per_token(att.out_proj_s, Z),
            mlp=per_token(block.mlp_s, Z),
        ),
    }
    encode_fn = lambda b, y, c, r, reg: b.encode(y, c, r, reg)
    decode_fn = lambda b, x, c, k, v, r, s: b.decode(x, c, (k, v), r, s)
    encode_args = (block, sds(M, D), c, rope(M), sds(S, D))
    decode_args = (block, sds(S, D), c, sds(M + S, H, K), sds(M + S, H, K), rope(S), sds(Z, D))
    encode = xla_flops(encode_fn, *encode_args)
    decode = xla_flops(decode_fn, *decode_args)
    parts["data"]["core"] = encode - sum(parts["data"].values()) - sum(parts["registers"].values())
    parts["sources"]["core"] = (
        decode - sum(parts["sources"].values()) - sum(parts["sinks"].values())
    )
    walked = dict(
        encode=matmul_flops(encode_fn, *encode_args),
        decode=matmul_flops(decode_fn, *decode_args),
    )
    return dict(parts=parts, xla=dict(encode=encode, decode=decode), matmul=walked)


def anatomy(name: str, problem: LisaGB) -> dict:
    cfg = config.load(name)
    net = cfg.network
    D, H, L = net["hidden_dim"], net["num_heads"], net["num_blocks"]

    def build(dtype, attention_dtype) -> LisaFlow:
        return eqx.filter_eval_shape(
            lambda: LisaFlow(
                problem,
                **net,
                dtype=dtype,
                attention_dtype=attention_dtype,
                param_dtype=jnp.float32,
                key=jr.key(cfg.seed),
            )
        )

    # FLOPs in float32 throughout; the cache's dtype from the trainer's precision
    flow = build(jnp.float32, None)
    trained = build(
        jnp.bfloat16 if cfg.bf16 == "backbone" else jnp.float32,
        None if cfg.bf16 == "none" else jnp.bfloat16,
    )
    batch = jax.eval_shape(
        lambda key: jax.vmap(lambda k: train_sample(problem, k, jnp.zeros(cfg.draws)))(
            jr.split(key, 1)
        ),
        jr.key(0),
    )
    one = jax.tree.map(lambda a: sds(*a.shape[1:], dtype=a.dtype), batch)
    y, f = one.y, one.f
    z = sds(*one.z0.shape[1:], dtype=one.z0.dtype)
    t = sds(dtype=one.t.dtype)

    # shapes along the forward
    tokens = shape_of(stft_tokens, y, flow.patch)
    tokens_y = shape_of(lambda m, z: jax.vmap(m)(z), flow.y_embed, tokens)
    delta, cache = shape_of(lambda m, y, f: m.encode(y, f), flow, y, f)
    cache_trained = shape_of(lambda m, y, f: m.encode(y, f), trained, y, f)[1]
    frame = shape_of(lambda d, f: reference_frame(problem, d, f), delta, f)
    out = shape_of(lambda m, *a: m(*a), flow, z, t, frame, f, cache)
    N, S = tokens_y.shape[0], z.shape[0]
    Z = flow.s_slot.shape[0]
    M = N
    row = lambda name, shape, what, dtype="float32": dict(
        name=name, shape=list(shape), what=what, dtype=dtype
    )
    stages = [
        row("y", y.shape, "whitened rfft window: re A, E, T, im A, E, T", str(y.dtype)),
        row(
            "stft",
            tokens.shape,
            "per token: asinh|z| and gated phase of its band's time samples",
        ),
        row(
            "tokens_y",
            tokens_y.shape,
            "data tokens: y_embed (linear) + y_mlp(features of the token's lowest and highest f)",
        ),
        row(
            "c_f",
            (flow.f_mlp.linear1.in_features,),
            "cos and sin of the window's lowest and highest f over freq_periods, into f_mlp -> data-stream conditioning [D]",
        ),
        row(
            "registers",
            flow.frame_slot.shape,
            "frame registers: c_f + frame_slot, one per f0-ordered source; they join the data tokens in encode",
        ),
        row(
            "cache",
            cache_trained[0].shape,
            "keys (and values) of every block's data tokens then registers, per observation",
            str(cache_trained[0].dtype),
        ),
        row(
            "delta",
            delta.shape,
            "frame head: frame_modulation + frame_unembed of the registers, (delta_mu, delta_s) per source",
            str(delta.dtype),
        ),
        row(
            "frame",
            (2, *frame[0].shape),
            "mu and log sigma of x1: the prior's moments shifted and scaled by delta",
            str(frame[0].dtype),
        ),
        row(
            "z",
            z.shape,
            "flow coordinates in the frame, x = mu + sigma z",
            str(z.dtype),
        ),
        row(
            "features_x",
            (S, flow.x_embed.in_features),
            "z, x, radius cos kφ and sin kφ of both helicity amplitudes of x (k = 2..6), mu, log sigma",
        ),
        row(
            "f0_x",
            (S, flow.f0_mlp.linear1.in_features),
            "cos and sin features of f0 in Hz, into f0_mlp",
        ),
        row(
            "tokens_x", (S, D), "source tokens: x_embed (linear) + f0_mlp"
        ),
        row(
            "c",
            (flow.c_mlp.linear1.in_features,),
            "sinusoids of t and log10(1 - t), into c_mlp, plus the window's f_mlp -> c [D]",
        ),
        row("sinks", flow.s_slot.shape, "attention sinks: c + s_slot"),
        row(
            "keys",
            (S + Z + M + S,),
            "keys each source/sink query attends: own S + Z, cached N data + S registers",
        ),
        row("out", out.shape, "mean and log std of z1 per source coordinate", str(out.dtype)),
    ]

    leaves = [
        dict(
            path=jax.tree_util.keystr(path),
            shape=list(leaf.shape),
            muon=bool(is_muon(path, leaf)),
        )
        for path, leaf in jax.tree_util.tree_leaves_with_path(
            eqx.filter(
                flow, lambda a: is_abstract(a) and jnp.issubdtype(a.dtype, jnp.inexact)
            )
        )
    ]

    # per-part XLA flops: one middle block, the last block, the embeddings and the head
    vec = lambda m, n: xla_flops(lambda m, z: jax.vmap(m)(z), m, sds(n[0], n[1]))
    embed = dict(
        stft=xla_flops(stft_tokens, y, flow.patch),
        y_embed=vec(flow.y_embed, tokens.shape),
        y_mlp=vec(flow.y_mlp, (N, flow.y_mlp.linear1.in_features)),
        f_mlp=xla_flops(
            lambda m, z: m(z), flow.f_mlp, sds(flow.f_mlp.linear1.in_features)
        ),
        c_mlp=xla_flops(
            lambda m, z: m(z), flow.c_mlp, sds(flow.c_mlp.linear1.in_features)
        ),
        x_embed=vec(flow.x_embed, (S, flow.x_embed.in_features)),
        f0_mlp=vec(flow.f0_mlp, (S, flow.f0_mlp.linear1.in_features)),
        x_modulation=xla_flops(
            lambda m, z, c: m(z, c), flow.x_modulation, sds(S, D), sds(D)
        ),
        x_unembed=vec(flow.x_unembed, (S, D)),
        frame_modulation=xla_flops(
            lambda m, z, c: m(z, c), flow.frame_modulation, sds(S, D), sds(D)
        ),
        frame_unembed=vec(flow.frame_unembed, (S, D)),
    )
    blocks = dict(
        middle=block_parts(first_block(flow.backbone.blocks), D, H, M, S, Z),
        last=block_parts(flow.backbone.last, D, H, M, S, Z),
    )
    xla_whole = dict(
        encode=xla_flops(lambda m, y, f: m.encode(y, f), flow, y, f),
        decode=xla_flops(lambda m, *a: m(*a), flow, z, t, frame, f, cache),
    )

    # scan-aware matmul flops: whole forward, sampler (1 and 2 draws of one observation), training step
    sampler = lambda n: matmul_flops(
        lambda m, u, y, f: sample_posterior(problem, m, u, y, f),
        flow,
        sds(n, *z.shape, dtype=z.dtype),
        y,
        f,
    )
    objective = lambda m, b: losses(m, problem, b)[0]
    params = lambda m: eqx.filter(m, eqx.is_inexact_array)
    grad = lambda m, b: jax.grad(
        lambda p: objective(
            eqx.combine(p, eqx.filter(m, eqx.is_inexact_array, inverse=True)), b
        )
    )(params(m))
    matmul = dict(
        encode=matmul_flops(lambda m, y, f: m.encode(y, f), flow, y, f),
        decode=matmul_flops(lambda m, *a: m(*a), flow, z, t, frame, f, cache),
        sampler_1=sampler(1),
        sampler_2=sampler(2),
        train_forward=matmul_flops(objective, flow, batch),
        train_grad=matmul_flops(grad, flow, batch),
    )
    return dict(
        config=name,
        hidden_dim=D,
        num_heads=H,
        num_blocks=L,
        head_dim=D // H,
        patch=flow.patch,
        window_bins=problem.window_bins,
        response_points=problem.response_points,
        data_tokens=N,
        sources=S,
        sinks=Z,
        draws=cfg.draws,
        ode_steps=ODE_STEPS,
        bf16=cfg.bf16,
        cache_bytes=sum(math.prod(a.shape) * a.dtype.itemsize for a in cache_trained),
        stages=stages,
        leaves=leaves,
        flops=dict(embed=embed, blocks=blocks, xla_whole=xla_whole, matmul=matmul),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="*", default=CONFIGS)
    parser.add_argument("--out", type=Path, default=OUT)
    cli = parser.parse_args()
    done = set()
    if cli.out.exists():
        done = {
            json.loads(line)["config"]
            for line in cli.out.read_text().splitlines()
            if line.strip()
        }
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    problems = {}
    for name in cli.configs:
        if name in done:
            print(f"{name}: already in {cli.out}", flush=True)
            continue
        start = time.perf_counter()
        band = name.split("-")[0]
        if band not in problems:
            problems[band] = LisaGB(**config.load(name).problem)
        record = anatomy(name, problems[band]) | dict(
            commit=commit, seconds=round(time.perf_counter() - start, 1)
        )
        m, b = record["flops"]["matmul"], record["flops"]["blocks"]["middle"]
        print(
            f"{name}: params {sum(math.prod(l['shape']) for l in record['leaves']):,} "
            f"encode {m['encode']:.4g} decode {m['decode']:.4g} sampler1 {m['sampler_1']:.4g} "
            f"block xla/matmul enc {b['xla']['encode'] / max(b['matmul']['encode'], 1):.3f} "
            f"dec {b['xla']['decode'] / max(b['matmul']['decode'], 1):.3f} "
            f"whole xla/matmul enc {record['flops']['xla_whole']['encode'] / m['encode']:.3f} "
            f"dec {record['flops']['xla_whole']['decode'] / m['decode']:.3f} ({record['seconds']} s)",
            flush=True,
        )
        cli.out.parent.mkdir(parents=True, exist_ok=True)
        with open(cli.out, "a") as file:
            file.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
