"""Where a LISA training step spends its time: the batch, forward + backward, the optimizer.

Each part runs as a lax.scan of N repetitions inside one jit, timed after a warm-up call, so
dispatch overhead is amortised the same way the trainer's fused epoch amortises it. The
optimizer is timed alone on fixed gradients, for schedule-free NorMuon (the trainer's), plain
Muon and plain AdamW. The full step is then timed at several token patch sizes. The full-step
line also reports the parameter count, the compile time and the process's peak GPU memory.
--params-only builds every band x size abstractly instead, writes their parameter splits to
data/params.jsonl and times nothing.

Run from the repo root:
    LISA_CONFIG=narrow-B uv run --extra cuda python reports/benchmarks/step-time-memory/scripts/bench_step.py [--patches 2 4 8 16] [--steps 50] [--no-parts]
    uv run python reports/benchmarks/step-time-memory/scripts/bench_step.py --params-only
or on a cluster GPU, with step.sbatch beside this file.
On a laptop GPU, time long loops only: its clock alternates between two power states.
"""

import argparse
import json
import time
import warnings
from functools import partial

warnings.filterwarnings("ignore")
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from einops import rearrange

from canna.lisa import config
from canna.lisa.network import LisaFlow
from canna.lisa.optim import is_muon
from canna.lisa.problem import LisaGB
from canna.lisa.train import losses, setup, train_epoch, train_sample


def timed(fn, *args, repeats: int = 3) -> float:
    """Best wall time of fn(*args) over `repeats` calls, after one warm-up (compile) call."""
    jax.block_until_ready(fn(*args))
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - start)
    return best


def count_params(name: str) -> dict:
    """The config's parameter split by backbone part, top-level field and optimizer, built with filter_eval_shape."""
    cfg = config.load(name)
    problem = LisaGB(**cfg.problem)
    flow = eqx.filter_eval_shape(
        lambda: LisaFlow(problem, **cfg.network, param_dtype=jnp.float32, key=jr.key(cfg.seed))
    )
    is_param = lambda x: isinstance(x, jax.ShapeDtypeStruct) and jnp.issubdtype(x.dtype, jnp.inexact)
    groups, fields, optimizers = {}, {}, {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(eqx.filter(flow, is_param)):
        name = jax.tree_util.keystr(path)
        group = (
            "embeddings and the rest"
            if not name.startswith(".backbone")
            else "attention"
            if ".attention." in name
            else "adaLN modulation"
            if ".mod_" in name
            else "MLPs"
        )
        field = path[0].name
        optimizer = "Muon" if is_muon(path, leaf) else "AdamW"
        groups[group] = groups.get(group, 0) + leaf.size
        fields[field] = fields.get(field, 0) + leaf.size
        optimizers[optimizer] = optimizers.get(optimizer, 0) + leaf.size
    record = dict(
        config=cfg.name,
        hidden_dim=cfg.network["hidden_dim"],
        num_heads=cfg.network["num_heads"],
        num_blocks=cfg.network["num_blocks"],
        window_bins=problem.window_bins,
        params=sum(fields.values()),
        param_groups=groups,
        param_fields=fields,
        param_optimizer=optimizers,
    )
    print(json.dumps(record), flush=True)
    return record


def bench(steps: int, patch: int | None, parts: bool) -> None:
    cfg = config.load() if patch is None else config.load(network=dict(patch=patch))
    problem, tx, state = setup(cfg)
    batch_size = cfg.batch_size
    draws = cfg.draws
    observations = batch_size // draws
    sample = partial(train_sample, problem)
    # batch_size samples, draws per observation, as train_epoch draws them
    make_batch = lambda key: jax.vmap(sample)(
        jr.split(key, observations),
        rearrange(
            jr.uniform(jr.fold_in(key, 1), (batch_size,)), "(b k) -> b k", k=draws
        ),
    )
    tokens = problem.window_bins // state.flow.patch
    params = sum(x.size for x in jax.tree.leaves(eqx.filter(state.flow, eqx.is_inexact_array)))
    label = f"{cfg.name} bf16 {cfg.bf16} {params / 1e6:.1f}M params, patch {state.flow.patch} ({tokens} tokens, {observations} observations x {draws} draws, {cfg.micro_batches} micro-batches)"

    @eqx.filter_jit
    def epoch(state):
        return train_epoch(
            problem,
            tx,
            state,
            batch_size,
            steps,
            draws,
            cfg.micro_batches,
        )

    start = time.perf_counter()
    jax.block_until_ready(epoch(state))
    first = time.perf_counter() - start
    full = timed(epoch, state) / steps
    stats = jax.devices()[0].memory_stats() or {}
    peak = stats.get("peak_bytes_in_use", float("nan")) / 2**30
    print(
        f"{label}: full step {1e3 * full:7.1f} ms, compile {first - steps * full:5.1f} s, peak {peak:5.2f} GiB",
        flush=True,
    )
    if not parts:
        return

    @jax.jit
    def batches(key):
        def body(k, _):
            k, kb = jr.split(k)
            b = make_batch(kb)
            return k, jnp.sum(b.x1) + jnp.sum(b.y)

        return jax.lax.scan(body, key, length=steps)[1]

    batch = make_batch(jr.key(1))
    flow_loss = lambda f, b: losses(f, problem, b)[0]

    @eqx.filter_jit
    def fwd_bwd(flow, batch):
        def body(f, _):
            loss, g = eqx.filter_value_and_grad(flow_loss)(f, batch)
            # feed the gradient back so the loop cannot be hoisted
            f = eqx.apply_updates(f, jax.tree.map(lambda x: 1e-12 * x, g))
            return f, loss

        return jax.lax.scan(body, flow, length=steps)[1]

    @eqx.filter_jit
    def forward(flow, batch):
        def body(f, _):
            return f, flow_loss(f, batch)

        return jax.lax.scan(body, flow, length=steps)[1]

    params = eqx.filter(state.flow, eqx.is_inexact_array)
    grads = jax.tree.map(lambda p: 1e-3 * jnp.ones_like(p), params)

    def optimizer_loop(tx):
        @jax.jit
        def run(params, opt_state):
            def body(carry, _):
                p, s = carry
                u, s = tx.update(grads, s, p)
                return (optax.apply_updates(p, u), s), None

            return jax.lax.scan(body, (params, opt_state), length=steps)[0]

        return timed(run, params, tx.init(params)) / steps

    muon_dimensions = lambda params: jax.tree_util.tree_map_with_path(
        lambda path, leaf: (
            optax.contrib.MuonDimensionNumbers(leaf.ndim - 1, leaf.ndim - 2)
            if is_muon(path, leaf)
            else None
        ),
        params,
    )
    muon_only = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.contrib.muon(
            1e-3,
            mu_dtype=jnp.bfloat16,
            consistent_rms=0.2,
            adam_b1=0.0,
            adam_b2=0.95,
            muon_weight_dimension_numbers=muon_dimensions,
        ),
    )
    rows = [
        ("batch generation", timed(batches, jr.key(0)) / steps),
        ("forward", timed(forward, state.flow, batch) / steps),
        ("forward + backward", timed(fwd_bwd, state.flow, batch) / steps),
        ("optimizer: schedule-free NorMuon, NS bfloat16", optimizer_loop(tx)),
        ("optimizer: Muon alone, NS float32", optimizer_loop(muon_only)),
        (
            "optimizer: AdamW alone",
            optimizer_loop(
                optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-3))
            ),
        ),
    ]
    for name, seconds in rows:
        print(
            f"    {name:46s} {1e3 * seconds:7.1f} ms  ({seconds / full:5.1%} of the step)",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patches", type=int, nargs="*", default=[])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--parts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--params-only", action="store_true")
    parser.add_argument("--out", default="reports/benchmarks/step-time-memory/data/params.jsonl")
    cli = parser.parse_args()
    if cli.params_only:
        records = [
            count_params(f"{band}-{size}")
            for band in ("narrow", "full")
            for size in ("XS", "S", "B", "L", "XL")
        ]
        with open(cli.out, "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in records)
        raise SystemExit
    print(
        f"backend {jax.default_backend()}, {jax.devices()[0].device_kind}", flush=True
    )
    bench(cli.steps, None, parts=cli.parts)
    for patch in cli.patches:
        bench(cli.steps, patch, parts=False)
