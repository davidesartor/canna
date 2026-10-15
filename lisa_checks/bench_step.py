"""Where a LISA training step spends its time: the batch, forward + backward, the optimizer.

Each part runs as a lax.scan of N repetitions inside one jit, timed after a warm-up call, so
dispatch overhead is amortised the same way the trainer's fused epoch amortises it. The
optimizer is timed alone on fixed gradients, for schedule-free Muon (the trainer's), plain
Muon and plain AdamW. The full step is then timed at several token patch sizes.

Run from the repo root:
    .venv/bin/python lisa_checks/bench_step.py XS-fd [--patches 2 4 8 16] [--steps 50]
On a laptop GPU, time long loops only: its clock alternates between two power states.
"""

import argparse
import sys
import time
import warnings
from functools import partial

warnings.filterwarnings("ignore")
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from canna.lisa import train as T


def timed(fn, *args, repeats: int = 3) -> float:
    """Best wall time of fn(*args) over `repeats` calls, after one warm-up (compile) call."""
    jax.block_until_ready(fn(*args))
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - start)
    return best


def bench(config: str, steps: int, patch: int | None, parts: bool) -> None:
    sys.argv = ["bench", "--config", config]
    args = T.parse_args()
    if patch is not None:
        args.network = dict(args.network, patch=patch)
    state = T.TrainState.from_config(args)
    problem, batch_size, tp = state.problem, args.batch_size, args.time_power
    make_batch = jax.vmap(partial(
        T.train_sample, problem, time_power=tp, templates=state.flow.templates,
        fisher_cap=args.fisher_cap,
    ))
    tokens = problem.window_bins // args.network["patch"]
    label = f"{config} patch {args.network['patch']} ({tokens} tokens)"

    @eqx.filter_jit
    def epoch(state):
        return state.train_epoch(batch_size, steps, tp, args.aux_weight, args.fisher_cap)

    full = timed(epoch, state) / steps
    print(f"{label}: full step {1e3 * full:7.1f} ms", flush=True)
    if not parts:
        return

    @jax.jit
    def batches(key):
        def body(k, _):
            k, kb = jr.split(k)
            b = make_batch(jr.split(kb, batch_size))
            return k, jnp.sum(b.dx) + jnp.sum(b.y)
        return jax.lax.scan(body, key, length=steps)[1]

    batch = make_batch(jr.split(jr.key(1), batch_size))

    @eqx.filter_jit
    def fwd_bwd(flow, batch):
        def body(f, _):
            loss, g = eqx.filter_value_and_grad(T.flow_loss)(f, batch)
            # feed the gradient back so the loop cannot be hoisted
            f = eqx.apply_updates(f, jax.tree.map(lambda x: 1e-12 * x, g))
            return f, loss
        return jax.lax.scan(body, flow, length=steps)[1]

    @eqx.filter_jit
    def forward(flow, batch):
        def body(f, _):
            loss = T.flow_loss(f, batch)
            return f, loss
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

    muon_only = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.contrib.muon(
            1e-3, mu_dtype=jnp.bfloat16, consistent_rms=0.2, adam_b1=0.0, adam_b2=0.95,
            muon_weight_dimension_numbers=T.muon_dimensions,
        ),
    )
    fp32_args = argparse.Namespace(**{**vars(args), "newton_schulz_dtype": "float32"})
    rows = [
        ("batch generation", timed(batches, jr.key(0)) / steps),
        ("forward", timed(forward, state.flow, batch) / steps),
        ("forward + backward", timed(fwd_bwd, state.flow, batch) / steps),
        (f"optimizer: schedule-free Muon, NS {args.newton_schulz_dtype}", optimizer_loop(state.tx)),
        ("optimizer: schedule-free Muon, NS float32", optimizer_loop(T.optimizer(fp32_args))),
        ("optimizer: Muon alone, NS float32", optimizer_loop(muon_only)),
        ("optimizer: AdamW alone", optimizer_loop(optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-3)))),
    ]
    for name, seconds in rows:
        print(f"    {name:42s} {1e3 * seconds:7.1f} ms  ({seconds / full:5.1%} of the step)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--patches", type=int, nargs="*", default=[])
    parser.add_argument("--steps", type=int, default=50)
    cli = parser.parse_args()
    print(f"backend {jax.default_backend()}, {jax.devices()[0].device_kind}", flush=True)
    bench(cli.config, cli.steps, None, parts=True)
    for patch in cli.patches:
        bench(cli.config, cli.steps, patch, parts=False)
