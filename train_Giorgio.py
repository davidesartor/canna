"""Train a conditional flow-matching network for Galactic-Binary inference.

A clean rewrite of ``train.py``. The key difference: the network is conditioned
on the **WDM image of the datastream** (real amortized inference of the posterior
p(params | data)), not on a noisy copy of the ground-truth parameters as in the
sanity-check draft (``lisa.get_train_batch`` overwrites ``y = x1 + noise``).

The problem (single GB, 4 parameters [f0, fdot, A, psi], fixed sky) is defined
once in :mod:`src.gb_problem` and shared with ``test_Giorgio.py`` and
``GB_inference.ipynb``.

Usage (quick "is it learning?" run):
    python train_Giorgio.py --seconds 180
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # shared GPU

import time
import argparse
import functools

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from src import lisa, gb_problem


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds", type=float, default=180.0, help="wall-clock training budget")
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--t-obs", type=float, default=lisa.MONTH_s)
    p.add_argument("--checkpoint", type=str, default="checkpoint_Giorgio.eqx")
    p.add_argument("--loss-plot", type=str, default="training_loss_Giorgio.pdf")
    args = p.parse_args()

    print(f"JAX backend: {jax.default_backend()}, devices: {jax.local_device_count()}")

    key = jr.key(args.seed)

    # --- shapes from one mock batch ---------------------------------------
    key, key_mock = jr.split(key)
    xt, dx, t, y = gb_problem.get_train_batch(key_mock, batch_size=2, t_obs=args.t_obs)
    print(f"data shapes: xt={xt.shape[1:]}, dx={dx.shape[1:]}, t={t.shape[1:]}, y={y.shape[1:]}")

    # --- model + optimizer -------------------------------------------------
    key, key_init = jr.split(key)
    flow = gb_problem.build_flow(
        y_dim=y.shape[-1],
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        key=key_init,
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr))
    opt_state = optimizer.init(eqx.filter(flow, eqx.is_array))

    @functools.partial(eqx.filter_jit, donate="all")
    def train_step(flow, opt_state, batch):
        xt, dx, t, y = batch

        def loss_fn(flow):
            pred = jax.vmap(flow)(xt, y, t)
            return jnp.mean((pred - dx) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(flow)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(flow, eqx.is_array))
        flow = eqx.apply_updates(flow, updates)
        return flow, opt_state, loss

    # Reference: predicting the mean velocity (0) gives MSE = Var(x1-x0) = 2/12.
    baseline = 2.0 / 12.0
    print(f"predict-the-mean baseline MSE = {baseline:.4f} (must drop below this to be useful)")

    # --- training loop -----------------------------------------------------
    losses, ema = [], None
    step = 0
    t_start = time.monotonic()
    while time.monotonic() - t_start < args.seconds:
        key, key_batch = jr.split(key)
        batch = gb_problem.get_train_batch(key_batch, batch_size=args.batch_size, t_obs=args.t_obs)
        flow, opt_state, loss = train_step(flow, opt_state, batch)
        loss = float(loss)
        losses.append(loss)
        ema = loss if ema is None else 0.98 * ema + 0.02 * loss
        step += 1
        if step % 20 == 0:
            elapsed = time.monotonic() - t_start
            print(f"  step {step:4d}  t={elapsed:5.1f}s  loss={loss:.4f}  ema={ema:.4f}")

    print(f"\ntrained {step} steps in {time.monotonic() - t_start:.1f}s, final ema loss = {ema:.4f}")

    # --- checkpoint + loss curve ------------------------------------------
    eqx.tree_serialise_leaves(args.checkpoint, flow)
    print(f"[checkpoint] saved -> {args.checkpoint}")

    plt.figure()
    plt.plot(losses, lw=0.6, alpha=0.5, label="loss")
    # EMA curve for readability
    ema_curve, e = [], None
    for l in losses:
        e = l if e is None else 0.98 * e + 0.02 * l
        ema_curve.append(e)
    plt.plot(ema_curve, color="C1", lw=2, label="EMA")
    plt.axhline(baseline, color="k", ls="--", lw=1, label="predict-mean baseline")
    plt.xlabel("step")
    plt.ylabel("flow-matching MSE")
    plt.yscale("log")
    plt.legend()
    plt.title("train_Giorgio: conditional flow-matching loss")
    plt.tight_layout()
    plt.savefig(args.loss_plot)
    print(f"[loss curve] saved -> {args.loss_plot}")


if __name__ == "__main__":
    main()
