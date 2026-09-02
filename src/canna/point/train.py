from functools import partial
from typing import NamedTuple
from jaxtyping import Array, Float, Key
from pathlib import Path
import os
import argparse
import yaml

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import equinox as eqx
from tqdm import tqdm
from .problem import NoisyPoint
from .network import PointFlow

if __name__ == "__main__":
    # headless: these run on cluster nodes with no display. Selecting the backend at
    # module level instead would hijack it for anything that merely imports the package
    # -- canna.lisa re-exports train_sample from here, so `from canna.lisa import LisaGB`
    # was silently switching notebooks to Agg and swallowing their figures.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


    class TrainSample(NamedTuple):
        xt: Float[Array, "D"]
        dx: Float[Array, "D"]
        t: Float[Array, ""]
        y: Float[Array, "D"]

    def train_sample(problem: NoisyPoint, key: Key[Array, ""]) -> TrainSample:
        """Draw one training example: conditioning, a point on the geodesic, its velocity."""
        key_p, key_o, key_x0, key_t = jr.split(key, 4)
        p = problem.sample_physical(key_p)
        y = problem.preprocess(problem.sample_observation(key_o, p))

        # sample and process flow quantities
        x0 = problem.sample_flow(key_x0)
        x1 = problem.physical_to_flow(p)
        t = jr.uniform(key_t, ())

        # geodesic and its velocity at t, in flow coordinates
        def geodesic(t: Float[Array, ""]):
            return problem.exp_map(x0, t * problem.log_map(x0, x1))

        xt = geodesic(t)
        dx = jax.jacobian(geodesic)(t)
        return TrainSample(xt=xt, dx=dx, t=t, y=y)

    # the named run .yaml sets the argparse defaults, so any CLI flag overrides it
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="B", help="name of a run .yaml")
    config_parser.add_argument(
        "--config_root", type=Path, default=Path(__file__).parent / "configs"
    )
    config_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--total_steps", type=int, default=10_000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float32"),
        help="compute dtype for the network; params are kept in float32",
    )
    parser.add_argument("--muon", action=argparse.BooleanOptionalAction, default=True)

    with open(config_args.config_root / f"{config_args.config}.yaml") as f:
        parser.set_defaults(**yaml.safe_load(f))
    args = parser.parse_args()

    # housekeeping
    run_id = f"point-{args.config}"
    out_dir: Path = args.output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "checkpoint.eqx"
    print(f"JAX backend: {jax.default_backend()}", flush=True)
    print(f"devices: {jax.local_device_count()}", flush=True)
    print(f"run {run_id} -> {out_dir}", flush=True)

    key_sample, key_network, key_train = jr.split(jr.key(args.seed), 3)
    problem = NoisyPoint(**args.problem)

    # the network is shaped by one sample of the problem
    sample = train_sample(problem, key_sample)
    flow = PointFlow(
        **args.network,
        x_shape=sample.xt.shape,
        y_shape=sample.y.shape,
        dtype=jnp.dtype(args.dtype),
        param_dtype=jnp.float32,
        key=key_network,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        (optax.contrib.muon if args.muon else optax.adamw)(
            args.learning_rate, weight_decay=args.weight_decay
        ),
    )
    opt_state = tx.init(eqx.filter(flow, eqx.is_inexact_array))

    epochs = args.total_steps // args.log_interval
    loss_history = np.full((epochs, args.log_interval), np.nan)
    if checkpoint.exists():
        flow, opt_state, loss_history = eqx.tree_deserialise_leaves(
            checkpoint, (flow, opt_state, loss_history)
        )

    # every finished epoch left its losses behind, so they count the epochs already done
    start_epoch = int(np.sum(np.isfinite(loss_history[:, 0])))
    if start_epoch:
        print(f"[checkpoint] resuming at epoch {start_epoch}", flush=True)

    @eqx.filter_jit
    def train_epoch(flow: PointFlow, opt_state: optax.OptState, key: Key[Array, ""]):
        """Fuse log_interval (draw + train step) pairs into one XLA dispatch via lax.scan."""
        params, static = eqx.partition(flow, eqx.is_array)

        def train_step(carry, key_batch: Key[Array, ""]):
            params, opt_state = carry
            flow = eqx.combine(params, static)
            batch = jax.vmap(partial(train_sample, problem))(
                jr.split(key_batch, args.batch_size)
            )

            def train_loss(flow: PointFlow) -> Float[Array, ""]:
                du_pred = jax.vmap(flow)(batch.xt, batch.t, batch.y)
                return jnp.mean(jnp.square(du_pred - batch.dx))

            loss, grads = eqx.filter_value_and_grad(train_loss)(flow)
            updates, opt_state = tx.update(
                grads, opt_state, eqx.filter(flow, eqx.is_inexact_array)
            )
            flow = eqx.apply_updates(flow, updates)
            return (eqx.filter(flow, eqx.is_array), opt_state), loss

        (params, opt_state), losses = jax.lax.scan(
            train_step, (params, opt_state), jr.split(key, args.log_interval)
        )
        return eqx.combine(params, static), opt_state, losses

    pbar = tqdm(range(start_epoch, epochs), initial=start_epoch, total=epochs)
    for epoch in pbar:
        # the epoch's key is folded from the seed, so a resume never redraws old batches
        flow, opt_state, epoch_losses = train_epoch(
            flow, opt_state, jr.fold_in(key_train, epoch)
        )
        loss_history[epoch] = jax.device_get(epoch_losses)

        # save a checkpoint and log the median of the epoch's losses
        eqx.tree_serialise_leaves(checkpoint, (flow, opt_state, loss_history))
        flow_l = np.median(loss_history[epoch])
        pbar.set_postfix(flow=f"{flow_l:.5f}")
        print(f"[epoch {epoch + 1}/{epochs}] flow={flow_l:.5f}", flush=True)

        # redraw loss curve: median per epoch, shaded 10-90 percentile spread
        xs = np.arange(1, epoch + 2) * args.log_interval
        l_lo, l_med, l_up = np.percentile(
            loss_history[: epoch + 1], [10, 50, 90], axis=1
        )
        fig, ax = plt.subplots()
        ax.loglog(xs, l_med, label="flow", color="#2a86cf", lw=2)
        ax.fill_between(xs, l_lo, l_up, color="#2a86cf", alpha=0.15)
        ax.set(xlabel="step", ylabel="loss", title=f"training losses ({run_id})")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)
        fig.savefig(os.path.join(out_dir, "losses.pdf"), bbox_inches="tight")
        plt.close(fig)

    print(f"[done] {run_id} -> {out_dir}", flush=True)
