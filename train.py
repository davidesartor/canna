import os
import time
import functools
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from src import lisa, networks

# problem
SEED = 0
N_SOURCES = 2
T_OBS = lisa.MONTH_s

# model
HIDDEN_DIM = 256
NUM_BLOCKS = 4
NUM_HEADS = 8

# training
LEARNING_RATE = 1e-4
BATCH_SIZE = 512
EPOCH_TIME_BUDGET_s = 5 * 60  # 5 minutes per epoch
EPOCHS = (48 * 60 * 60) // EPOCH_TIME_BUDGET_s  # 48h total
# Output paths (env-overridable so parallel runs on different GPUs don't clobber
# each other). Defaults preserve the original single-run behaviour.
RUN_TAG = os.environ.get("RUN_TAG", "")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", f"checkpoint{RUN_TAG}.eqx")
LOSS_PLOT_PATH = os.environ.get("LOSS_PLOT_PATH", f"training_loss{RUN_TAG}.pdf")


if __name__ == "__main__":
    ########################################
    # mock data to get the shapes
    key, key_mock = jr.split(jr.key(SEED))
    x, dx, t, y = lisa.get_train_batch(
        key_mock, batch_size=2, n_sources=N_SOURCES, t_obs=T_OBS
    )
    print(f"data shapes:")
    print(f"  x={x.shape[1:]}")
    print(f"  dx={dx.shape[1:]}")
    print(f"  t={t.shape[1:]}")
    print(f"  y={y.shape[1:]}")

    ########################################
    # build model and optimizer
    key, key_init = jr.split(key)
    flow = networks.MMDiT(
        x_dim=x.shape[-1],
        y_dim=y.shape[-1],
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        key=key_init,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(LEARNING_RATE),
    )
    opt_state = optimizer.init(eqx.filter(flow, eqx.is_array))

    ########################################
    # define the training/validation loop
    @functools.partial(eqx.filter_jit, donate="all")
    def train_step(flow, opt_state, batch):
        xt, dx, t, y = batch

        def loss_fn(flow):
            pred = jax.vmap(flow)(xt, y, t)
            return jnp.mean((pred - dx) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(flow)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(flow, eqx.is_array)
        )
        flow = eqx.apply_updates(flow, updates)
        return flow, opt_state, loss

    def train_epoch(flow, opt_state, key):
        ########################################
        # training
        time_start = time.monotonic()
        pbar = tqdm(total=EPOCH_TIME_BUDGET_s, unit="s", desc="epoch")
        losses = []
        while time.monotonic() - time_start < EPOCH_TIME_BUDGET_s:
            key, key_batch = jr.split(key)
            batch = lisa.get_train_batch(
                key_batch, batch_size=BATCH_SIZE, n_sources=N_SOURCES, t_obs=T_OBS
            )
            flow, opt_state, loss = train_step(flow, opt_state, batch)
            losses.append(loss.item())
            pbar.update(int(time.monotonic() - time_start - pbar.n))
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.refresh()
        pbar.close()

        ########################################
        # validation
        # TODO add some validation step here (?)

        ########################################
        # checkpointing
        eqx.tree_serialise_leaves(CHECKPOINT_PATH, flow)
        print(f"\n[checkpoint] saved at → {CHECKPOINT_PATH}")
        return flow, opt_state, jnp.array(losses)

    ########################################
    # run the training
    loss_mean = []
    loss_min = []
    loss_max = []
    for epoch, key in enumerate(jr.split(key, EPOCHS)):
        print(f"\n[epoch {epoch}]")
        flow, opt_state, epoch_losses = train_epoch(flow, opt_state, key)
        loss_mean.append(jnp.mean(epoch_losses))
        loss_min.append(jnp.min(epoch_losses))
        loss_max.append(jnp.max(epoch_losses))

        # visualize the training loss over epochs
        print(
            f"[epoch {epoch}] loss={loss_mean[-1]:.4f} ({loss_min[-1]:.4f}, {loss_max[-1]:.4f})"
        )
        plt.loglog(loss_mean, label="Loss")
        plt.fill_between(range(len(loss_mean)), loss_min, loss_max, alpha=0.3)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.title("Training Loss Over Epochs")
        plt.savefig(LOSS_PLOT_PATH)
        plt.clf()

    ##########################################
    # testing
    # TODO add some testing step here (?)
