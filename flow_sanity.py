"""Flow-matching sanity check: posterior p(x | y), multi-source.

SIMPLIFIED=1 restricts inference to [f0, fdot, A, psi], marginalizing the rest as
nuisances (via the lisa_simplified drop-in); otherwise the full 8-param problem.
The two share this script: a single flag swaps the lisa interface and output tag.

Knobs (env): N_SOURCES (>= 1, default 1), NOISE_SCALE (default 1.0), RUN_SECONDS,
CHECKPOINT_INTERVAL, SIMPLIFIED, TAG_SUFFIX.
"""

import functools
import itertools
import os
import time

import corner
import equinox as eqx
import jax
import jax.random as jr
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import optax
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

from src import networks
from src import lisa

# problem knobs (env)
N_SOURCES = int(os.environ.get("N_SOURCES", "1"))
NOISE_SCALE = float(os.environ.get("NOISE_SCALE", "1.0"))
RUN_SECONDS = float(os.environ.get("RUN_SECONDS", "1800"))
T_OBS = lisa.MONTH
CHECKPOINT_INTERVAL = float(os.environ.get("CHECKPOINT_INTERVAL", "1800"))  # [s]
SIMPLIFIED = os.environ.get("SIMPLIFIED", "0") not in ("0", "", "false", "False")

# the SIMPLIFIED flag swaps in a drop-in lisa with x restricted to [f0, fdot, A, psi]
if SIMPLIFIED:
    from src import lisa_simplified as lisa

assert N_SOURCES >= 1, "N_SOURCES must be >= 1"
assert N_SOURCES <= 4, "N_SOURCES must be <= 4"

# flow and training hyperparameters
SEED = 0
HIDDEN_DIM = 256
NUM_BLOCKS = 4
NUM_HEADS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 512

# eval parameters and pathnames
ODE_STEPS = 16
N_EVAL = 3  # number of injections corner-plotted each checkpoint
N_POSTERIOR = 1024  # posterior draws per corner plot
TAG = (
    f"{N_SOURCES}src_all_noise{NOISE_SCALE:g}"
    + ("_simplified" if SIMPLIFIED else "")
    + os.environ.get("TAG_SUFFIX", "")
)
CHECKPOINT_PATH = f"checkpoint_flow_{TAG}.eqx"
LOSS_PLOT_PATH = f"training_loss_flow_{TAG}.pdf"
CORNER_PATH = f"flow_corner_{TAG}.pdf"


if __name__ == "__main__":
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.local_device_count()}")
    print(
        f"N_SOURCES={N_SOURCES}  NOISE_SCALE={NOISE_SCALE:g}  SIMPLIFIED={SIMPLIFIED}"
    )
    print(f"inferring {lisa.PARAMETER_NAMES}")
    key = jr.key(SEED)

    key, key_mock = jr.split(key)
    xt, dx, t, y, x0, x1, params, datastream = lisa.get_train_batch(
        key_mock, batch_size=2, n_sources=N_SOURCES, t_obs=T_OBS
    )
    print(f"x:{xt.shape[1:]}  y:{y.shape[1:]}")

    key, key_init = jr.split(key)
    flow = networks.MMDiT(
        x_dim=xt.shape[-1],
        y_dim=y.shape[-1],
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        key=key_init,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY),
    )
    opt_state = optimizer.init(eqx.filter(flow, eqx.is_array))

    @functools.partial(eqx.filter_jit, donate="all")
    def train_step(flow, opt_state, xt, dx, t, y, x1):
        def loss_fn(flow):
            # compute the flow-matching loss
            dx_pred, x_mle_pred = jax.vmap(flow)(xt, y, t)
            loss_flow = (dx_pred - dx) ** 2
            # auxiliary regression loss on the MLE point estimate of x1
            _, loss_reg = jax.vmap(lisa.match_sources)(x_mle_pred, x1)
            return loss_flow.mean() + loss_reg.mean()

        loss, grads = eqx.filter_value_and_grad(loss_fn)(flow)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(flow, eqx.is_array)
        )
        flow = eqx.apply_updates(flow, updates)
        return flow, opt_state, loss

    @eqx.filter_jit
    def sample_posterior(flow, x0, y):
        return jax.vmap(lambda xi: flow.push(xi, y, ODE_STEPS, lisa.exp_map))(x0)

    def checkpoint_and_eval(key, flow, losses, elapsed):
        eqx.tree_serialise_leaves(CHECKPOINT_PATH, flow)

        plt.figure()
        plt.loglog(losses)
        plt.xlabel("step")
        plt.ylabel("flow-matching MSE")
        plt.grid()
        plt.title(f"flow p(x|y) ({TAG})")
        plt.savefig(LOSS_PLOT_PATH)
        plt.close()

        print(f"[{elapsed/3600:.2f} h] loss={np.mean(losses[-200:]):.5f}", flush=True)

        # one corner plot per injection, all pages in a single PDF
        with PdfPages(CORNER_PATH) as pdf:
            for j, key_eval in enumerate(jr.split(key, N_EVAL)):
                _, _, _, y, x0, x1, params, datastream = lisa.get_train_batch(
                    key_eval,
                    batch_size=N_POSTERIOR,
                    n_sources=N_SOURCES,
                    t_obs=T_OBS,
                    noise_scale=NOISE_SCALE,
                )
                post = sample_posterior(flow, x0, y[0])  # (N,S,x_dim)
                x_dim = xt.shape[-1]
                truth = x1[0]  # (S, x_dim)
                # move to physical params
                post = np.asarray(lisa.prior_inverse_cdf((post + 1.0) / 2.0))
                truth = np.asarray(lisa.prior_inverse_cdf((truth + 1.0) / 2.0))
                post_flat = post.reshape(N_POSTERIOR, -1)

                labels = [
                    f"{n}_{s}" for s in range(N_SOURCES) for n in lisa.PARAMETER_NAMES
                ]

                fig = corner.corner(
                    post_flat,
                    labels=labels,
                    color="C1",
                    range=[(-1, 1)] * (N_SOURCES * x_dim),
                    show_titles=True,
                    title_fmt=".2f",
                    hist_kwargs={"density": True},
                )
                # the flow is permutation invariant across sources, so every
                # relabelling of the truth is an equally valid posterior mode
                for sigma in itertools.permutations(range(N_SOURCES)):
                    truth_perm = truth[list(sigma)].reshape(-1)
                    corner.overplot_lines(fig, truth_perm, color="black")
                    corner.overplot_points(
                        fig,
                        truth_perm[None, :],
                        marker="s",
                        color="black",
                        markersize=4,
                    )
                fig.legend(
                    handles=[
                        mlines.Line2D([], [], color="C1", label="flow posterior"),
                        mlines.Line2D([], [], color="black", label="truth (all perms)"),
                    ],
                    loc="upper right",
                    fontsize=12,
                    frameon=False,
                )
                title = f"flow posterior, injection {j} ({TAG}, {elapsed/3600:.1f}h)"
                fig.suptitle(title, y=1.0)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    losses = []
    t0 = time.monotonic()
    last_save = t0
    pbar = tqdm(total=RUN_SECONDS, unit="s", desc="training")
    while time.monotonic() - t0 < RUN_SECONDS:
        key, key_batch = jr.split(key)
        xt, dx, t, y, x0, x1, params, datastream = lisa.get_train_batch(
            key_batch,
            batch_size=BATCH_SIZE,
            n_sources=N_SOURCES,
            t_obs=T_OBS,
            noise_scale=NOISE_SCALE,
        )
        flow, opt_state, loss = train_step(flow, opt_state, xt, dx, t, y, x1)
        losses.append(loss.item())
        pbar.update(int(time.monotonic() - t0 - pbar.n))
        pbar.set_postfix(loss=f"{loss.item():.5f}")
        if time.monotonic() - last_save >= CHECKPOINT_INTERVAL:
            key, key_eval = jr.split(key)
            checkpoint_and_eval(key_eval, flow, losses, time.monotonic() - t0)
            last_save = time.monotonic()
    pbar.close()

    checkpoint_and_eval(key, flow, losses, time.monotonic() - t0)
    print(f"[checkpoint] final saved -> {CHECKPOINT_PATH}")
    print(f"saved {CORNER_PATH}")
