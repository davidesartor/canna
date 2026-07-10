"""Flow-matching sanity check: full 8-param posterior p(x | y), multi-source.

Generative counterpart of regression_sanity.py, and the full-parameter
extension of the committed 2-source amplitude-only flow sanity
(flow_sanity_2src_A.py). Where the regressor predicts a single point estimate
at fixed t=0, this trains the full conditional flow-matching objective (same
MMDiT backbone) and recovers the posterior by integrating the learned velocity
field from the Uniform(-1, 1) base to t=1.

Uses lisa.get_train_batch, which returns (xt, dx, t, y, x0, x1, params,
datastream) for every source's full Uniform(-1, 1) geodesic:
    f0, fdot, A, ra, dec, psi, iota, phi0.
x0/params/datastream are unused here (kept for MCMC follow-up); x1 is the
t=1 endpoint used as the regression target below.

The net's final projection is 2*x_dim, split into (dx, x_mle): dx is the
flow-matching velocity, x_mle is a direct point-estimate of x1 (the
regression_sanity.py objective), trained jointly as an auxiliary loss.

Trains with conditional flow matching, then samples the posterior for a handful
of fixed injections and corner-plots the recovered parameters vs truth. The
MMDiT flow is permutation invariant in the sources by construction, so no
relabelling of draws is needed.

Knobs (env): N_SOURCES (>= 1, default 1), NOISE_SCALE (default 0.0),
CHECKPOINT_EVERY_s. Periodic checkpoint + eval so long runs are crash-safe.
"""

import argparse
import functools
import itertools
import os
import time
import warnings

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

from src import lisa, networks

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=1800.0)
args = parser.parse_args()


# define the problem
T_OBS = lisa.MONTH
N_SOURCES = int(os.environ.get("N_SOURCES", "1"))
NOISE_SCALE = float(os.environ.get("NOISE_SCALE", "1.0"))

assert N_SOURCES >= 1, "N_SOURCES must be >= 1"
assert N_SOURCES <= 4, "N_SOURCES must be <= 4"

# define the flow and training hyperparameters
SEED = 0
HIDDEN_DIM = 256
NUM_BLOCKS = 4
NUM_HEADS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 512

# define the eval parameters and pathnames
ODE_STEPS = 16
N_EVAL = 3  # number of injections corner-plotted each checkpoint
N_POSTERIOR = 1024  # posterior draws per corner plot
TAG = f"{N_SOURCES}src_all_noise{NOISE_SCALE:g}" + os.environ.get("TAG_SUFFIX", "")
CHECKPOINT_INTERVAL = float(os.environ.get("CHECKPOINT_INTERVAL", "1800"))  # [s]
CHECKPOINT_PATH = f"checkpoint_flow_{TAG}.eqx"
LOSS_PLOT_PATH = f"training_loss_flow_{TAG}.pdf"
CORNER_PATH = f"flow_corner_{TAG}.pdf"


def main(seconds: float):
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.local_device_count()}")
    print(f"N_SOURCES={N_SOURCES}  NOISE_SCALE={NOISE_SCALE:g}")
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

                post = np.asarray(sample_posterior(flow, x0, y[0]))  # (N,S,x_dim)
                x_dim = xt.shape[-1]
                # only source slot 0 is corner-plotted (2*x_dim already OOMs);
                # the flow is permutation invariant across sources, so slot 0's
                # posterior may equally correspond to any injected source --
                # overlay every source's fiducial params as a candidate truth
                post_flat = post[:, 0]  # (N, x_dim)
                truth = np.asarray(x1[0])  # (S, x_dim)

                fig = corner.corner(
                    post_flat,
                    labels=lisa.PARAMETER_NAMES,
                    truths=truth[0],
                    truth_color="black",
                    color="C1",
                    range=[(-1, 1)] * x_dim,
                    show_titles=True,
                    title_fmt=".2f",
                    hist_kwargs={"density": True},
                )
                fiducial_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][2:]
                legend_handles = [
                    mlines.Line2D([], [], color="C1", label="flow posterior"),
                    mlines.Line2D([], [], color="black", label="src 0 truth"),
                ]
                for s in range(1, N_SOURCES):
                    c = fiducial_colors[(s - 1) % len(fiducial_colors)]
                    corner.overplot_lines(fig, truth[s], color=c)
                    corner.overplot_points(
                        fig, truth[s][None, :], marker="s", color=c, markersize=6
                    )
                    legend_handles.append(
                        mlines.Line2D([], [], color=c, label=f"src {s} truth")
                    )
                fig.legend(
                    handles=legend_handles,
                    loc="upper right",
                    fontsize=12,
                    frameon=False,
                )
                title = f"flow posterior (slot 0), injection {j} ({TAG}, {elapsed/3600:.1f}h)"
                fig.suptitle(title, y=1.0)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    losses = []
    t0 = time.monotonic()
    last_save = t0
    pbar = tqdm(total=seconds, unit="s", desc="training")
    while time.monotonic() - t0 < seconds:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=1800.0)
    args = parser.parse_args()
    main(args.seconds)
