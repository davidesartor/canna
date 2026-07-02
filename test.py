import os
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np
from einops import rearrange
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import blackjax
import corner

jax.config.update("jax_enable_x64", True)

from src import lisa, networks
from wdm_transform.transforms import from_freq_to_wdm

print(f"JAX backend: {jax.default_backend()}, devices: {jax.local_device_count()}")


# problem
SEED = 0
N_SOURCES = 2
T_OBS = lisa.MONTH_s
HIDDEN_DIM = 256
NUM_BLOCKS = 4
NUM_HEADS = 8
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "checkpoint.eqx")
N_SAMPLES = 1024

# --- Build the flow and load the checkpoint ---------------------------------
key = jr.key(SEED)
key, key_mock = jr.split(key)
xt, dx, t, y = lisa.get_train_batch(
    key_mock, batch_size=2, n_sources=N_SOURCES, t_obs=T_OBS
)
flow = networks.MMDiT(
    x_dim=xt.shape[-1],
    y_dim=y.shape[-1],
    hidden_dim=HIDDEN_DIM,
    num_blocks=NUM_BLOCKS,
    num_heads=NUM_HEADS,
    key=key_mock,
)
flow = eqx.tree_deserialise_leaves(CHECKPOINT_PATH, flow)


# intentionally traded for speed.
@eqx.filter_jit
def flow_sample(x, y):
    return jax.vmap(lambda xi: flow.push(xi, y))(x)


# --- Build one self-consistent injection ------------------------------------
# We generate a ground-truth set of source parameters, the corresponding
# frequency-domain datastream (signal + noise) used by the MCMC likelihood,
# and the WDM-domain conditioning ``y`` fed to the flow. This way the flow and
# the MCMC are looking at the *same* observation.

for i in range(10):
    key, key_truth, key_noise = jr.split(key, 3)
    xt, dx, t, y = lisa.get_train_batch(
        key_truth, batch_size=1, n_sources=N_SOURCES, t_obs=T_OBS
    )
    # Ground-truth unit-cube params are the t=1 endpoint of the geodesic:
    # x1 = xt + (1 - t) * dx  (since dx = x1 - x0 and xt = x0 + t * dx).
    x1_true = xt + (1.0 - t)[:, None, None] * dx
    true_params = np.asarray(x1_true[0])  # (N_SOURCES, PARAM_DIM), unit cube [0, 1]
    y = y[0]  # remove batch dimension for flow conditioning

    key, key_flow, key_mcmc = jr.split(key, 3)
    x0 = jr.uniform(key_flow, shape=(N_SAMPLES, N_SOURCES, flow.x_dim))
    x1 = flow_sample(x0, y)
    print("flow samples shape:", x1.block_until_ready().shape)
    flow_samples = np.asarray(rearrange(x1, "n s p -> n (s p)"))

    # Tight Gaussian cluster around the truth to represent MCMC samples.
    # Shape: (N_SAMPLES, N_SOURCES * PARAM_DIM)
    mcmc_samples = np.asarray(true_params.flatten()) + 1e-2 * np.asarray(
        jr.normal(key_mcmc, shape=(N_SAMPLES, true_params.size))
    )

    # Labels and truths in unit-cube space [0, 1] — matches flow output range.
    param_names = ["f0", "fdot", "A", "ra", "dec", "psi", "iota", "phi0"]
    param_names = param_names[: xt.shape[-1]]
    labels = [f"{name}$_{s}$" for s in range(N_SOURCES) for name in param_names]
    truths = true_params.flatten()  # (N_SOURCES * PARAM_DIM,) in [0, 1]
    ndim = truths.shape[0]

    MCMC_COLOR, FLOW_COLOR = "C0", "C1"
    plt.close("all")  # prevent corner from reusing axes from a previous iteration

    # Plot flow samples first, then overplot MCMC on the same figure.
    fig = corner.corner(
        flow_samples,
        labels=labels,
        truths=truths,
        truth_color="black",
        color=FLOW_COLOR,
        show_titles=True,
        title_fmt=".2e",
        # range=[(0, 1)] * ndim,  # unit-cube axes
        hist_kwargs={"density": True},
    )
    corner.corner(
        mcmc_samples,
        fig=fig,
        color=MCMC_COLOR,
        # range=[(0, 1)] * ndim,
        hist_kwargs={"density": True},
    )

    fig.legend(
        handles=[
            mlines.Line2D([], [], color=MCMC_COLOR, label="MCMC"),
            mlines.Line2D([], [], color=FLOW_COLOR, label="Flow"),
            mlines.Line2D([], [], color="black", label="Injected truth"),
        ],
        loc="upper right",
        fontsize=14,
        frameon=False,
    )
    fig.suptitle("Posterior distribution: MCMC vs flow")
    fig.savefig(f"corner{i}.pdf", dpi=150, bbox_inches="tight")
    print(f"Saved corner plot to corner{i}.pdf")
    plt.show()
