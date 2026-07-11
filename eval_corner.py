"""Standalone posterior corner-plot eval for a flow_sanity.py checkpoint.

flow_sanity.py's own eval used to call get_train_batch(batch_size=N_POSTERIOR)
just to get N_POSTERIOR IID x0 draws, generating (and immediately discarding)
N_POSTERIOR-1 redundant FFT-heavy injections along the way -- that batch of
FFTs is what OOMs once MAX_FREQUENCY is large. This script instead makes one
cheap get_train_batch(batch_size=1) call per injection to build its truth and
conditioning signal y, then draws the N_POSTERIOR posterior samples by
chunking the ODE push over EVAL_BATCH_SIZE-sized batches of x0 (pure flow
forward passes, no FFT, flow held fixed). That's light enough to run on a
small GPU (e.g. a 1080Ti) even for checkpoints too big to eval in-line with
training.

Knobs (env): mirror flow_sanity.py's problem knobs so TAG/CHECKPOINT_PATH
resolve to the same checkpoint (N_SOURCES, SIMPLIFIED, TAG_SUFFIX), plus
MIN_SNR (SNR floor applied to eval injections, default 0 = natural difficulty),
N_EVAL, N_POSTERIOR, EVAL_BATCH_SIZE, PLOT_PHYSICAL.
"""

import itertools
import os

import corner
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

jax.config.update("jax_enable_x64", True)

from src import networks
from src import lisa

# problem knobs (env) -- must match the checkpoint being loaded
N_SOURCES = int(os.environ.get("N_SOURCES", "1"))
SIMPLIFIED = os.environ.get("SIMPLIFIED", "0") not in ("0", "", "false", "False")
MIN_SNR = float(os.environ.get("MIN_SNR", "0.0"))  # SNR floor applied to eval injections
PLOT_PHYSICAL = os.environ.get("PLOT_PHYSICAL", "0") not in ("0", "", "false", "False")

if SIMPLIFIED:
    from src import lisa_simplified as lisa

assert N_SOURCES >= 1, "N_SOURCES must be >= 1"
assert N_SOURCES <= 4, "N_SOURCES must be <= 4"

# flow hyperparameters -- must match the checkpoint being loaded
SEED = int(os.environ.get("SEED", "0"))
HIDDEN_DIM = 256
NUM_BLOCKS = 4
NUM_HEADS = 8

T_OBS = lisa.MONTH
ODE_STEPS = 16
N_EVAL = int(os.environ.get("N_EVAL", "3"))  # number of injections corner-plotted
N_POSTERIOR = int(os.environ.get("N_POSTERIOR", "1024"))  # posterior draws per corner plot
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "64"))  # posterior draws per chunk

TAG = (
    f"{N_SOURCES}src"
    + ("_simplified" if SIMPLIFIED else "")
    + os.environ.get("TAG_SUFFIX", "")
)
CHECKPOINT_PATH = f"checkpoint_flow_{TAG}.eqx"
CORNER_PATH = f"flow_corner_{TAG}.pdf"

TEX_NAMES = {
    "f0": "f_0",
    "fdot": r"\dot f",
    "A": "A",
    "ra": r"\alpha",
    "dec": r"\delta",
    "psi": r"\psi",
    "iota": r"\iota",
    "phi0": r"\phi_0",
}


if __name__ == "__main__":
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.local_device_count()}")
    print(f"N_SOURCES={N_SOURCES}  MIN_SNR={MIN_SNR:g}  SIMPLIFIED={SIMPLIFIED}")
    key = jr.key(SEED)

    key, key_mock = jr.split(key)
    xt, dx, t, y, x0, x1, params, datastream = lisa.get_train_batch(
        key_mock, batch_size=2, n_sources=N_SOURCES, t_obs=T_OBS
    )
    x_dim = xt.shape[-1]
    print(f"x:{xt.shape[1:]}  y:{y.shape[1:]}")

    key, key_init = jr.split(key)
    flow_skeleton = networks.MMDiT(
        x_dim=x_dim,
        y_dim=y.shape[-1],
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        key=key_init,
    )
    flow = eqx.tree_deserialise_leaves(CHECKPOINT_PATH, flow_skeleton)
    print(f"loaded {CHECKPOINT_PATH}")

    @eqx.filter_jit
    def sample_posterior_chunk(flow, x0_chunk, y):
        return jax.vmap(lambda xi: flow.push(xi, y, ODE_STEPS, lisa.exp_map))(x0_chunk)

    with PdfPages(CORNER_PATH) as pdf:
        for j, key_eval in enumerate(jr.split(key, N_EVAL)):
            key_inj, key_x0 = jr.split(key_eval)

            # one FFT-heavy call (batch_size=1) to build the truth injection + y
            _, _, _, y, _, x1, params, _ = lisa.get_train_batch(
                key_inj,
                batch_size=1,
                n_sources=N_SOURCES,
                t_obs=T_OBS,
                snr_threshold=MIN_SNR,
            )
            y0 = y[0]

            # accumulate posterior draws in chunks: no FFT, just flow forward passes
            n_chunks = -(-N_POSTERIOR // EVAL_BATCH_SIZE)  # ceil division
            post_chunks = []
            n_remaining = N_POSTERIOR
            for key_chunk in jr.split(key_x0, n_chunks):
                n = min(EVAL_BATCH_SIZE, n_remaining)
                x0_chunk = jr.uniform(
                    key_chunk, shape=(n, N_SOURCES, x_dim), minval=-1.0, maxval=1.0
                )
                post_chunks.append(sample_posterior_chunk(flow, x0_chunk, y0))
                n_remaining -= n
            post = jnp.concatenate(post_chunks, axis=0)  # (N_POSTERIOR,S,x_dim), x1-space

            if PLOT_PHYSICAL:
                # move posterior to physical params
                truth = np.array(params[0])  # (S, x_dim), physical params
                post = np.array(lisa.prior_inverse_cdf((post + 1.0) / 2.0))
                post_flat = post.reshape(N_POSTERIOR, -1)

                # f0 and A are log-uniform in the prior: plot log10 of these
                # dims directly (rather than a log axis scale) so the corner
                # marginals/contours aren't crushed into a sliver near 0
                LOG_PARAMS = {"f0", "A"}

                def tex_label(n, s):
                    sym = TEX_NAMES.get(n, n)
                    body = f"\\log_{{10}} {sym}" if n in LOG_PARAMS else sym
                    return f"${body}$" + (f" (src {s})" if N_SOURCES > 1 else "")

                labels = [
                    tex_label(n, s)
                    for s in range(N_SOURCES)
                    for n in lisa.PARAMETER_NAMES
                ]

                log_mask = np.array(
                    [
                        n in LOG_PARAMS
                        for _ in range(N_SOURCES)
                        for n in lisa.PARAMETER_NAMES
                    ]
                )
                post_flat = np.where(log_mask, np.log10(post_flat), post_flat)
                truth = np.where(
                    log_mask.reshape(N_SOURCES, x_dim), np.log10(truth), truth
                )
            else:
                # native flow space, uniform in [-1, 1]
                truth = np.array(x1[0])  # (S, x_dim)
                post_flat = np.array(post).reshape(N_POSTERIOR, -1)
                labels = [
                    f"${TEX_NAMES.get(n, n)}$"
                    + (f" (src {s})" if N_SOURCES > 1 else "")
                    for s in range(N_SOURCES)
                    for n in lisa.PARAMETER_NAMES
                ]

            # matched-filter SNR of each injected source (alone) and combined
            params_phys = jnp.array(params[0])  # (S, 8), always physical
            per_source_snr = np.array(
                jax.vmap(lambda p: lisa.optimal_snr(p[None, :], t_obs=T_OBS))(
                    params_phys
                )
            )
            total_snr = float(lisa.optimal_snr(params_phys, t_obs=T_OBS))
            snr_str = "SNR: " + ", ".join(
                f"src{s}={v:.1f}" for s, v in enumerate(per_source_snr)
            )
            if N_SOURCES > 1:
                snr_str += f" (combined {total_snr:.1f})"

            fig = corner.corner(
                post_flat,
                labels=labels,
                color="C1",
                show_titles=True,
                title_fmt=".2f",
                hist_kwargs={"density": True},
            )
            fig.tight_layout()
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
            title = f"flow posterior, injection {j}\n{snr_str}"
            fig.suptitle(title, y=1.0)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            print(f"[{j+1}/{N_EVAL}] {snr_str}", flush=True)

    print(f"saved {CORNER_PATH}")
