"""Evaluate a checkpoint: flow posterior corner plots across SNR quantiles.

Config is shared with scripts/train.py (TrainConfig / env vars) so the model dims
and tag always match the checkpoint. Two extra flags:
  PLOT_PHYSICAL=1  corner in physical space (log10 f0/fdot/A, angles raw) instead of u-space
  FAST_PARAMS=1    restrict the corner to f0/fdot/A/psi (default: all inferred dims)
"""

import gc
import itertools
import os
import sys
import time
from types import SimpleNamespace

import corner
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
from flax import serialization
from matplotlib.backends.backend_pdf import PdfPages

jax.config.update("jax_enable_x64", True)

from canna import networks
from canna import lisa  # SIMPLIFIED_PROBLEM (env) selects the inferred params
from canna import flow_utils as flow_utils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import (  # noqa: E402  reuse the training globals so model dims / tag match
    CKPT_DIR,
    HIDDEN_DIM,
    NUM_BLOCKS,
    NUM_HEADS,
    OUTPUT_DIR,
    SEED,
    env,
    make_tag,
)

# eval-only knobs
PLOT_PHYSICAL = os.environ.get("PLOT_PHYSICAL", "0") not in ("0", "", "false", "False")
FAST_PARAMS = os.environ.get("FAST_PARAMS", "0") not in ("0", "", "false", "False")
ODE_STEPS = int(os.environ.get("ODE_STEPS", "4"))
N_CANDIDATES = int(os.environ.get("N_CANDIDATES", "1024"))  # batch to rank by SNR
SNR_CHUNK_SIZE = int(os.environ.get("SNR_CHUNK_SIZE", "32"))  # SNR draws per chunk
N_QUANTILES = int(os.environ.get("N_QUANTILES", "10"))  # SNR quantiles to plot
N_POSTERIOR = int(os.environ.get("N_POSTERIOR", "1024"))  # flow posterior draws
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "64"))  # flow draws per chunk

# canonical 8-parameter metadata (physical display uses log10 on the log-uniform priors)
PARAM_TEX = [r"f_0", r"\dot f", r"A", r"\alpha", r"\delta", r"\psi", r"\iota", r"\phi_0"]
PHYS_LABELS = [r"\log_{10} f_0", r"\log_{10} \dot f", r"\log_{10} A", *PARAM_TEX[3:]]
PARAM_IS_LOG = np.array([True, True, True, False, False, False, False, False])
PHYS_RANGE = [
    (np.log10(1e-4), np.log10(lisa.MAX_FREQUENCY)),
    (np.log10(1e-22), np.log10(1e-18)),
    (np.log10(1e-24), np.log10(1e-22)),
    (0.0, 2.0 * np.pi),
    (-np.pi / 2.0, np.pi / 2.0),
    (0.0, np.pi),
    (0.0, np.pi),
    (-np.pi, np.pi),
]
FAST_IDX = [0, 1, 2, 5]  # f0, fdot, A, psi


def to_display(phys):
    """(..., 8) physical params -> display: log10 the log-uniform dims, leave angles raw."""
    disp = np.asarray(phys, dtype=float).copy()
    disp[..., PARAM_IS_LOG] = np.log10(disp[..., PARAM_IS_LOG])
    return disp


if __name__ == "__main__":
    cfg = SimpleNamespace(
        seed=SEED,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        n_sources=lisa.N_SOURCES,
        output_dir=env("output_dir", OUTPUT_DIR),
        ckpt_dir=env("ckpt_dir", CKPT_DIR),
    )
    tag = make_tag()
    N_SOURCES = cfg.n_sources

    checkpoint_path = os.path.join(cfg.ckpt_dir, f"checkpoint_flow_{tag}.msgpack")
    os.makedirs(cfg.output_dir, exist_ok=True)
    space = "physical" if PLOT_PHYSICAL else "u"
    corner_name = f"flow_corner_{tag}_{space}" + ("_fast" if FAST_PARAMS else "") + ".pdf"
    corner_path = os.path.join(cfg.output_dir, corner_name)
    cache_path = os.path.join(
        cfg.output_dir, f"eval_snr_{N_SOURCES}src_seed{cfg.seed}_n{N_CANDIDATES}_q{N_QUANTILES}.npz"
    )

    # inferred canonical indices, and the subset we actually plot
    inferred_idx = np.array(lisa.MASK) if lisa.SIMPLIFIED_PROBLEM else np.arange(8)
    plot_idx = [i for i in inferred_idx if (not FAST_PARAMS or i in FAST_IDX)]
    post_cols = [int(np.where(inferred_idx == i)[0][0]) for i in plot_idx]  # column of i in post

    print(f"JAX backend: {jax.default_backend()}, devices: {jax.local_device_count()}")
    print(f"tag={tag}  inferring {lisa.PARAMETER_NAMES}  space={space}  fast={FAST_PARAMS}")
    key = jr.key(cfg.seed)

    # ---- model: same dims as training, weights from the checkpoint ----
    key, key_mock = jr.split(key)
    u_s, _, _, _, y_s, _ = lisa.get_physics_sample(key_mock, n_sources=N_SOURCES)
    x_dim = u_s.shape[-1]

    # checkpoint was trained in fp32 (data generation stays fp64)
    key, key_init = jr.split(key)
    flow = networks.MMDiT(
        x_dim=x_dim,
        y_channels=y_s.shape[-1],
        hidden_dim=cfg.hidden_dim,
        num_blocks=cfg.num_blocks,
        num_heads=cfg.num_heads,
        rngs=nnx.Rngs(key_init),
        dtype=jnp.float32,
    )
    with open(checkpoint_path, "rb") as f:
        restored = serialization.from_bytes(nnx.to_pure_dict(nnx.state(flow)), f.read())
    state = nnx.state(flow)
    nnx.replace_by_pure_dict(state, restored)
    nnx.update(flow, state)
    print(f"loaded {checkpoint_path}")

    def sample_posterior_chunk(flow, x0_chunk, y):
        # eager wrapper: casts + the already-jitted sample_posterior (no nested nnx.jit,
        # which would pass the flow graph node across trace levels -> TraceContextError)
        x0_chunk, y = x0_chunk.astype("float32"), y.astype("float32")
        post = flow_utils.sample_posterior(flow, x0_chunk, y, ODE_STEPS, lisa.exponential_map)
        return post.astype("float64")  # back to fp64 for the physical param mapping

    @jax.jit
    def snr_chunk(params_chunk):
        return jax.vmap(lisa.optimal_snr)(params_chunk)

    # ---- 1. choose one injection per SNR quantile (cached; ranking is the slow part) ----
    quantiles = np.linspace(1.0 / N_QUANTILES, 1.0, N_QUANTILES)
    if os.path.exists(cache_path):
        cache = np.load(cache_path)
        u_chosen, params_chosen, snr_chosen = cache["u"], cache["params"], cache["snr"]
        print(f"loaded cached injections {cache_path}")
    else:
        key, key_cand = jr.split(key)
        u_cand = np.array(jr.uniform(key_cand, shape=(N_CANDIDATES, N_SOURCES, 8)))
        params_cand = np.array(lisa.prior_inverse_cdf(u_cand))  # (N_CANDIDATES, N_SOURCES, 8)

        t0 = time.time()
        snrs = np.empty(N_CANDIDATES)
        for i in range(0, N_CANDIDATES, SNR_CHUNK_SIZE):
            snrs[i : i + SNR_CHUNK_SIZE] = np.array(snr_chunk(params_cand[i : i + SNR_CHUNK_SIZE]))
        print(f"ranked {N_CANDIDATES} candidates by SNR in {time.time() - t0:.1f}s")

        order = np.argsort(snrs)
        chosen = order[np.round(quantiles * (N_CANDIDATES - 1)).astype(int)]
        u_chosen, params_chosen, snr_chosen = u_cand[chosen], params_cand[chosen], snrs[chosen]
        np.savez(cache_path, u=u_chosen, params=params_chosen, snr=snr_chosen, quantiles=quantiles)
        print(f"cached injections to {cache_path}")

    # ---- 2. per quantile: build injection, draw flow posterior, corner-plot ----
    if PLOT_PHYSICAL:
        base_labels = [PHYS_LABELS[i] for i in plot_idx]
        base_range = [PHYS_RANGE[i] for i in plot_idx]
    else:
        base_labels = [f"u_{{{PARAM_TEX[i]}}}" for i in plot_idx]
        base_range = [(0.0, 1.0)] * len(plot_idx)
    display_labels = [
        f"${lab}$" + (f" (src {s})" if N_SOURCES > 1 else "")
        for s in range(N_SOURCES)
        for lab in base_labels
    ]
    display_range = base_range * N_SOURCES

    def post_to_plot(post):
        """(N, S, x_dim) posterior draws -> (N, S, P) plotted coordinates."""
        if not PLOT_PHYSICAL:
            return np.asarray(post)[..., post_cols]
        u8 = jnp.full(post.shape[:-1] + (8,), 0.5).at[..., inferred_idx].set(post)
        return to_display(lisa.prior_inverse_cdf(u8))[..., plot_idx]

    def truth_to_plot(u_true, phys_true):
        """(S, 8) truth -> (S, P) plotted coordinates."""
        if not PLOT_PHYSICAL:
            return np.asarray(u_true)[..., plot_idx]
        return to_display(phys_true)[..., plot_idx]

    key, key_noise = jr.split(key)
    noise_keys = jr.split(key_noise, N_QUANTILES)

    with PdfPages(corner_path) as pdf:
        for j, (q, key_n) in enumerate(zip(quantiles, noise_keys)):
            p_true = params_chosen[j]  # (N_SOURCES, 8) physical truth
            snr = float(snr_chosen[j])

            signal = lisa.clean_signal(p_true)
            datastream = signal + lisa.sample_noise(key_n)
            y0 = lisa.preprocess_datastream(datastream)

            # flow posterior: chunked, flow held fixed
            n_chunks = -(-N_POSTERIOR // EVAL_BATCH_SIZE)  # ceil division
            post_chunks, n_remaining = [], N_POSTERIOR
            for key_chunk in jr.split(key_n, n_chunks):
                n = min(EVAL_BATCH_SIZE, n_remaining)
                x0_chunk = jr.uniform(key_chunk, shape=(n, N_SOURCES, x_dim))
                post_chunks.append(sample_posterior_chunk(flow, x0_chunk, y0))
                n_remaining -= n
            post = jnp.concatenate(post_chunks, axis=0)  # (N_POSTERIOR, N_SOURCES, x_dim)

            samples = post_to_plot(post).reshape(N_POSTERIOR, -1)  # (N_POSTERIOR, S*P)
            truth = truth_to_plot(u_chosen[j], p_true)  # (S, P)

            fig = corner.corner(
                samples,
                labels=display_labels,
                range=display_range,
                color="C1",
                show_titles=True,
                title_fmt=".2f",
                hist_kwargs={"density": True},
            )
            legend_handles = [mlines.Line2D([], [], color="C1", label="flow posterior")]

            # flow is permutation invariant across sources: every relabelling of truth is a valid mode
            for sigma in itertools.permutations(range(N_SOURCES)):
                truth_perm = truth[list(sigma)].reshape(-1)
                corner.overplot_lines(fig, truth_perm, color="black")
                corner.overplot_points(
                    fig, truth_perm[None, :], marker="s", color="black", markersize=4
                )
            legend_handles.append(mlines.Line2D([], [], color="black", label="truth (all perms)"))

            fig.legend(handles=legend_handles, loc="upper right", fontsize=12, frameon=False)
            fig.suptitle(f"flow posterior ({space}), SNR quantile {q:.1f}\nSNR={snr:.1f}", y=1.0)
            # rasterize panel content: PdfPages buffers every page until close, and vector
            # scatter/overplots (~N! * 136 panels) balloon RSS for many-source corners
            for ax in fig.axes:
                ax.set_rasterized(True)
            pdf.savefig(fig, bbox_inches="tight", dpi=150)
            plt.close(fig)
            # corner figures have circular refs; free them now so RSS doesn't climb
            # each quantile and trip the cgroup OOM killer before Python's gc runs
            del fig, samples, truth, post, post_chunks
            gc.collect()
            print(f"[{j+1}/{N_QUANTILES}] quantile={q:.1f} SNR={snr:.1f}", flush=True)

    print(f"saved {corner_path}")
