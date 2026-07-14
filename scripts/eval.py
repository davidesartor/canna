"""Evaluate a checkpoint: flow posterior corner plots across SNR quantiles.

Model dims / run tag are imported from train.py so they always match the checkpoint;
the import-time toggles in canna.lisa (SIMPLIFIED_PROBLEM, N_SOURCES) must match the
training run too. Flip PLOT_PHYSICAL / FAST_PARAMS below to change the corner.
"""

from typing import NamedTuple
from jaxtyping import Array, Float
import gc
import itertools
import os
import time

import corner
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx, serialization
from matplotlib.backends.backend_pdf import PdfPages

jax.config.update("jax_enable_x64", True)

from canna import lisa, networks
from train import (
    CHECKPOINT_DIR,
    HIDDEN_DIM,
    NUM_BLOCKS,
    NUM_HEADS,
    OUTPUT_DIR,
    SEED,
)

NETWORK_DTYPE = jnp.float32  # integrate the flow in fp32; checkpoints store fp32 params

# --- eval knobs ---
PLOT_PHYSICAL = os.environ.get("PLOT_PHYSICAL", "1") not in ("0", "", "false", "False")
# ^ physical space (log-scaled f0/fdot/A axes) vs u-space; env-overridable so both can run
FAST_PARAMS = False  # restrict the corner to f0/fdot/A/psi (default: all inferred dims)
ODE_STEPS = 4
N_CANDIDATES = 1024  # SNR-ranked pool to pick injections from
N_QUANTILES = 10  # SNR quantiles to plot
N_POSTERIOR = 1024  # flow posterior draws per injection

# canonical 8-parameter metadata; the log-uniform dims are plotted on a log-scaled axis
PARAM_TEX = [
    r"f_0",
    r"\dot f",
    r"A",
    r"\alpha",
    r"\delta",
    r"\psi",
    r"\iota",
    r"\phi_0",
]
PARAM_IS_LOG = np.array([True, True, True, False, False, False, False, False])
FAST_IDX = [0, 1, 2, 5]  # f0, fdot, A, psi


def setup(x_dim: int, y_channels: int, ckpt_path: str) -> networks.MMDiT:
    """Build the flow network (train dims) and restore its weights from the checkpoint."""
    flow = networks.MMDiT(
        x_dim=x_dim,
        y_channels=y_channels,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dtype=NETWORK_DTYPE,
        param_dtype=NETWORK_DTYPE,
        rngs=nnx.Rngs(SEED),
    )
    with open(ckpt_path, "rb") as f:
        restored = serialization.from_bytes(nnx.to_pure_dict(nnx.state(flow)), f.read())
    state = nnx.state(flow)
    nnx.replace_by_pure_dict(state, restored)
    nnx.update(flow, state)
    return flow


@nnx.jit(static_argnames=("ode_steps",))
def sample_posterior(
    flow: networks.MMDiT,
    u: Float[Array, "N S P"],
    y: Float[Array, "T F C"],
    ode_steps: int = ODE_STEPS,
) -> Float[Array, "N S P"]:
    """RK4-integrate the flow velocity t: 0->1 to draw posterior samples for one observation y.

    vmap over the draw axis: each call is unbatched (x: "S P", y: "T F C", scalar t).
    """

    @nnx.vmap(in_axes=(None, 0))
    def push(flow, u: Float[Array, "S P"]) -> Float[Array, "S P"]:
        dt = jnp.asarray(1.0 / ode_steps, u.dtype)
        for i in range(ode_steps):
            t = i * dt
            k1, _, _ = flow(u, y, t)
            k2, _, _ = flow(u + k1 * dt / 2, y, t + dt / 2)
            k3, _, _ = flow(u + k2 * dt / 2, y, t + dt / 2)
            k4, _, _ = flow(u + k3 * dt, y, t + dt)
            u = lisa.exponential_map(u, (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6)
        return u

    return push(flow, u)


class Injections(NamedTuple):
    u: Float[Array, "Q S 8"]  # full 8-dim u (cache stores every canonical dim)
    params: Float[Array, "Q S 8"]
    snr: Float[Array, "Q"]
    quantiles: Float[Array, "Q"]


def pick_injections(key, S: int, cache_path: str) -> Injections:
    """One injection per SNR quantile from an SNR-ranked candidate pool (cached; ranking is slow)."""
    quantiles = np.linspace(1.0 / N_QUANTILES, 1.0, N_QUANTILES)
    if os.path.exists(cache_path):
        cache = np.load(cache_path)
        print(f"loaded cached injections {cache_path}")
        return Injections(cache["u"], cache["params"], cache["snr"], quantiles)

    u_cand = np.array(jr.uniform(key, (N_CANDIDATES, S, 8)))
    params_cand = np.array(lisa.prior_inverse_cdf(u_cand))

    t0 = time.time()
    snrs = np.array(
        jax.lax.map(lisa.optimal_snr, params_cand)
    )  # one waveform at a time
    print(f"ranked {N_CANDIDATES} candidates by SNR in {time.time() - t0:.1f}s")

    chosen = np.argsort(snrs)[np.round(quantiles * (N_CANDIDATES - 1)).astype(int)]
    inj = Injections(u_cand[chosen], params_cand[chosen], snrs[chosen], quantiles)
    np.savez(cache_path, u=inj.u, params=inj.params, snr=inj.snr, quantiles=quantiles)
    print(f"cached injections to {cache_path}")
    return inj


class PlotLayout(NamedTuple):
    labels: list[str]  # per (source, param) corner-axis labels
    scale: list[str]  # matching "log"/"linear" axis scales
    inferred_idx: np.ndarray  # canonical-8 indices the model infers
    plot_idx: list[int]  # canonical-8 indices actually plotted
    post_cols: list[int]  # their columns within the inferred/posterior vector


def plot_layout(S: int) -> PlotLayout:
    """Corner-axis labels, scales, and index maps for the plotted parameters."""
    inferred_idx = np.array(lisa.MASK) if lisa.SIMPLIFIED_PROBLEM else np.arange(8)
    plot_idx = [i for i in inferred_idx if (not FAST_PARAMS or i in FAST_IDX)]
    post_cols = [int(np.where(inferred_idx == i)[0][0]) for i in plot_idx]

    if PLOT_PHYSICAL:
        base_labels = [PARAM_TEX[i] for i in plot_idx]
        # corner log-scales these axes (logspace bins + set_xscale) so we plot raw physical values
        base_scale = ["log" if PARAM_IS_LOG[i] else "linear" for i in plot_idx]
    else:
        base_labels = [f"u_{{{PARAM_TEX[i]}}}" for i in plot_idx]
        base_scale = ["linear"] * len(plot_idx)
    labels = [
        f"${lab}$" + (f" (src {s})" if S > 1 else "")
        for s in range(S)
        for lab in base_labels
    ]
    return PlotLayout(labels, base_scale * S, inferred_idx, plot_idx, post_cols)


def to_plot_coords(u_or_post, layout: PlotLayout):
    """u-space samples/truth (..., x_dim) -> plotted coordinates (..., P)."""
    if not PLOT_PHYSICAL:
        return np.asarray(u_or_post)[..., layout.post_cols]
    u8 = (
        jnp.full(u_or_post.shape[:-1] + (8,), 0.5)
        .at[..., layout.inferred_idx]
        .set(u_or_post)
    )
    phys = np.asarray(lisa.prior_inverse_cdf(u8), dtype=float)
    return phys[..., layout.plot_idx]  # raw physical; log dims get a log-scaled axis


def data_range(samples, truth, layout: PlotLayout):
    """Per-axis (lo, hi) with a 5% margin, from samples + every truth permutation."""
    P = len(layout.plot_idx)
    lo, hi = samples.min(0), samples.max(0)
    for c in range(samples.shape[1]):
        t_c = truth[:, c % P]  # any source can land in this column under permutation
        lo[c], hi[c] = min(lo[c], t_c.min()), max(hi[c], t_c.max())
    ranges = []
    for l, h, sc in zip(lo, hi, layout.scale):
        if sc == "log":
            ll, hh = np.log10(l), np.log10(h)
            m = 0.05 * (hh - ll) or 0.1
            ranges.append((10 ** (ll - m), 10 ** (hh + m)))
        else:
            m = 0.05 * (h - l) or 0.1
            ranges.append((l - m, h + m))
    return ranges


def plot_quantile(pdf, flow, layout, inj, j, S, key_n, space):
    """Draw the flow posterior for injection j and add its corner page to the PDF."""
    p_true, q, snr = inj.params[j], float(inj.quantiles[j]), float(inj.snr[j])
    snr_per_src = [float(lisa.optimal_snr(p_true[s : s + 1])) for s in range(S)]
    y0 = lisa.preprocess_datastream(
        lisa.clean_signal(p_true) + lisa.sample_noise(key_n)
    )

    # draw + map to plotted coordinates
    x_dim = len(layout.inferred_idx)
    u0 = jr.uniform(key_n, (N_POSTERIOR, S, x_dim), NETWORK_DTYPE)
    post = sample_posterior(flow, u0, y0.astype(NETWORK_DTYPE), ODE_STEPS).astype(
        "float64"
    )
    samples = to_plot_coords(post, layout).reshape(
        N_POSTERIOR, -1
    )  # (N_POSTERIOR, S*P)
    truth = to_plot_coords(inj.u[j][..., layout.inferred_idx], layout)  # (S, P)

    fig = corner.corner(
        samples,
        labels=layout.labels,
        range=data_range(samples, truth, layout),
        axes_scale=layout.scale,
        color="C1",
        show_titles=True,
        title_fmt=".2g",  # .2f collapses the tiny log-uniform dims (A~1e-23) to 0.00
        hist_kwargs={"density": True},
    )
    handles = [mlines.Line2D([], [], color="C1", label="flow posterior")]
    # flow is permutation-invariant across sources: every relabelling of truth is a valid mode
    for sigma in itertools.permutations(range(S)):
        truth_perm = truth[list(sigma)].reshape(-1)
        corner.overplot_lines(fig, truth_perm, color="black")
        corner.overplot_points(
            fig, truth_perm[None, :], marker="s", color="black", markersize=4
        )
    handles.append(mlines.Line2D([], [], color="black", label="truth (all perms)"))

    fig.legend(handles=handles, loc="upper right", fontsize=12, frameon=False)
    per_src = "  ".join(f"src{s}={v:.1f}" for s, v in enumerate(snr_per_src))
    fig.suptitle(
        f"flow posterior ({space}), SNR quantile {q:.1f}\n"
        f"combined SNR={snr:.1f}   per-source: {per_src}",
        y=1.0,
    )
    # rasterize panels: PdfPages buffers every page until close, and vector scatter balloons RSS
    for ax in fig.axes:
        ax.set_rasterized(True)
    pdf.savefig(fig, bbox_inches="tight", dpi=150)

    # corner figures have circular refs; free them now so RSS doesn't trip the OOM killer
    plt.close(fig)
    del fig, samples, truth, post
    gc.collect()
    print(f"[{j + 1}/{N_QUANTILES}] quantile={q:.1f} SNR={snr:.1f}", flush=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    S = lisa.N_SOURCES
    tag = f"{S}src_{'simplified' if lisa.SIMPLIFIED_PROBLEM else 'full'}"
    space = "physical" if PLOT_PHYSICAL else "u"
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_flow_{tag}.msgpack")
    corner_path = os.path.join(
        OUTPUT_DIR,
        f"flow_corner_{tag}_{space}" + ("_fast" if FAST_PARAMS else "") + ".pdf",
    )
    cache_path = os.path.join(
        OUTPUT_DIR, f"eval_snr_{S}src_seed{SEED}_n{N_CANDIDATES}_q{N_QUANTILES}.npz"
    )

    print(f"JAX backend: {jax.default_backend()}, devices: {jax.local_device_count()}")
    print(
        f"tag={tag}  inferring {lisa.PARAMETER_NAMES}  space={space}  fast={FAST_PARAMS}"
    )
    key = jr.key(SEED)

    # --- model: same dims as training, weights from the checkpoint ---
    key, key_mock = jr.split(key)
    u_s, _, _, _, y_s, _ = lisa.get_physics_sample(key_mock, n_sources=S)
    flow = setup(u_s.shape[-1], y_s.shape[-1], ckpt_path)
    print(f"loaded {ckpt_path}")

    # --- pick one injection per SNR quantile ---
    key, key_cand = jr.split(key)
    inj = pick_injections(key_cand, S, cache_path)

    # --- per quantile: draw flow posterior, corner-plot ---
    layout = plot_layout(S)
    key, key_noise = jr.split(key)
    with PdfPages(corner_path) as pdf:
        for j, key_n in enumerate(jr.split(key_noise, N_QUANTILES)):
            plot_quantile(pdf, flow, layout, inj, j, S, key_n, space)

    print(f"saved {corner_path}")


if __name__ == "__main__":
    main()
