from jaxtyping import Array
from pathlib import Path
import itertools

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import orbax.checkpoint as ocp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner
import equinox as eqx

from .problem import NoisySinusoid
from .network import SinusoidFlow
from .train import TrainState, parse_args

ODE_STEPS = 4
N_POSTERIOR = 1024
N_CANDIDATES = 1024
N_QUANTILES = 10
N_FISHER_DRAWS = 32

PARAM_LABELS = ["A", "f", "\\phi_0"]
PARAM_IS_LOG = [True, True, False]


@eqx.filter_jit
def sample_posterior(
    problem: NoisySinusoid,
    flow: SinusoidFlow,
    u: Array,
    y: Array,
    ode_steps: int = ODE_STEPS,
) -> Array:
    """RK4 transport of prior draws u along the learned velocity field, on the manifold."""

    @eqx.filter_vmap(in_axes=(None, 0))
    def push(flow: SinusoidFlow, u: Array) -> Array:
        dt = jnp.asarray(1.0 / ode_steps, u.dtype)
        for i in range(ode_steps):
            t = i * dt
            k1 = flow(u, t, y)[0]
            k2 = flow(u + k1 * dt / 2, t + dt / 2, y)[0]
            k3 = flow(u + k2 * dt / 2, t + dt / 2, y)[0]
            k4 = flow(u + k3 * dt, t + dt, y)[0]
            u = problem.exp_map(u, (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6)
        return u

    return push(flow, u)


if __name__ == "__main__":
    args = parse_args()

    out_dir: Path = args.output_dir / f"sinusoid-{args.config}"
    corner_dir = out_dir / "corner"
    corner_dir.mkdir(parents=True, exist_ok=True)

    # rebuild the state skeleton, then overwrite its params from the checkpoint
    state = TrainState.from_config(args)
    checkpoints = ocp.CheckpointManager(
        (out_dir / "checkpoints").absolute(),
        options=ocp.CheckpointManagerOptions(max_to_keep=1),
    )
    state, *_ = state.restore_from(checkpoints)
    problem, flow = state.problem, state.flow

    n_sources = problem.n_sources
    labels = [
        f"${label}$" + (f" (s{s})" if n_sources > 1 else "")
        for s in range(n_sources)
        for label in PARAM_LABELS
    ]
    scale = [
        "log" if is_log else "linear"
        for _ in range(n_sources)
        for is_log in PARAM_IS_LOG
    ]

    key_pick, key_noise = jr.split(jr.key(args.seed))
    latents = jax.vmap(problem.sample_physical)(jr.split(key_pick, N_CANDIDATES))

    # Gaussian-approx prior precision, in physical units
    prior_prec_p = jnp.linalg.inv(jnp.cov(latents.reshape(N_CANDIDATES, -1).T))

    # spread the injections over the SNR distribution
    snrs = np.asarray(jax.lax.map(problem.snr, latents))
    quantiles = np.linspace(1.0 / N_QUANTILES, 1.0, N_QUANTILES)
    chosen = np.argsort(snrs)[np.round(quantiles * (N_CANDIDATES - 1)).astype(int)]
    latents, snrs = np.asarray(latents)[chosen], snrs[chosen]

    for j, key_n in enumerate(jr.split(key_noise, N_QUANTILES)):
        latent = latents[j]
        truth = latent.reshape(n_sources, len(PARAM_LABELS))

        # inject, sample the flow, and map back to physical units
        o = problem.sample_observation(key_n, latent)
        y = problem.preprocess(o)
        u0 = jax.vmap(problem.sample_flow)(jr.split(key_n, N_POSTERIOR))
        post = sample_posterior(problem, flow, u0, y)
        samples = np.asarray(jax.vmap(problem.flow_to_physical)(post)).reshape(
            N_POSTERIOR, -1
        )

        # Fisher forecast at the injection, straight in physical parameters: the nll
        # Hessian averaged over noise realizations (the data-dependent part cancels, the
        # Hessian at a single draw is indefinite), plus the prior precision
        p0 = latent.reshape(-1)
        nll = lambda p, o: -problem.log_likelihood(p.reshape(latent.shape), o)
        replicas = jax.vmap(problem.sample_observation, in_axes=(0, None))(
            jr.split(jr.fold_in(key_n, 2), N_FISHER_DRAWS), latent
        )
        prec_p = (
            jax.lax.map(lambda o: jax.hessian(nll)(p0, o), replicas).mean(0)
            + prior_prec_p
        )
        fisher_samples = np.asarray(
            jr.multivariate_normal(
                jr.fold_in(key_n, 1), p0, jnp.linalg.inv(prec_p), (N_POSTERIOR,)
            )
        )

        # the Fisher is centred on one labelling, the posterior on all of them
        fisher_samples = np.concatenate(
            [
                fisher_samples.reshape(N_POSTERIOR, n_sources, -1)[
                    :, list(sigma)
                ].reshape(N_POSTERIOR, -1)
                for sigma in itertools.permutations(range(n_sources))
            ]
        )

        # pad each axis by 5% of its span, measured in that axis' own scale, over both
        ranges = []
        for c in range(samples.shape[1]):
            # a linear-space Fisher can spill below zero on a log axis -- ignore those
            fisher_c = fisher_samples[:, c]
            fisher_c = fisher_c[fisher_c > 0] if scale[c] == "log" else fisher_c
            lo = min(
                [samples[:, c].min()] + ([fisher_c.min()] if fisher_c.size else [])
            )
            hi = max(
                [samples[:, c].max()] + ([fisher_c.max()] if fisher_c.size else [])
            )
            if scale[c] == "log":
                ll, hh = np.log10(lo), np.log10(hi)
                m = 0.05 * (hh - ll) or 0.1
                ranges.append((10 ** (ll - m), 10 ** (hh + m)))
            else:
                m = 0.05 * (hi - lo) or 0.1
                ranges.append((lo - m, hi + m))

        # one corner file per injection, named by its SNR quantile
        fig = corner.corner(
            samples,
            labels=labels,
            range=ranges,
            axes_scale=scale,
            color="C1",
            show_titles=True,
            title_fmt=".2g",
            hist_kwargs={"density": True},
        )
        corner.corner(
            fisher_samples,
            fig=fig,
            range=ranges,
            axes_scale=scale,
            color="C0",
            hist_kwargs={"density": True},
            plot_datapoints=False,
            contour_kwargs={"linestyles": "dashed"},
        )

        # source labelling is arbitrary, so mark every permutation of the truth
        for sigma in itertools.permutations(range(n_sources)):
            corner.overplot_lines(fig, truth[list(sigma)].reshape(-1), color="black")

        tag = f"q{quantiles[j]:.2f}"
        fig.suptitle(
            f"flow (orange) vs Fisher (blue), SNR quantile {quantiles[j]:.1f},"
            f" SNR={snrs[j]:.1f}",
            y=1.0,
        )
        for ax in fig.axes:
            ax.set_rasterized(True)
        page_path = corner_dir / f"{tag}.pdf"
        fig.savefig(page_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"[{j + 1}/{N_QUANTILES}] {tag} -> {page_path}", flush=True)

    print(f"saved {N_QUANTILES} plots to {corner_dir}", flush=True)
