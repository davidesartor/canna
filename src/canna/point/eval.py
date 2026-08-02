from jaxtyping import Array
from pathlib import Path

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

from .problem import NoisyPoint
from .network import PointFlow
from .train import TrainState, parse_args

ODE_STEPS = 4
N_POSTERIOR = 1024
N_INJECTIONS = 10
N_FISHER_DRAWS = 32


@eqx.filter_jit
def sample_posterior(
    problem: NoisyPoint,
    flow: PointFlow,
    u: Array,
    y: Array,
    ode_steps: int = ODE_STEPS,
) -> Array:
    """RK4 transport of prior draws u along the learned velocity field, on the manifold."""

    @eqx.filter_vmap(in_axes=(None, 0))
    def push(flow: PointFlow, u: Array) -> Array:
        dt = jnp.asarray(1.0 / ode_steps, u.dtype)
        for i in range(ode_steps):
            t = i * dt
            k1 = flow(u, t, y)
            k2 = flow(u + k1 * dt / 2, t + dt / 2, y)
            k3 = flow(u + k2 * dt / 2, t + dt / 2, y)
            k4 = flow(u + k3 * dt, t + dt, y)
            u = problem.exp_map(u, (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6)
        return u

    return push(flow, u)


if __name__ == "__main__":
    args = parse_args()

    out_dir: Path = args.output_dir / f"point-{args.config}"
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

    labels = [f"$x_{i}$" for i in range(problem.dim)]
    key_pick, key_noise = jr.split(jr.key(args.seed))
    latents = np.asarray(
        jax.vmap(problem.sample_physical)(jr.split(key_pick, N_INJECTIONS))
    )

    # Gaussian-approx prior precision, in physical units
    prior_prec_p = jnp.linalg.inv(problem.cov)

    for j, key_n in enumerate(jr.split(key_noise, N_INJECTIONS)):
        latent = latents[j]

        # inject, sample the flow, and map back to physical units
        o = problem.sample_observation(key_n, latent)
        y = problem.preprocess(o)
        u0 = jax.vmap(problem.sample_point)(jr.split(key_n, N_POSTERIOR))
        post = sample_posterior(problem, flow, u0, y)
        samples = np.asarray(jax.vmap(problem.flow_to_physical)(post))

        # Fisher forecast at the injection, straight in physical parameters: the nll
        # Hessian averaged over noise realizations (the data-dependent part cancels, the
        # Hessian at a single draw is indefinite), plus the prior precision
        nll = lambda p, o: -problem.log_likelihood(p, o)
        replicas = jax.vmap(problem.sample_observation, in_axes=(0, None))(
            jr.split(jr.fold_in(key_n, 2), N_FISHER_DRAWS), latent
        )
        prec_p = (
            jax.lax.map(lambda o: jax.hessian(nll)(latent, o), replicas).mean(0)
            + prior_prec_p
        )
        fisher_samples = np.asarray(
            jr.multivariate_normal(
                jr.fold_in(key_n, 1), latent, jnp.linalg.inv(prec_p), (N_POSTERIOR,)
            )
        )

        # pad each axis by 5% of its span, over both clouds
        ranges = []
        for c in range(samples.shape[1]):
            lo = min(samples[:, c].min(), fisher_samples[:, c].min())
            hi = max(samples[:, c].max(), fisher_samples[:, c].max())
            m = 0.05 * (hi - lo) or 0.1
            ranges.append((lo - m, hi + m))

        fig = corner.corner(
            samples,
            labels=labels,
            range=ranges,
            color="C1",
            show_titles=True,
            title_fmt=".2g",
            hist_kwargs={"density": True},
        )
        corner.corner(
            fisher_samples,
            fig=fig,
            range=ranges,
            color="C0",
            hist_kwargs={"density": True},
            plot_datapoints=False,
            contour_kwargs={"linestyles": "dashed"},
        )
        corner.overplot_lines(fig, latent, color="black")

        fig.suptitle("flow (orange) vs Fisher (blue)", y=1.0)
        for ax in fig.axes:
            ax.set_rasterized(True)
        page_path = corner_dir / f"{j:02d}.pdf"
        fig.savefig(page_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"[{j + 1}/{N_INJECTIONS}] -> {page_path}", flush=True)

    print(f"saved {N_INJECTIONS} plots to {corner_dir}", flush=True)
