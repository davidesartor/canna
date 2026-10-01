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

from .problem import LisaGB
from .network import LisaFlow
from .train import TrainState, parse_args, path_speed

# the velocity field is stiff near t=1, and too few steps smear the posterior out: on
# the trained XS model the loudest source's f0 width is 0.76/0.42/0.27/0.22/0.21 bins
# at 4/8/16/32/64 steps, so by 32 what is left is the network's own resolution
ODE_STEPS = 32
N_POSTERIOR = 1024
N_CANDIDATES = 1024
N_QUANTILES = 10
N_FISHER_DRAWS = 32

PARAM_LABELS = [
    "f_0",
    "\\mathcal{M}",
    "A",
    "\\lambda",
    "\\beta",
    "\\psi",
    "\\iota",
    "\\phi_0",
]
PARAM_IS_LOG = [True, False, True, False, False, False, False, False]


@eqx.filter_jit
def sample_posterior(
    problem: LisaGB,
    flow: LisaFlow,
    u: Array,
    y: Array,
    f: Array,
    ode_steps: int = ODE_STEPS,
    time_power: float = 1.0,
) -> Array:
    """RK4 transport of prior draws u along the learned velocity field, on the manifold.

    The steps run in a fori_loop, not a python loop: unrolled, every step inlines four
    copies of the network into the graph, and compile time grows with ode_steps.

    The network gives the velocity along the path, d/ds; on the warped clock the ODE is
    in t, so each evaluation is scaled by ds/dt. Steps uniform in t then crowd towards
    the end of the path: with time_power 3 the last of 32 ends at 1 - s = 3e-5.
    """

    @eqx.filter_vmap(in_axes=(None, 0))
    def push(flow: LisaFlow, u: Array) -> Array:
        dt = jnp.asarray(1.0 / ode_steps, u.dtype)

        def velocity(u: Array, t: Array) -> Array:
            v = flow(u, t, y, f)[0]
            return v if time_power == 1 else path_speed(t, time_power) * v

        def step(i: Array, u: Array) -> Array:
            t = i * dt
            k1 = velocity(u, t)
            k2 = velocity(u + k1 * dt / 2, t + dt / 2)
            k3 = velocity(u + k2 * dt / 2, t + dt / 2)
            k4 = velocity(u + k3 * dt, t + dt)
            return problem.exp_map(u, (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6)

        return jax.lax.fori_loop(0, ode_steps, step, u)

    return push(flow, u)


def scaled_inverse(m: Array) -> Array:
    """Inverse of a symmetric matrix over parameters that live on very different scales.

    In physical units the diagonal of a precision or covariance over the source
    parameters spans ~45 orders of magnitude (amplitude ~1e-21 against O(1) angles),
    far past what float64 can invert directly. Inverting in the unit-diagonal frame and
    scaling back leaves only the conditioning the physics actually sets.
    """
    d = jnp.sqrt(jnp.abs(jnp.diag(m)))
    return jnp.linalg.inv(m / jnp.outer(d, d)) / jnp.outer(d, d)


def fisher_draws(
    key: Array, mean: Array, precision: Array, prior_precision: Array, n: int
) -> Array:
    """n draws from a Gaussian with the given precision, sampled in the unit-diagonal frame.

    The precision is diagonalised after Jacobi scaling, so the ill-conditioned covariance
    is never formed. Along any direction the true precision is at least the prior's --
    the Fisher information it adds is positive semi-definite -- but the Hessian averaged
    over a finite set of noise realisations can undershoot that, or go negative, along
    directions the data leave unconstrained (amplitude against inclination near face-on).
    Such a direction is held at the prior's precision.
    """
    precision = (precision + precision.T) / 2
    d = jnp.sqrt(jnp.abs(jnp.diag(precision)))
    w, v = jnp.linalg.eigh(precision / jnp.outer(d, d))
    prior_w = jnp.einsum("ik,ij,jk->k", v, prior_precision / jnp.outer(d, d), v)
    w = jnp.maximum(w, prior_w)
    z = jr.normal(key, (n, mean.size), mean.dtype)
    return mean + (z / jnp.sqrt(w)) @ v.T / d


if __name__ == "__main__":
    args = parse_args()

    out_dir: Path = args.output_dir / f"lisa-{args.config}"
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

    key_pick, key_window, key_noise = jr.split(jr.key(args.seed), 3)

    # each injection gets its own measurement setup, and keeps it all the way through
    windows = jax.vmap(problem.sample_f)(jr.split(key_window, N_CANDIDATES))
    latents = jax.vmap(problem.sample_physical)(
        jr.split(key_pick, N_CANDIDATES), windows
    )

    # Gaussian-approx prior precision, in physical units
    prior_prec_p = scaled_inverse(jnp.cov(latents.reshape(N_CANDIDATES, -1).T))

    # spread the injections over the SNR distribution
    snrs = np.asarray(jax.lax.map(lambda pc: problem.snr(*pc), (latents, windows)))
    quantiles = np.linspace(1.0 / N_QUANTILES, 1.0, N_QUANTILES)
    chosen = np.argsort(snrs)[np.round(quantiles * (N_CANDIDATES - 1)).astype(int)]
    # keep the latents on device: clean_signal writes the chirp in with .at[].set(),
    # which a numpy array does not have, and both the injection and the Fisher
    # replicas below go through it. jax indexes fine with numpy integer arrays
    latents, snrs, windows = (
        latents[chosen],
        snrs[chosen],
        windows[chosen],
    )

    for j, key_n in enumerate(jr.split(key_noise, N_QUANTILES)):
        latent, f = latents[j], windows[j]
        truth = np.asarray(latent).reshape(n_sources, len(PARAM_LABELS))

        # inject, sample the flow, and map back to physical units
        o = problem.sample_observation(key_n, latent, f)
        y = problem.preprocess(o, f)
        u0 = jax.vmap(problem.sample_flow, in_axes=(0, None))(
            jr.split(key_n, N_POSTERIOR), f
        )
        post = sample_posterior(
            problem, flow, u0, y, f, time_power=args.time_power
        )
        samples = np.asarray(
            jax.vmap(problem.flow_to_physical, in_axes=(0, None))(post, f)
        ).reshape(N_POSTERIOR, -1)

        # Fisher forecast at the injection, straight in physical parameters: the nll
        # Hessian averaged over noise realizations (the data-dependent part cancels, the
        # Hessian at a single draw is indefinite), plus the prior precision
        p0 = latent.reshape(-1)
        nll = lambda p, o: -problem.log_likelihood(p.reshape(latent.shape), o, f)
        replicas = jax.vmap(problem.sample_observation, in_axes=(0, None, None))(
            jr.split(jr.fold_in(key_n, 2), N_FISHER_DRAWS), latent, f
        )
        prec_p = (
            jax.lax.map(lambda o: jax.hessian(nll)(p0, o), replicas).mean(0)
            + prior_prec_p
        )
        fisher_samples = np.asarray(
            fisher_draws(jr.fold_in(key_n, 1), p0, prec_p, prior_prec_p, N_POSTERIOR)
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
