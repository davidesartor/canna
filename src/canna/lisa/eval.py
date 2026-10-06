from jaxtyping import Array
from pathlib import Path
import itertools

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import equinox as eqx

from .problem import LisaGB
from .network import LisaFlow
from .train import load_trained, parse_args, path_speed, peak_memory_report

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


def match_sources(problem: LisaGB, draws: np.ndarray, truth: np.ndarray, f) -> np.ndarray:
    """Relabel each draw's sources onto the true ones, by the cheapest permutation.

    The flow's source slots are interchangeable, so a draw's slot 0 can be any of the
    true sources. Each draw is matched in f0 (over the window span), log amplitude and
    sky direction.
    """
    lo, hi = (float(v) for v in problem.f0_window(f))
    perms = np.array(list(itertools.permutations(range(truth.shape[0]))))

    def features(p):
        lam, beta = p[..., 3], p[..., 4]
        return np.stack(
            [
                p[..., 0] / (hi - lo),
                np.log10(np.abs(p[..., 2]) + 1e-300),
                np.cos(beta) * np.cos(lam),
                np.cos(beta) * np.sin(lam),
                np.sin(beta),
            ],
            axis=-1,
        )

    cost = ((features(draws)[:, perms] - features(truth)[None, None]) ** 2).sum((-1, -2))
    best = perms[np.argmin(cost, axis=1)]
    return np.take_along_axis(draws, best[:, :, None], axis=1)


def axis_ranges(flow: np.ndarray, fisher: np.ndarray, scale: list[str]) -> list[tuple]:
    """Each axis' plot range: the span of both sample sets, padded by 5% in its own scale."""
    ranges = []
    for c in range(flow.shape[1]):
        # a linear-space Fisher can spill below zero on a log axis -- ignore those
        fisher_c = fisher[:, c]
        fisher_c = fisher_c[fisher_c > 0] if scale[c] == "log" else fisher_c
        lo = min([flow[:, c].min()] + ([fisher_c.min()] if fisher_c.size else []))
        hi = max([flow[:, c].max()] + ([fisher_c.max()] if fisher_c.size else []))
        if scale[c] == "log":
            ll, hh = np.log10(lo), np.log10(hi)
            m = 0.05 * (hh - ll) or 0.1
            ranges.append((10 ** (ll - m), 10 ** (hh + m)))
        else:
            m = 0.05 * (hi - lo) or 0.1
            ranges.append((lo - m, hi + m))
    return ranges


def corner_page(
    path: Path,
    flow: np.ndarray,
    fisher: np.ndarray,
    labels: list[str],
    scale: list[str],
    title: str,
    truths: np.ndarray,
    pooled: bool = False,
    bins: int = 20,
) -> None:
    """One corner page: flow draws (orange) over Fisher draws (blue, dashed), truths in black.

    flow and fisher are [n, ndim]; truths is [k, ndim]. Each truth row draws lines across
    the page; that is what marks a full page, whose rows are the label permutations of the
    truth. A pooled page stacks every source into the same ndim parameters, so a row is one
    source: it gets a line on the diagonal and a point in the 2D panels, since lines there
    would cross at k^2 places of which only k are real.
    """
    import corner
    import matplotlib.pyplot as plt

    ranges = axis_ranges(flow, fisher, scale)
    # a log axis over less than a factor of 2 (f0 inside one window, ~1%) only crowds its
    # tick labels together; draw it linear. Amplitudes span decades and stay log
    scale = ["linear" if sc == "log" and hi < 2 * lo else sc for sc, (lo, hi) in zip(scale, ranges)]
    style = dict(range=ranges, axes_scale=scale, bins=bins, hist_kwargs={"density": True})
    fig = corner.corner(
        flow, labels=labels, color="C1", show_titles=not pooled, title_fmt=".2g", **style
    )
    corner.corner(
        fisher,
        fig=fig,
        color="C0",
        plot_datapoints=False,
        contour_kwargs={"linestyles": "dashed"},
        **style,
    )
    if pooled:
        ndim = flow.shape[1]
        axes = np.array(fig.axes).reshape(ndim, ndim)
        for i in range(ndim):
            for t in truths:
                axes[i, i].axvline(t[i], color="black", lw=0.8)
        corner.overplot_points(fig, truths, marker="s", ms=3, color="black")
    else:
        for t in truths:
            corner.overplot_lines(fig, t, color="black")
    fig.suptitle(title, y=1.0)
    for ax in fig.axes:
        ax.set_rasterized(True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    # headless, and only here: scorecard imports sample_posterior from this module, and
    # selecting the backend at import would switch any notebook that does the same to Agg
    # (see train.py). corner_page imports pyplot itself, after this
    import matplotlib

    matplotlib.use("Agg")

    args = parse_args()

    # rebuild the state skeleton, then overwrite its params from the checkpoint
    state, _, out_dir = load_trained(args)
    problem, flow = state.problem, state.flow
    corner_dir = out_dir / "corner"
    corner_dir.mkdir(parents=True, exist_ok=True)

    n_sources, n_params = problem.n_sources, len(PARAM_LABELS)
    # a full page has every source's parameters side by side; a pooled or per-source page
    # has the n_params parameters once
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
    source_labels = [f"${label}$" for label in PARAM_LABELS]
    source_scale = scale[:n_params]

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

    # every eval source, for the per-GB pages once all injections are done
    sources = []

    for j, key_n in enumerate(jr.split(key_noise, N_QUANTILES)):
        latent, f = latents[j], windows[j]
        truth = np.asarray(latent).reshape(n_sources, n_params)
        tag = f"q{quantiles[j]:.2f}"

        # inject, sample the flow, and map back to physical units: [draw, source, param]
        o = problem.sample_observation(key_n, latent, f)
        y = problem.preprocess(o, f)
        u0 = jax.vmap(problem.sample_flow, in_axes=(0, None))(
            jr.split(key_n, N_POSTERIOR), f
        )
        post = sample_posterior(
            problem, flow, u0, y, f, args.ode_steps or ODE_STEPS, args.time_power
        )
        draws = np.asarray(
            jax.vmap(problem.flow_to_physical, in_axes=(0, None))(post, f)
        )

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
        fisher = np.asarray(
            fisher_draws(jr.fold_in(key_n, 1), p0, prec_p, prior_prec_p, N_POSTERIOR)
        ).reshape(N_POSTERIOR, n_sources, n_params)

        # full page, one per injection, named by its SNR quantile: every parameter of
        # every source. The Fisher is centred on one labelling and the posterior on all
        # of them, so the Fisher draws and the truth lines come in every permutation
        perms = list(itertools.permutations(range(n_sources)))
        corner_page(
            corner_dir / f"{tag}.pdf",
            draws.reshape(N_POSTERIOR, -1),
            np.concatenate([fisher[:, list(p)] for p in perms]).reshape(-1, len(labels)),
            labels,
            scale,
            f"flow (orange) vs Fisher (blue), SNR quantile {quantiles[j]:.1f},"
            f" SNR={snrs[j]:.1f}",
            np.stack([truth[list(p)].reshape(-1) for p in perms]),
        )

        # pooled page: [draw, source, param] -> [draw * source, param], so each marginal
        # holds every source at once (f0 shows one spike per GB) and labels drop out
        corner_page(
            corner_dir / f"{tag}-pooled.pdf",
            draws.reshape(-1, n_params),
            fisher.reshape(-1, n_params),
            source_labels,
            source_scale,
            f"all {n_sources} sources pooled, flow (orange) vs Fisher (blue),"
            f" SNR quantile {quantiles[j]:.1f}, SNR={snrs[j]:.1f}",
            truth,
            pooled=True,
            bins=50,
        )
        print(f"[{j + 1}/{N_QUANTILES}] {tag} -> {corner_dir / tag}(-pooled).pdf", flush=True)

        # keep each source, its draws relabelled onto it, for the per-GB pages
        matched = match_sources(problem, draws, truth, f)
        source_snr = np.asarray(problem.snr(jnp.asarray(truth)[:, None, :], f))
        for s in range(n_sources):
            sources.append((source_snr[s], tag, s, matched[:, s], fisher[:, s], truth[s]))

    # per-GB pages: the loudest, the median and the faintest single source over all the
    # eval injections, each against its own Fisher forecast
    sources.sort(key=lambda rec: rec[0])
    picks = {"loudest": sources[-1], "median": sources[len(sources) // 2], "faintest": sources[0]}
    for name, (snr, tag, s, flow_s, fisher_s, truth_s) in picks.items():
        corner_page(
            corner_dir / f"gb-{name}.pdf",
            flow_s,
            fisher_s,
            source_labels,
            source_scale,
            f"{name} of the {len(sources)} eval sources: SNR {snr:.1f}"
            f" (injection {tag}, source {s}), flow (orange) vs Fisher (blue)",
            truth_s[None],
        )

    print(
        f"saved {N_QUANTILES} full and {N_QUANTILES} pooled pages and the"
        f" loudest/median/faintest GB pages to {corner_dir}",
        flush=True,
    )
    if peak_memory_report():
        print(peak_memory_report(), flush=True)
