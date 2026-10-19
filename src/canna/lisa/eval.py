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
from .train import load_trained, parse_args, path_speed, peak_memory_report, with_residual

# the velocity field is stiff near t=1, and too few steps smear the posterior out: on
# the trained XS model the loudest source's f0 width is 0.76/0.42/0.27/0.22/0.21 bins
# at 4/8/16/32/64 steps, so by 32 what is left is the network's own resolution
ODE_STEPS = 32
N_POSTERIOR = 1024
N_CANDIDATES = 1024
N_QUANTILES = 10
N_FISHER_DRAWS = 32
# draws pushed through the network at once by sample_posterior (B needs it: see there)
CHUNK = 256

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
    chunk: int = CHUNK,
) -> Array:
    """RK4 transport of prior draws u along the learned velocity field, on the manifold.

    The steps run in a fori_loop, not a python loop: unrolled, every step inlines four
    copies of the network into the graph, and compile time grows with ode_steps.

    The network gives the velocity along the path, d/ds; on the warped clock the ODE is
    in t, so each evaluation is scaled by ds/dt. Steps uniform in t then crowd towards
    the end of the path: with time_power 3 the last of 32 ends at 1 - s = 3e-5.

    The draws go through in chunks of `chunk`, one after another: on B (2048 tokens) all
    1024 at once need a single 27 GiB buffer and ran out of memory on an A100. A draw's
    path does not depend on the others, so chunking changes nothing but the memory.
    """

    @eqx.filter_vmap(in_axes=(None, 0))
    def push(flow: LisaFlow, u: Array) -> Array:
        dt = jnp.asarray(1.0 / ode_steps, u.dtype)

        def velocity(u: Array, t: Array) -> Array:
            # a template network conditions on the residual at its current point
            templates = getattr(flow, "templates", False)  # stand-in flows in the tests have none
            v = flow(u, t, with_residual(problem, u, y, f) if templates else y, f)[0]
            return v if time_power == 1 else path_speed(t, time_power) * v

        def step(i: Array, u: Array) -> Array:
            t = i * dt
            k1 = velocity(u, t)
            k2 = velocity(u + k1 * dt / 2, t + dt / 2)
            k3 = velocity(u + k2 * dt / 2, t + dt / 2)
            k4 = velocity(u + k3 * dt, t + dt)
            return problem.exp_map(u, (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6)

        return jax.lax.fori_loop(0, ode_steps, step, u)

    n = u.shape[0]
    if n <= chunk:
        return push(flow, u)
    assert n % chunk == 0, f"{n} draws do not split into chunks of {chunk}"
    chunks = jax.lax.map(lambda c: push(flow, c), u.reshape(n // chunk, chunk, *u.shape[1:]))
    return chunks.reshape(u.shape)


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


ROBUST_WIDTHS = 8  # how far the flow's axis reaches past its median, in robust widths


def axis_ranges(
    flow: np.ndarray,
    fisher: np.ndarray,
    scale: list[str],
    mcmc: np.ndarray | None = None,
    robust: bool = True,
) -> list[tuple]:
    """Each axis' plot range: the union of the central 99% (0.5-99.5%) of the reference draws
    (the MCMC's, or the Fisher's where there is no MCMC) and of the flow's, padded by 5% in its
    own scale. With `robust`, the flow's part is further held to its median +- ROBUST_WIDTHS
    robust widths (1.4826 MAD): a percent of strays (draws latched onto another source, a bin
    or more away) would otherwise set the axis and squash a 0.04-bin posterior into one
    histogram bin. Pooled pages, multimodal by construction, pass robust=False. Draws outside
    are left off the page; corner_page says how many."""
    ranges = []
    for c in range(flow.shape[1]):
        sets = [flow[:, c], mcmc[:, c] if mcmc is not None else fisher[:, c]]
        if scale[c] == "log":
            # a linear-space Fisher can spill below zero on a log axis -- ignore those
            sets = [np.log10(x[x > 0]) for x in sets]
        bounds = [np.quantile(x, [0.005, 0.995]) for x in sets if x.size]
        if robust and sets[0].size:
            med = np.median(sets[0])
            width = 1.4826 * np.median(np.abs(sets[0] - med))
            if width > 0:
                b = bounds[0]
                bounds[0] = np.array([max(b[0], med - ROBUST_WIDTHS * width), min(b[1], med + ROBUST_WIDTHS * width)])
        lo, hi = min(b[0] for b in bounds), max(b[1] for b in bounds)
        f = fisher[:, c]
        f = np.log10(f[f > 0]) if scale[c] == "log" else f
        if f.size and not np.any((f >= lo) & (f <= hi)):
            # a Fisher wholly off the page would make its density histogram 0/0: take it in
            fb = np.quantile(f, [0.005, 0.995])
            lo, hi = min(lo, fb[0]), max(hi, fb[1])
        m = 0.05 * (hi - lo) or 0.1
        lo, hi = lo - m, hi + m
        ranges.append((10**lo, 10**hi) if scale[c] == "log" else (lo, hi))
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
    mcmc: np.ndarray | None = None,
    save_samples: bool = True,
) -> None:
    """One corner page: flow draws (orange) over Fisher draws (blue, dashed), truths in black,
    and the MCMC (green) when given (canna.lisa.mcmc's narrowband chains).

    flow and fisher are [n, ndim]; truths is [k, ndim]. Each truth row draws lines across
    the page; that is what marks a full page, whose rows are the label permutations of the
    truth. A pooled page stacks every source into the same ndim parameters, so a row is one
    source: it gets a line on the diagonal and a point in the 2D panels, since lines there
    would cross at k^2 places of which only k are real.

    Everything the page plots is saved beside it (path with .npz), so `replot_page` can
    restyle it later with no network, Fisher or MCMC run.
    """
    import corner
    import matplotlib.pyplot as plt

    if save_samples:
        extra = {} if mcmc is None else {"mcmc": mcmc}
        np.savez_compressed(
            path.with_suffix(".npz"), flow=flow, fisher=fisher, truths=truths,
            labels=np.array(labels), scale=np.array(scale), title=np.array(title),
            pooled=pooled, bins=bins, **extra,
        )
    ranges = axis_ranges(flow, fisher, scale, mcmc, robust=not pooled)
    # draws with any coordinate off the page, so the cut is never silent
    lo_hi = np.array(ranges)
    off = np.mean(np.any((flow < lo_hi[:, 0]) | (flow > lo_hi[:, 1]), axis=1))
    if off > 0:
        title = f"{title}\n{100 * off:.1f}% of the flow draws lie outside the plotted ranges"
    # a log axis over less than a factor of 2 (f0 inside one window, ~1%) only crowds its
    # tick labels together; draw it linear. Amplitudes span decades and stay log
    scale = ["linear" if sc == "log" and hi < 2 * lo else sc for sc, (lo, hi) in zip(scale, ranges)]
    style = dict(range=ranges, axes_scale=scale, bins=bins)
    # corner writes its color into hist_kwargs, so a dict shared by the calls left every
    # marginal in the flow's orange: each call gets its own
    fig = corner.corner(
        flow, labels=labels, color="C1", show_titles=not pooled, title_fmt=".2g",
        hist_kwargs={"density": True}, **style,
    )
    corner.corner(
        fisher,
        fig=fig,
        color="C0",
        plot_datapoints=False,
        contour_kwargs={"linestyles": "dashed"},
        hist_kwargs={"density": True},
        **style,
    )
    if mcmc is not None:
        corner.corner(
            mcmc, fig=fig, color="C2", plot_datapoints=False, hist_kwargs={"density": True}, **style
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


def replot_page(samples: Path, path: Path | None = None) -> Path:
    """Redraw a corner page from the samples corner_page saved beside it (default: the .pdf
    next to the .npz), with the current styling."""
    d = np.load(samples)
    path = samples.with_suffix(".pdf") if path is None else path
    corner_page(
        path, d["flow"], d["fisher"], list(d["labels"]), list(d["scale"]), str(d["title"]),
        d["truths"], pooled=bool(d["pooled"]), bins=int(d["bins"]),
        mcmc=d["mcmc"] if "mcmc" in d else None, save_samples=False,
    )
    return path


def eval_injections(problem: LisaGB, seed: int):
    """The ten eval injections, spread over the SNR distribution of N_CANDIDATES prior draws.

    Returns (latents [10, S, 8], windows, window SNRs, quantiles, noise keys, prior precision).
    Every script that compares against the eval (eval_draws.py, canna.lisa.mcmc, the
    scorecard's eval rows) draws them here, so the data are the same noise realisations.
    """
    key_pick, key_window, key_noise = jr.split(jr.key(seed), 3)
    # each injection gets its own measurement setup, and keeps it all the way through
    windows = jax.vmap(problem.sample_f)(jr.split(key_window, N_CANDIDATES))
    latents = jax.vmap(problem.sample_physical)(jr.split(key_pick, N_CANDIDATES), windows)
    # Gaussian-approx prior precision, in physical units
    prior_prec_p = scaled_inverse(jnp.cov(latents.reshape(N_CANDIDATES, -1).T))
    snrs = np.asarray(jax.lax.map(lambda pc: problem.snr(*pc), (latents, windows)))
    quantiles = np.linspace(1.0 / N_QUANTILES, 1.0, N_QUANTILES)
    chosen = np.argsort(snrs)[np.round(quantiles * (N_CANDIDATES - 1)).astype(int)]
    # keep the latents on device: clean_signal writes the chirp in with .at[].set(),
    # which a numpy array does not have. jax indexes fine with numpy integer arrays
    return (
        latents[chosen], windows[chosen], snrs[chosen], quantiles,
        jr.split(key_noise, N_QUANTILES), prior_prec_p,
    )


def fisher_samples(
    problem: LisaGB, latent: Array, f: Array, key_n: Array, prior_prec_p: Array
) -> np.ndarray:
    """Fisher forecast draws at an injection [N_POSTERIOR, S, 8], in physical parameters.

    The nll Hessian averaged over noise realizations (the data-dependent part cancels, the
    Hessian at a single draw is indefinite), plus the prior precision.
    """
    p0 = latent.reshape(-1)
    nll = lambda p, o: -problem.log_likelihood(p.reshape(latent.shape), o, f)
    replicas = jax.vmap(problem.sample_observation, in_axes=(0, None, None))(
        jr.split(jr.fold_in(key_n, 2), N_FISHER_DRAWS), latent, f
    )
    prec_p = jax.lax.map(lambda o: jax.hessian(nll)(p0, o), replicas).mean(0) + prior_prec_p
    draws = fisher_draws(jr.fold_in(key_n, 1), p0, prec_p, prior_prec_p, N_POSTERIOR)
    return np.asarray(draws).reshape(N_POSTERIOR, *latent.shape)


def gb_picks(source_snr: np.ndarray) -> dict[str, tuple[int, int]]:
    """The per-GB pages' sources: the loudest, median and faintest of all the eval sources,
    as (injection, source) from their SNRs [injection, source]."""
    order = sorted(np.ndindex(source_snr.shape), key=lambda js: source_snr[js])
    return {"loudest": order[-1], "median": order[len(order) // 2], "faintest": order[0]}


def labels_and_scales(n_sources: int) -> tuple[list[str], list[str]]:
    """Axis labels and scales of a full page: every source's parameters side by side."""
    labels = [
        f"${label}$" + (f" (s{s})" if n_sources > 1 else "")
        for s in range(n_sources)
        for label in PARAM_LABELS
    ]
    scale = ["log" if is_log else "linear" for _ in range(n_sources) for is_log in PARAM_IS_LOG]
    return labels, scale


def fold_orientation(p: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Draws [..., 8] moved by the likelihood's exact symmetries psi -> psi + pi and
    (psi, phi0) -> (psi + pi/2, phi0 + pi) onto the copy nearest the truth [8] (or [S, 8]
    against draws [..., S, 8]): psi within pi/4 of it, phi0 within pi.

    A pair (psi, phi0) is only defined up to these, and the sample sets cover different
    copies: the MCMC the two or three its boxes reach, a psi_phi0 flow all four, a helicity
    flow a fixed one (LisaGB.polarisation). Folded, the pages compare like with like, as
    canna.lisa.mcmc.fold does for the tables."""
    p = np.array(p, np.float64)
    k = np.round((p[..., 5] - truth[..., 5]) / (np.pi / 2))
    p[..., 5] -= k * np.pi / 2
    p[..., 7] = truth[..., 7] + (p[..., 7] - k * np.pi - truth[..., 7] + np.pi) % (2 * np.pi) - np.pi
    return p


def injection_pages(
    corner_dir: Path,
    tag: str,
    quantile: float,
    snr: float,
    draws: np.ndarray,
    fisher: np.ndarray,
    truth: np.ndarray,
    mcmc: np.ndarray | None = None,
) -> None:
    """The full and pooled pages of one injection. draws and fisher are [N, S, 8], mcmc
    [M, S, 8] (each source's own chains side by side) or None."""
    n_sources, n_params = truth.shape
    draws, fisher = fold_orientation(draws, truth), fold_orientation(fisher, truth)
    mcmc = None if mcmc is None else fold_orientation(mcmc, truth)
    labels, scale = labels_and_scales(n_sources)
    with_mcmc = ", MCMC (green)" if mcmc is not None else ""
    # full page: every parameter of every source. The Fisher is centred on one labelling
    # and the posterior on all of them, so the Fisher draws and the truth lines come in
    # every permutation. The MCMC follows the true labels, as the Fisher does
    perms = list(itertools.permutations(range(n_sources)))
    corner_page(
        corner_dir / f"{tag}.pdf",
        draws.reshape(len(draws), -1),
        np.concatenate([fisher[:, list(p)] for p in perms]).reshape(-1, len(labels)),
        labels,
        scale,
        f"flow (orange) vs Fisher (blue){with_mcmc}, SNR quantile {quantile:.1f}, SNR={snr:.1f}",
        np.stack([truth[list(p)].reshape(-1) for p in perms]),
        mcmc=None if mcmc is None else mcmc.reshape(len(mcmc), -1),
    )
    # pooled page: [draw, source, param] -> [draw * source, param], so each marginal
    # holds every source at once (f0 shows one spike per GB) and labels drop out
    corner_page(
        corner_dir / f"{tag}-pooled.pdf",
        draws.reshape(-1, n_params),
        fisher.reshape(-1, n_params),
        [f"${label}$" for label in PARAM_LABELS],
        scale[:n_params],
        f"all {n_sources} sources pooled, flow (orange) vs Fisher (blue){with_mcmc},"
        f" SNR quantile {quantile:.1f}, SNR={snr:.1f}",
        truth,
        pooled=True,
        bins=50,
        mcmc=None if mcmc is None else mcmc.reshape(-1, n_params),
    )


def gb_page(
    corner_dir: Path,
    name: str,
    n_eval_sources: int,
    snr: float,
    tag: str,
    s: int,
    flow_s: np.ndarray,
    fisher_s: np.ndarray,
    truth_s: np.ndarray,
    mcmc_s: np.ndarray | None = None,
) -> None:
    """One per-GB page: a single source's draws, relabelled onto it, against its Fisher."""
    with_mcmc = ", MCMC (green)" if mcmc_s is not None else ""
    flow_s, fisher_s = fold_orientation(flow_s, truth_s), fold_orientation(fisher_s, truth_s)
    mcmc_s = None if mcmc_s is None else fold_orientation(mcmc_s, truth_s)
    corner_page(
        corner_dir / f"gb-{name}.pdf",
        flow_s,
        fisher_s,
        [f"${label}$" for label in PARAM_LABELS],
        ["log" if is_log else "linear" for is_log in PARAM_IS_LOG],
        f"{name} of the {n_eval_sources} eval sources: SNR {snr:.1f}"
        f" (injection {tag}, source {s}), flow (orange) vs Fisher (blue){with_mcmc}",
        truth_s[None],
        mcmc=mcmc_s,
    )


if __name__ == "__main__":
    # headless, and only here: scorecard imports sample_posterior from this module, and
    # selecting the backend at import would switch any notebook that does the same to Agg
    # (see train.py). corner_page imports pyplot itself, after this
    import matplotlib

    matplotlib.use("Agg")
    from .mcmc import load_chains

    args = parse_args()

    # rebuild the state skeleton, then overwrite its params from the checkpoint
    state, _, out_dir = load_trained(args)
    problem, flow = state.problem, state.flow
    corner_dir = out_dir / "corner"
    corner_dir.mkdir(parents=True, exist_ok=True)

    latents, windows, snrs, quantiles, noise_keys, prior_prec_p = eval_injections(
        problem, args.seed
    )
    # the narrowband MCMC of the eval's sources, where it has been run (canna.lisa.mcmc)
    chains = load_chains(problem, args.seed)
    if chains:
        print(f"MCMC overlay: {len(chains)} sources ({sorted(chains)})", flush=True)

    # every eval source, for the per-GB pages once all injections are done
    sources = {}
    for j, key_n in enumerate(noise_keys):
        latent, f = latents[j], windows[j]
        truth = np.asarray(latent)
        tag = f"q{quantiles[j]:.2f}"

        # inject, sample the flow, and map back to physical units: [draw, source, param]
        o = problem.sample_observation(key_n, latent, f)
        y = problem.preprocess(o, f)
        u0 = jax.vmap(problem.sample_flow, in_axes=(0, None))(jr.split(key_n, N_POSTERIOR), f)
        post = sample_posterior(problem, flow, u0, y, f, args.ode_steps or ODE_STEPS, args.time_power)
        draws = np.asarray(jax.vmap(problem.flow_to_physical, in_axes=(0, None))(post, f))
        fisher = fisher_samples(problem, latent, f, key_n, prior_prec_p)

        complete = all((j, s) in chains for s in range(problem.n_sources))
        mcmc = np.stack([chains[j, s] for s in range(problem.n_sources)], axis=1) if complete else None
        injection_pages(corner_dir, tag, quantiles[j], snrs[j], draws, fisher, truth, mcmc)
        print(f"[{j + 1}/{N_QUANTILES}] {tag} -> {corner_dir / tag}(-pooled).pdf", flush=True)

        # keep each source, its draws relabelled onto it, for the per-GB pages
        matched = match_sources(problem, draws, truth, f)
        source_snr = np.asarray(problem.snr(jnp.asarray(truth)[:, None, :], f))
        for s in range(problem.n_sources):
            sources[j, s] = (source_snr[s], tag, matched[:, s], fisher[:, s], truth[s])

    # per-GB pages: the loudest, the median and the faintest single source over all the
    # eval injections, each against its own Fisher forecast
    source_snr = np.array([[sources[j, s][0] for s in range(problem.n_sources)] for j in range(N_QUANTILES)])
    for name, (j, s) in gb_picks(source_snr).items():
        snr, tag, flow_s, fisher_s, truth_s = sources[j, s]
        gb_page(corner_dir, name, len(sources), snr, tag, s, flow_s, fisher_s, truth_s, chains.get((j, s)))

    print(
        f"saved {N_QUANTILES} full and {N_QUANTILES} pooled pages and the"
        f" loudest/median/faintest GB pages to {corner_dir}",
        flush=True,
    )
    if peak_memory_report():
        print(peak_memory_report(), flush=True)
