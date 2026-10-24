"""Strub et al. 2022's faint source (arXiv:2204.04467, Table VII): its snr and MCMC posteriors on today's simulator."""

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Complex, Float

from canna.lisa import config as run_config
from canna.lisa import mcmc
from canna.lisa.mcmc import (
    ANGLES,
    N_PARAMS,
    Window,
    gelman_rubin,
    log_likelihood,
    sample_chains,
    to_physical,
    to_sampler,
)
from canna.lisa.physics import (
    GRAVITATIONAL_CONSTANT,
    SPEED_OF_LIGHT,
    SUN_MASS,
)
from canna.lisa.problem import WINDOW_STRIDE, LisaGB, f0_window, window_power
from canna.lisa.simulator import clean_signal, sample_observation, snr

warnings.filterwarnings("ignore", message=r"Using `field\(init=False\)`")

HERE = Path(__file__).parents[1]
CHAINS = Path("outputs/strub-source")
F0, LOG10_FDOT, AMPLITUDE = 1.40457e-3, -17.3, 4.55e-23
ECLIPTIC_LON, ECLIPTIC_LAT = 1.5, 0.2
PSI, IOTA, PHI0 = 2.0, 1.2, 3.0
T_OBS = 63115200.0
OBLIQUITY = np.deg2rad(23.4392911)
# Strub et al.'s priors: f0 +- 0.5 uHz, log10 fdot and log10 A uniform, isotropic sky and orientation
F0_HALF_WIDTH = 0.5e-6
LOG10_FDOT_RANGE = (-18.5, -16.0)
LOG10_A_RANGE = (-23.5, -21.5)
N_REALISATIONS = 48
SIDEBAND_CUT = 15e-9  # [Hz], half the 1/yr antenna sideband spacing
# Strub et al.'s reported seed posterior: the f0 interval and iota
STRUB_REPORTED = dict(
    strub_f0_plus_nhz=1.9,
    strub_f0_minus_nhz=1.5,
    strub_iota_median=1.33,
    strub_iota_plus=0.10,
    strub_iota_minus=0.18,
)
NOISE_SEED, SWEEP_SEED = 0, 1000
N_START = 4096  # the Fisher-and-box draws the walkers start from
STRUB_NAMES = [
    "f0",
    "log10_fdot",
    "log10_A",
    "ecl_lon",
    "ecl_lat",
    "psi",
    "cos_iota",
    "phi0",
]


def ecliptic_to_equatorial(lon, lat):
    """Rotate ecliptic (lambda, beta) about x by the obliquity into equatorial (RA, Dec)."""
    x, y, z = np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)
    c, s = np.cos(OBLIQUITY), np.sin(OBLIQUITY)
    return np.arctan2(c * y - s * z, x), np.arcsin(np.clip(s * y + c * z, -1, 1))


def equatorial_to_ecliptic(ra, dec):
    x, y, z = np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)
    c, s = np.cos(OBLIQUITY), np.sin(OBLIQUITY)
    return np.arctan2(c * y + s * z, x), np.arcsin(np.clip(-s * y + c * z, -1, 1))


def chirp_mass(fdot: float, f0: float) -> float:
    """The chirp mass [solar masses] whose radiation reaction gives fdot at f0."""
    ratio = 96.0 / 5.0 * np.pi ** (8 / 3) * f0 ** (11 / 3)
    return (
        (fdot / ratio) ** (3 / 5)
        * SPEED_OF_LIGHT**3
        / (GRAVITATIONAL_CONSTANT * SUN_MASS)
    )


def truth_physical(rotate: bool = True) -> Float[np.ndarray, "8"]:
    """Strub's source as [f0, fdot, A, RA, Dec, psi, pi/2 - iota, phi0]."""
    sky = (
        ecliptic_to_equatorial(ECLIPTIC_LON, ECLIPTIC_LAT)
        if rotate
        else (ECLIPTIC_LON, ECLIPTIC_LAT)
    )
    return np.array([F0, 10**LOG10_FDOT, AMPLITUDE, *sky, PSI, np.pi / 2 - IOTA, PHI0])


def to_strub(p: Float[np.ndarray, "N 8"]) -> Float[np.ndarray, "N 8"]:
    """Physical draws -> Strub's variables [f0, log10 fdot, log10 A, lambda, beta, psi, cos iota, phi0]."""
    p = np.asarray(p, np.float64)
    lon, lat = equatorial_to_ecliptic(p[:, 3], p[:, 4])
    return np.stack(
        [
            p[:, 0],
            np.log10(p[:, 1]),
            np.log10(p[:, 2]),
            lon,
            lat,
            p[:, 5],
            np.sin(p[:, 6]),
            p[:, 7],
        ],
        axis=-1,
    )


def make_problem(response_points: int) -> LisaGB:
    """One source, 2 yr."""
    return LisaGB(
        n_sources=1,
        t_obs=T_OBS,
        response_points=response_points,
        f0_range=(1.0e-3, 2.0e-3),
    )


def window_centre(problem: LisaGB, f0: float = F0) -> Float[Array, ""]:
    """The snapped window (as sample_f) whose centre is nearest f0."""
    index = np.round((f0 * problem.t_obs - problem.window_bins / 2) / WINDOW_STRIDE)
    return jnp.asarray(
        (index * WINDOW_STRIDE + problem.window_bins / 2) / problem.t_obs
    )


def inner(
    a: Complex[Array, "F 3"], b: Complex[Array, "F 3"], power: Float[Array, "F 3"]
) -> float:
    """Noise-weighted inner product <a|b> = 2 sum Re(a b*) / P over A, E, T."""
    return float(2.0 * jnp.sum((a * jnp.conj(b)).real / power))


def prior_box(centres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Strub's prior box in sampler coordinates: psi spans pi, as theirs, the other angles 2 pi."""
    lower = np.array(
        [
            np.log(F0 - F0_HALF_WIDTH),
            LOG10_FDOT_RANGE[0] * np.log(10),
            LOG10_A_RANGE[0] * np.log(10),
            0,
            -np.pi / 2,
            0,
            -1,
            0,
        ]
    )
    upper = np.array(
        [
            np.log(F0 + F0_HALF_WIDTH),
            LOG10_FDOT_RANGE[1] * np.log(10),
            LOG10_A_RANGE[1] * np.log(10),
            0,
            np.pi / 2,
            0,
            1,
            0,
        ]
    )
    half = np.array([0, 0, 0, np.pi, 0, np.pi / 2, 0, np.pi])
    lower[ANGLES], upper[ANGLES] = (centres - half)[ANGLES], (centres + half)[ANGLES]
    return lower, upper


def strub_window(
    problem: LisaGB, f, obs, truth: np.ndarray, lower=None, upper=None
) -> Window:
    """mcmc's one-source Window for a hand-placed source, its box Strub's prior (strub_log_prior)."""
    centres = np.zeros(N_PARAMS)
    if lower is None:
        centres[ANGLES] = to_sampler(truth, np.zeros(N_PARAMS))[ANGLES]
        lower, upper = prior_box(centres)
    else:
        centres[ANGLES] = ((lower + upper) / 2)[ANGLES]
    f_lo, f_hi = (float(v) for v in f0_window(problem, f))
    assert (
        f_lo <= np.exp(lower[0]) and np.exp(upper[0]) <= f_hi
    ), "f0 box overruns the window"
    rho = float(snr(problem, jnp.asarray(truth)[None], f))
    window = Window(
        j=0,
        f=f,
        truth=truth[None],
        snr=np.array([rho]),
        window_snr=rho,
        obs=obs,
        power=window_power(problem, f),
        centres=centres[None],
        lower=lower[None],
        upper=upper[None],
        fisher=None,
    )
    # the walkers' start: the Fisher at the truth plus the box as a weak Gaussian
    t0 = to_sampler(truth, centres)
    precision = fisher(problem, window, t0) + np.diag(12.0 / (upper - lower) ** 2)
    theta = np.random.default_rng(0).multivariate_normal(
        t0, np.linalg.inv(precision), N_START
    )
    return window._replace(fisher=np.asarray(to_physical(jnp.asarray(theta)))[:, None])


def strub_log_prior(window: Window, theta: jax.Array) -> jax.Array:
    """Strub's prior in sampler coordinates: flat over the box, an isotropic sky."""
    inside = jnp.all((theta >= window.lower[0]) & (theta <= window.upper[0]))
    return jnp.where(inside, jnp.log(jnp.cos(theta[4])), -jnp.inf)


def fisher(problem: LisaGB, window: Window, theta: np.ndarray) -> np.ndarray:
    """Fisher matrix in sampler coordinates at theta."""

    def signal(t):
        h = clean_signal(problem, to_physical(t)[None], window.f) / jnp.sqrt(
            window.power
        )
        return jnp.concat([h.real.ravel(), h.imag.ravel()])

    jac = jax.jacfwd(signal)(jnp.asarray(theta))
    return 2.0 * np.asarray(jac.T @ jac)


def fisher_box(
    problem: LisaGB, window: Window, theta_ml: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Box at the ML point: +-3 sigma of the Fisher without fdot, f0 +-1 sigma, psi and phi0 pinned."""
    keep = [0, 2, 3, 4, 5, 6, 7]
    sigma = np.full(N_PARAMS, np.nan)
    sigma[keep] = np.sqrt(
        np.diag(np.linalg.inv(fisher(problem, window, theta_ml)[np.ix_(keep, keep)]))
    )
    half = 3 * sigma
    half[0], half[5], half[7] = sigma[0], np.pi / 1000, 2 * np.pi / 1000
    lower, upper = window.lower[0].copy(), window.upper[0].copy()
    lower[keep] = np.maximum(lower[keep], (theta_ml - half)[keep])
    upper[keep] = np.minimum(upper[keep], (theta_ml + half)[keep])
    return lower, upper, sigma


def refine(
    problem: LisaGB, window: Window, theta0: np.ndarray
) -> tuple[np.ndarray, float]:
    """L-BFGS-B from the best chain draw to the box's local likelihood maximum, in Fisher-scaled steps."""
    from scipy.optimize import minimize

    lower, upper = window.lower[0], window.upper[0]
    precision = fisher(problem, window, theta0) + np.diag(12.0 / (upper - lower) ** 2)
    scale = np.sqrt(np.diag(np.linalg.inv(precision)))
    loss = jax.jit(
        lambda u: -log_likelihood(problem, window, jnp.asarray(theta0) + u * scale)
    )
    grad = jax.jit(jax.jacfwd(loss))
    bounds = list(zip((lower - theta0) / scale, (upper - theta0) / scale))
    res = minimize(
        lambda u: float(loss(u)),
        np.zeros(N_PARAMS),
        jac=lambda u: np.asarray(grad(u)),
        method="L-BFGS-B",
        bounds=bounds,
    )
    return theta0 + res.x * scale, float(-res.fun)


def sample_source(problem: LisaGB, window: Window, args, seed: int) -> dict:
    """mcmc.sample_chains on the one-source window, its per-source arrays without the source axis."""
    out = sample_chains(problem, window, args, seed)
    return out | {k: out[k][..., 0, :] for k in ("physical", "rhat", "tau", "ess")}


def chain_summary(
    problem: LisaGB, window: Window, out: dict, nwalker: int, prefix: str
) -> dict:
    """The arrays a task keeps: draws in Strub's variables, R-hats, the ML point and its snr."""
    chains = out["samples"].reshape(nwalker, -1, N_PARAMS)
    best = int(np.argmax(out["loglik"]))
    theta_ml = out["samples"][best]
    ll_ml = float(log_likelihood(problem, window, jnp.asarray(theta_ml)))
    theta_opt, ll_opt = refine(problem, window, theta_ml)
    return {
        f"{prefix}strub": to_strub(out["physical"]),
        f"{prefix}loglik": out["loglik"],
        f"{prefix}rhat": out["rhat"],
        f"{prefix}rhat_walker": np.array(
            [gelman_rubin(chains[:, :, d], groups=nwalker) for d in range(N_PARAMS)]
        ),
        f"{prefix}tau": out["tau"],
        f"{prefix}ess": out["ess"],
        f"{prefix}acceptance": out["acceptance"],
        f"{prefix}seconds": out["seconds"],
        f"{prefix}theta_ml": theta_ml,
        f"{prefix}strub_ml": to_strub(
            np.asarray(to_physical(jnp.asarray(theta_ml)))[None]
        )[0],
        f"{prefix}loglik_ml": ll_ml,
        f"{prefix}loglik_ml_chain": float(out["loglik"][best]),
        f"{prefix}theta_opt": theta_opt,
        f"{prefix}loglik_opt": ll_opt,
    }


def chain(args) -> None:
    """Task 0: noiseless; task 1: noise seed 0, full prior then Fisher box; task 2 + k: realisation k."""
    mcmc.log_prior = strub_log_prior  # in place of the catalogue prior, for sample_chains
    problem = make_problem(args.response_points)
    f = window_centre(problem)
    truth = truth_physical()
    h = clean_signal(problem, jnp.asarray(truth)[None], f)
    if args.task == 0:
        obs, noise_key = h, -1
    else:
        noise_key = NOISE_SEED if args.task == 1 else SWEEP_SEED + args.task - 2
        obs = sample_observation(
            problem, jr.key(noise_key), jnp.asarray(truth)[None], f
        )
    window = strub_window(problem, f, obs, truth)
    power = window.power
    rho_opt = float(snr(problem, jnp.asarray(truth)[None], f))
    record = dict(
        task=args.task,
        noise_key=noise_key,
        rho_opt=rho_opt,
        rho_obs=inner(obs, h, power) / rho_opt,
        response_points=problem.response_points,
        nwalker=args.nwalker,
        temps=np.asarray(args.temps),
        burn=args.burn,
        keep=args.keep,
        thin=args.thin,
        lower=window.lower[0],
        upper=window.upper[0],
    )
    out = sample_source(problem, window, args, seed=args.task)
    record |= chain_summary(problem, window, out, args.nwalker, "")
    print(
        f"task {args.task}: {out['seconds']:.0f}s, rhat max {out['rhat'].max():.4f}",
        flush=True,
    )

    if args.task == 1:
        lower, upper, sigma = fisher_box(problem, window, record["theta_ml"])
        ml = np.asarray(to_physical(jnp.asarray(record["theta_ml"])))
        box = strub_window(problem, f, obs, ml, lower, upper)
        out = sample_source(problem, box, args, seed=1001)
        record |= chain_summary(problem, box, out, args.nwalker, "box_")
        record |= dict(box_lower=lower, box_upper=upper, fisher_sigma=sigma)
        print(
            f"box: {out['seconds']:.0f}s, rhat max {out['rhat'].max():.4f}", flush=True
        )

    CHAINS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CHAINS / f"task-{args.task:03d}.npz", **record)


def posterior(strub: np.ndarray, prefix: str) -> dict:
    """Widths and correlations of one posterior in Strub's variables."""
    q16, q50, q84 = np.percentile(strub, [16, 50, 84], axis=0)
    iota = np.arccos(strub[:, 6])
    i16, i50, i84 = np.percentile(iota, [16, 50, 84])
    out = {f"{prefix}std_{n}": float(v) for n, v in zip(STRUB_NAMES, strub.std(axis=0))}
    out |= {f"{prefix}median_{n}": float(v) for n, v in zip(STRUB_NAMES, q50)}
    out |= {
        f"{prefix}sigma_f0_nhz": float(strub[:, 0].std() * 1e9),
        f"{prefix}f0_plus_nhz": float((q84[0] - q50[0]) * 1e9),
        f"{prefix}f0_minus_nhz": float((q50[0] - q16[0]) * 1e9),
        f"{prefix}corr_f0_log10_fdot": float(
            np.corrcoef(strub[:, 0], strub[:, 1])[0, 1]
        ),
        f"{prefix}corr_cos_iota_log10_A": float(
            np.corrcoef(strub[:, 6], strub[:, 2])[0, 1]
        ),
        f"{prefix}log10_fdot_std_over_prior_std": float(
            strub[:, 1].std() / (np.diff(LOG10_FDOT_RANGE)[0] / np.sqrt(12))
        ),
        f"{prefix}log10_fdot_q05": float(np.percentile(strub[:, 1], 5)),
        f"{prefix}log10_fdot_q95": float(np.percentile(strub[:, 1], 95)),
        f"{prefix}sigma_cos_iota": float(strub[:, 6].std()),
        f"{prefix}cos_iota_half_width68": float((q84[6] - q16[6]) / 2),
        f"{prefix}cos_iota_q16": float(q16[6]),
        f"{prefix}cos_iota_q84": float(q84[6]),
        f"{prefix}f0_off_mode_fraction": float(
            np.mean(np.abs(strub[:, 0] - q50[0]) > SIDEBAND_CUT)
        ),
        f"{prefix}iota_median": float(i50),
        f"{prefix}iota_plus": float(i84 - i50),
        f"{prefix}iota_minus": float(i50 - i16),
    }
    return out


def save_seed0(t) -> None:
    """Seed 0's f0 (nHz from its median) and cos iota draws, both priors, for the page: data/seed0.npz."""
    out = {}
    for prefix, strub in (("strub", t["strub"]), ("box", t["box_strub"])):
        out[f"{prefix}_f0_nhz"] = ((strub[:, 0] - np.median(strub[:, 0])) * 1e9).astype(np.float32)
        out[f"{prefix}_cos_iota"] = strub[:, 6].astype(np.float32)
    np.savez_compressed(HERE / "data" / "seed0.npz", **out)


def summarise(args) -> None:
    """SNR checks, then every task's chains, into data/strub_source.json."""
    truth = truth_physical()
    record: dict = dict(
        commit=subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        src_tree=subprocess.run(
            ["git", "rev-parse", "HEAD:src"], capture_output=True, text=True
        ).stdout.strip(),
        f0=F0,
        log10_fdot=LOG10_FDOT,
        amplitude=AMPLITUDE,
        ecliptic_lon=ECLIPTIC_LON,
        ecliptic_lat=ECLIPTIC_LAT,
        psi=PSI,
        iota=IOTA,
        phi0=PHI0,
        t_obs=T_OBS,
        ra=float(truth[3]),
        dec=float(truth[4]),
        chirp_mass=chirp_mass(10**LOG10_FDOT, F0),
    )

    # snr, all channels and A, E alone, over window sizes, the full band's problem and unrotated angles
    for rp in args.snr_response_points:
        problem = make_problem(rp)
        f = window_centre(problem)
        h = clean_signal(problem, jnp.asarray(truth)[None], f)
        power = window_power(problem, f)
        record[f"snr_rp{rp}"] = float(snr(problem, jnp.asarray(truth)[None], f))
        record[f"snr_ae_rp{rp}"] = float(
            np.sqrt(inner(h[:, :2], h[:, :2], power[:, :2]))
        )
        record[f"snr_unrotated_rp{rp}"] = float(
            snr(problem, jnp.asarray(truth_physical(False))[None], f)
        )
    full_b = LisaGB(**run_config.load("full-B").problem)
    record["snr_full_b"] = float(
        snr(full_b, jnp.asarray(truth)[None], window_centre(full_b))
    )
    record["full_b_response_points"] = full_b.response_points
    problem = make_problem(args.response_points)
    record["response_points"] = problem.response_points
    record["snr"] = float(
        snr(problem, jnp.asarray(truth)[None], window_centre(problem))
    )
    record["snr_strub"] = 7.93
    record["snr_relative_gap"] = record["snr"] / 7.93 - 1
    record |= STRUB_REPORTED

    tasks = {
        int(p.stem.split("-")[1]): dict(np.load(p))
        for p in sorted(CHAINS.glob("task-*.npz"))
    }
    first = next(iter(tasks.values()))
    record |= dict(
        nwalker=int(first["nwalker"]),
        temps=first["temps"].tolist(),
        burn=int(first["burn"]),
        keep=int(first["keep"]),
        thin=int(first["thin"]),
        chain_response_points=int(first["response_points"]),
        noise_seed=NOISE_SEED,
        sweep_seed=SWEEP_SEED,
        tasks=sorted(tasks),
        seconds_max=float(
            max(t["seconds"] + t.get("box_seconds", 0) for t in tasks.values())
        ),
    )

    # noiseless injection: convergence and where the truth sits
    if 0 in tasks:
        t = tasks[0]
        strub = t["strub"]
        truth_strub = to_strub(truth[None])[0]
        record["noiseless_rhat_max"] = float(t["rhat"].max())
        record["noiseless_rhat_walker_max"] = float(t["rhat_walker"].max())
        for i, n in enumerate(mcmc.NAMES):
            key = n.replace(" ", "_")
            record[f"noiseless_rhat_{key}"] = float(t["rhat"][i])
            record[f"noiseless_rhat_walker_{key}"] = float(t["rhat_walker"][i])
        for i, n in enumerate(STRUB_NAMES):
            record[f"noiseless_pull_{n}"] = float(
                (np.median(strub[:, i]) - truth_strub[i]) / strub[:, i].std()
            )
            record[f"noiseless_truth_quantile_{n}"] = float(
                np.mean(strub[:, i] < truth_strub[i])
            )
        record["noiseless_rho_mle"] = float(np.sqrt(2 * max(t["loglik_ml"], 0)))
        record["noiseless_rho_opt"] = float(t["rho_opt"])
        record["noiseless_rho_mle_refined"] = float(
            np.sqrt(2 * max(t["loglik_opt"], 0))
        )

    # seed-0 posterior, under Strub's prior and in the Fisher box
    if 1 in tasks:
        t = tasks[1]
        record["seed0_rho_opt"] = float(t["rho_opt"])
        record["seed0_rho_obs"] = float(t["rho_obs"])
        record["seed0_rho_mle"] = float(np.sqrt(2 * t["loglik_ml"]))
        record["seed0_rho_mle_refined"] = float(np.sqrt(2 * t["loglik_opt"]))
        record["seed0_rhat_max"] = float(t["rhat"].max())
        record |= posterior(t["strub"], "seed0_")
        record |= {
            f"seed0_ml_{n}": float(v) for n, v in zip(STRUB_NAMES, t["strub_ml"])
        }
        if "box_strub" in t:
            record["box_rhat_max"] = float(t["box_rhat"].max())
            record |= posterior(t["box_strub"], "box_")
            record |= {
                f"box_fisher_sigma_{n.replace(' ', '_')}": float(s)
                for n, s in zip(mcmc.NAMES, t["fisher_sigma"])
            }
            record |= {
                f"box_half_width_{n.replace(' ', '_')}": float((hi - lo) / 2)
                for n, lo, hi in zip(mcmc.NAMES, t["box_lower"], t["box_upper"])
            }
            save_seed0(t)

    # 48 noise realisations: snr gain of the best chain sample over the optimal and observed snr
    sweep = [tasks[k] for k in sorted(tasks) if k >= 2]
    if sweep:
        rho_opt = np.array([t["rho_opt"] for t in sweep])
        rho_obs = np.array([t["rho_obs"] for t in sweep])
        rho_mle = np.sqrt(2 * np.array([t["loglik_ml"] for t in sweep]))
        rho_refined = np.sqrt(2 * np.array([t["loglik_opt"] for t in sweep]))
        sigma_f0 = np.array([t["strub"][:, 0].std() * 1e9 for t in sweep])
        rhat = np.array([t["rhat"].max() for t in sweep])
        # draws off the main f0 mode, at the 1/yr antenna sidebands (31.7 nHz) or beyond
        off_mode = np.array(
            [
                np.mean(
                    np.abs(t["strub"][:, 0] - np.median(t["strub"][:, 0]))
                    > SIDEBAND_CUT
                )
                for t in sweep
            ]
        )
        half68 = np.array(
            [
                np.diff(np.percentile(t["strub"][:, 0], [16, 84]))[0] / 2 * 1e9
                for t in sweep
            ]
        )
        converged = rhat < 1.01
        gain_obs, gain_opt = rho_mle - rho_obs, rho_mle - rho_opt
        refined_obs, refined_opt = rho_refined - rho_obs, rho_refined - rho_opt
        n = len(sweep)
        record |= dict(
            sweep_count=n,
            sweep_converged=int(converged.sum()),
            sweep_rhat_max=float(rhat.max()),
            sweep_rho_opt=float(rho_opt[0]),
            sweep_rho_obs_mean=float(rho_obs.mean()),
            sweep_rho_obs_std=float(rho_obs.std(ddof=1)),
            sweep_rho_mle_mean=float(rho_mle.mean()),
            sweep_rho_mle_std=float(rho_mle.std(ddof=1)),
            sweep_gain_obs_mean=float(gain_obs.mean()),
            sweep_gain_obs_std=float(gain_obs.std(ddof=1)),
            sweep_gain_obs_sem=float(gain_obs.std(ddof=1) / np.sqrt(n)),
            sweep_gain_opt_mean=float(gain_opt.mean()),
            sweep_gain_opt_std=float(gain_opt.std(ddof=1)),
            sweep_gain_opt_sem=float(gain_opt.std(ddof=1) / np.sqrt(n)),
            sweep_rho_mle_refined_mean=float(rho_refined.mean()),
            sweep_gain_obs_refined_mean=float(refined_obs.mean()),
            sweep_gain_obs_refined_sem=float(refined_obs.std(ddof=1) / np.sqrt(n)),
            sweep_gain_opt_refined_mean=float(refined_opt.mean()),
            sweep_gain_opt_refined_sem=float(refined_opt.std(ddof=1) / np.sqrt(n)),
            sweep_gain_obs_converged_mean=(
                float(gain_obs[converged].mean()) if converged.any() else None
            ),
            sweep_sigma_f0_nhz_median=float(np.median(sigma_f0)),
            sweep_sigma_f0_nhz_median_converged=(
                float(np.median(sigma_f0[converged])) if converged.any() else None
            ),
            sweep_sigma_f0_nhz_mean=float(sigma_f0.mean()),
            strub_gain=8.25 - 7.93,
            sweep_delta_rho2_refined_mean=float((rho_refined**2 - rho_obs**2).mean()),
            sweep_delta_rho2_refined_sem=float(
                (rho_refined**2 - rho_obs**2).std(ddof=1) / np.sqrt(n)
            ),
            sweep_sigma_f0_nhz_q16=float(np.percentile(sigma_f0, 16)),
            sweep_sigma_f0_nhz_q84=float(np.percentile(sigma_f0, 84)),
            sweep_f0_half_width68_nhz_median=float(np.median(half68)),
            sweep_off_mode_count=int((off_mode > 0.01).sum()),
            sweep_off_mode_fraction_max=float(off_mode.max()),
            sweep_sigma_f0_nhz_median_main_mode=float(
                np.median(sigma_f0[off_mode <= 0.01])
            ),
            sweep_noise_keys=[int(t["noise_key"]) for t in sweep],
            sweep_rho_obs_list=rho_obs.tolist(),
            sweep_rho_mle_list=rho_mle.tolist(),
            sweep_rho_mle_refined_list=rho_refined.tolist(),
            sweep_sigma_f0_nhz_list=sigma_f0.tolist(),
            sweep_off_mode_fraction_list=off_mode.tolist(),
            sweep_rhat_max_list=rhat.tolist(),
        )

    path = HERE / "data" / "strub_source.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))


def extend(args) -> None:
    """Add the seed-0 keys summarise now writes and the JSON predates, from the saved chains; existing keys stay. Rewrites data/seed0.npz."""
    path = HERE / "data" / "strub_source.json"
    record = json.loads(path.read_text())
    t = np.load(CHAINS / "task-001.npz")
    save_seed0(t)
    new = (
        posterior(t["strub"], "seed0_")
        | posterior(t["box_strub"], "box_")
        | STRUB_REPORTED
    )
    added = {k: v for k, v in new.items() if k not in record}
    differing = [k for k, v in new.items() if k in record and not np.isclose(record[k], v)]
    path.write_text(json.dumps(record | added, indent=1) + "\n")
    print(f"added {sorted(added)}; differing from the record, left as is: {differing}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("chain", "summarise", "extend"):
        p = sub.add_parser(name)
        p.add_argument("--response_points", type=int, default=64)
    c = sub.choices["chain"]
    c.add_argument("--task", type=int, required=True)
    c.add_argument("--nwalker", type=int, default=32)
    c.add_argument(
        "--temps", type=float, nargs="+", default=[1.0, 1.58, 2.51, 3.98, 6.31, 10.0]
    )
    c.add_argument("--chunk", type=int, default=2000)
    c.add_argument("--burn", type=int, default=10000)
    c.add_argument("--keep", type=int, default=30000)
    c.add_argument("--thin", type=int, default=10)
    s = sub.choices["summarise"]
    s.add_argument(
        "--snr_response_points", type=int, nargs="+", default=[64, 128, 256, 512]
    )
    args = parser.parse_args()
    {"chain": chain, "summarise": summarise, "extend": extend}[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
