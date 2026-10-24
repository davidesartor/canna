"""Draw the orientation prior per band and measure the helicity chart: angles, z±, round trips, degeneracies, Fisher.

Run from the repo root (JAX: a job, not the login node):
    uv run --frozen python reports/prior/polarization/scripts/draw_polarization.py reports/prior/polarization/data/polarization.json [windows]
"""

import json
import math
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from canna.lisa import config
from canna.lisa.chart import flow_to_physical, physical_to_flow
from canna.lisa.problem import LisaGB, amplitude_unit, sample_f, window_power
from canna.lisa.simulator import clean_signal, sample_physical, source_snrs

BANDS = ("narrow", "full")
WINDOWS = 8192
BINS = 48
QUANTILES = [0.01, 0.16, 0.5, 0.84, 0.99]
Z_NAMES = ["Re z+", "Im z+", "Re z-", "Im z-"]
SCATTER = 1500
CHECK_SOURCES = 256
POLE_GAPS = [1e-2, 1e-3, 1e-6, 1e-9, 0.0]
FISHER_GAPS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
TEST_VECTORS = 12
SHIFT = 0.7  # [rad] the psi step of the pole and degeneracy checks
TAU = 2 * math.pi


def draw(problem: LisaGB, key: jax.Array) -> dict:
    """One window of the prior, as the old prior page drew it (same keys, same seed)."""
    key_f, key_p = jr.split(key)
    f = sample_f(problem, key_f)
    p = sample_physical(problem, key_p, f)
    return dict(
        f=f,
        p=p,
        snr=source_snrs(problem, p, f),
        unit=amplitude_unit(problem, f),
        x=physical_to_flow(problem, p, f),
    )


def hist(
    values: np.ndarray, lo: float, hi: float, reference=None, bins=BINS, cdf=None
) -> dict:
    """Counts on equal bins over [lo, hi], with the prior's expected counts from a density at the centres or a cdf at the edges."""
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    out = dict(
        edges=edges.tolist(),
        counts=counts.tolist(),
        outside=int(values.size - counts.sum()),
    )
    if reference is not None:
        centres = (edges[:-1] + edges[1:]) / 2
        out["reference"] = (
            reference(centres) * values.size * (edges[1] - edges[0])
        ).tolist()
    if cdf is not None:
        out["reference"] = (np.diff(cdf(edges)) * values.size).tolist()
    return out


def spread(values: np.ndarray) -> dict:
    return dict(
        mean=float(values.mean()),
        std=float(values.std()),
        quantiles=np.quantile(values, QUANTILES).tolist(),
    )


def wrap(angle):
    return (angle + math.pi) % TAU - math.pi


def mismatch(problem: LisaGB, pa, pb, f):
    """sqrt(<ha - hb|ha - hb> / <ha|ha>) of two source sets in window f."""
    ha, hb = clean_signal(problem, pa, f), clean_signal(problem, pb, f)
    power = window_power(problem, f)
    return jnp.sqrt(
        jnp.sum(jnp.abs(ha - hb) ** 2 / power) / jnp.sum(jnp.abs(ha) ** 2 / power)
    )


def fisher(problem: LisaGB, to_p, theta, f):
    """F_ij = <d_i h|d_j h> = 2 Re sum d_i h* d_j h / power of one source in window f."""
    jac = jax.jacfwd(lambda t: clean_signal(problem, to_p(t)[None], f))(theta)
    return (
        2
        * jnp.einsum(
            "fci,fcj,fc->ij", jac.conj(), jac, 1 / window_power(problem, f)
        ).real
    )


def angles_to_p(problem: LisaGB, theta, middle: bool):
    """(f0 T, ln fdot, lon, lat, ln A, sin iota, psi, phi) -> p, phi the chart's phase when middle."""
    nu, log_fdot, lon, lat, log_a, s, psi, phi = theta
    f0, fdot, t = nu / problem.t_obs, jnp.exp(log_fdot), problem.t_obs
    if middle:
        phi = phi + jnp.pi * f0 * t + jnp.pi * fdot * t**2 / 4
    return jnp.stack([f0, fdot, jnp.exp(log_a), lon, lat, psi, jnp.arcsin(s), phi])


def normalized(m: np.ndarray) -> np.ndarray:
    """m scaled to unit diagonal; all NaN when float64 cannot represent it (a non-positive diagonal)."""
    diag = np.diag(m)
    if not np.all(np.isfinite(m)) or np.any(diag <= 0):
        return np.full_like(m, np.nan)
    d = np.sqrt(diag)
    return m / np.outer(d, d)


def corr_of(fisher_matrix: np.ndarray, i: int, j: int) -> tuple[float, float]:
    """Conditional (others fixed) and marginal correlation of i and j; NaN where the matrix is singular in float64."""
    f = normalized(fisher_matrix)
    try:
        c = normalized(np.linalg.inv(f))
    except np.linalg.LinAlgError:
        c = np.full_like(f, np.nan)
    return float(-f[i, j]), float(c[i, j])


def condition(fisher_matrix: np.ndarray) -> float:
    """Condition number of the unit-diagonal Fisher matrix (inf when singular in float64)."""
    f = normalized(fisher_matrix)
    return float(np.linalg.cond(f)) if np.all(np.isfinite(f)) else math.nan


def orientation(problem: LisaGB, d: dict) -> dict:
    """The drawn angles, the z± they induce and the chart's flow coordinates: histograms, spreads, scatter."""
    p, x, unit = d["p"], d["x"], d["unit"][:, None]
    psi, iota, phi0 = p[..., 5], p[..., 6], p[..., 7]
    s = np.sin(iota)
    amp = p[..., 2] / unit
    radius = np.hypot(x[..., 4::2], x[..., 5::2])
    mod = np.sinh(radius)  # |z±| in amplitude_unit
    arg = np.arctan2(x[..., 5::2], x[..., 4::2])
    flat = lambda lo, hi: lambda v: np.full_like(v, 1 / (hi - lo))

    # V/I and the axis ratio from |z±| (the chart), against the closed forms over sin iota uniform
    stokes_v = (mod[..., 1] ** 2 - mod[..., 0] ** 2) / (
        mod[..., 1] ** 2 + mod[..., 0] ** 2
    )
    axis_ratio = (mod[..., 1] - mod[..., 0]) / (mod[..., 1] + mod[..., 0])
    grid = np.linspace(-1, 1, 200001)
    v_of_s = 4 * grid * (1 + grid**2) / (1 + 6 * grid**2 + grid**4)
    v_cdf = lambda e: (np.interp(e, v_of_s, grid) + 1) / 2
    eps_cdf = lambda e: np.where(
        e > 0, (1 - np.sqrt(1 - np.minimum(e, 1) ** 2)) / np.maximum(e, 1e-300), 0
    )
    log_ratio = np.log10(mod[..., 0] / mod[..., 1])
    log_amp = np.log10(amp)
    # whole decades around the bulk of A and |z±|, which sit far below one amplitude_unit
    log_lo = math.floor(np.quantile(np.log10(mod), 0.001))
    log_hi = math.ceil(log_amp.max())
    ratio_density = lambda q: math.log(10) / 8 / np.cosh(q * math.log(10) / 4) ** 2

    # the flow's four helicity coordinates, keyed as on the old prior page
    flow = []
    for k, name in enumerate(Z_NAMES):
        v = x[..., 4 + k].ravel()
        flow.append(
            dict(
                name=name,
                hist=hist(v, *np.quantile(v, [0, 1])),
                **spread(v),
            )
        )

    # the same four numbers per source in three coordinate systems
    r_max = float(radius.max())
    systems = dict(
        physical=dict(
            names=["log10 A/unit", "sin iota", "psi [rad]", "phi0 [rad]"],
            values=[log_amp, s, psi, phi0],
            ranges=[tuple(np.quantile(log_amp, [0, 1])), (-1, 1), (0, TAU), (0, TAU)],
        ),
        polar=dict(
            names=["asinh |z+|", "asinh |z-|", "arg z+ [rad]", "arg z- [rad]"],
            values=[radius[..., 0], radius[..., 1], arg[..., 0], arg[..., 1]],
            ranges=[(0, r_max), (0, r_max), (-math.pi, math.pi), (-math.pi, math.pi)],
        ),
        cartesian=dict(
            names=Z_NAMES,
            values=[x[..., 4 + k] for k in range(4)],
            ranges=[(-r_max, r_max)] * 4,
        ),
    )
    flat_refs = dict(
        physical=[None, flat(-1, 1), flat(0, TAU), flat(0, TAU)],
        polar=[None, None, flat(-math.pi, math.pi), flat(-math.pi, math.pi)],
        cartesian=[None] * 4,
    )
    systems = {
        key: dict(
            names=sys_["names"],
            hists=[
                hist(v.ravel(), lo, hi, ref)
                for v, (lo, hi), ref in zip(
                    sys_["values"], sys_["ranges"], flat_refs[key]
                )
            ],
            corr=np.corrcoef(np.stack([v.ravel() for v in sys_["values"]])).tolist(),
        )
        for key, sys_ in systems.items()
    }

    rng = np.random.default_rng(0)
    pick = rng.choice(s.size, size=min(SCATTER, s.size), replace=False)
    # angles and logs to 4 decimals; snr and the radii, far below one, to 4 significant digits
    scatter = {
        k: (
            [float(f"{x:.4g}") for x in np.asarray(v, dtype=np.float64).ravel()[pick]]
            if k in ("snr", "rp", "rm")
            else np.round(np.asarray(v, dtype=np.float64).ravel()[pick], 4).tolist()
        )
        for k, v in dict(
            sin_iota=s,
            psi=psi,
            phi0=phi0,
            log_amp=log_amp,
            snr=d["snr"],
            rp=radius[..., 0],
            rm=radius[..., 1],
            ap=arg[..., 0],
            am=arg[..., 1],
        ).items()
    }

    return dict(
        physical=dict(
            orientation={
                "psi [rad]": hist(psi.ravel(), 0, TAU, flat(0, TAU)),
                "sin iota": hist(s.ravel(), -1, 1, flat(-1, 1)),
                "phi0 [rad]": hist(phi0.ravel(), 0, TAU, flat(0, TAU)),
                "iota [rad]": hist(
                    iota.ravel(), -math.pi / 2, math.pi / 2, lambda v: np.cos(v) / 2
                ),
            }
        ),
        helicity={
            "log10 |z+|": hist(np.log10(mod[..., 0]).ravel(), log_lo, log_hi),
            "log10 |z-|": hist(np.log10(mod[..., 1]).ravel(), log_lo, log_hi),
            "log10 A/unit": hist(log_amp.ravel(), log_lo, log_hi),
            "log10 |z+|/|z-|": hist(log_ratio.ravel(), -8, 8, ratio_density),
            "V/I": hist(stokes_v.ravel(), -1, 1, cdf=v_cdf),
            "axis ratio": hist(np.abs(axis_ratio).ravel(), 0, 1, cdf=eps_cdf),
            "arg z+ [rad]": hist(
                arg[..., 0].ravel(), -math.pi, math.pi, flat(-math.pi, math.pi)
            ),
            "arg z- [rad]": hist(
                arg[..., 1].ravel(), -math.pi, math.pi, flat(-math.pi, math.pi)
            ),
        },
        # the circular part: the larger |z| over the sum of |z|, which is the axis ratio
        axis_ratio_check=float(
            np.max(np.abs(np.abs(axis_ratio) - 2 * np.abs(s) / (1 + s**2)))
        ),
        flow=flow,
        radius=[
            dict(
                name=n,
                hist=hist(radius[..., k].ravel(), 0, r_max),
                rms=float(np.sqrt(np.mean(radius[..., k] ** 2))),
                **spread(radius[..., k].ravel()),
            )
            for k, n in enumerate(["|z+|", "|z-|"])
        ],
        corr=np.corrcoef(x.reshape(-1, 8), rowvar=False).tolist(),
        flow_std=x.reshape(-1, 8).std(0).tolist(),
        systems=systems,
        scatter=scatter,
        # sin iota and the share of the A it takes: the chart's amplitude split
        split_check=float(
            np.max(
                np.abs(
                    (np.sqrt(mod[..., 1]) - np.sqrt(mod[..., 0]))
                    / (np.sqrt(mod[..., 1]) + np.sqrt(mod[..., 0]))
                    - s
                )
            )
        ),
    )


def round_trip(problem: LisaGB, d: dict) -> dict:
    """physical -> flow -> physical over every prior source, in float64 and from a float32 flow output."""
    f, p, x = (jnp.asarray(d[k]) for k in ("f", "p", "x"))
    back_of = jax.jit(jax.vmap(partial(flow_to_physical, problem)))
    flow_of = jax.jit(jax.vmap(partial(physical_to_flow, problem)))
    signal = jax.jit(jax.vmap(partial(mismatch, problem)))
    n = min(CHECK_SOURCES, p.shape[0])

    def errors(back, x_back) -> dict:
        back = np.asarray(back, dtype=np.float64)
        pn = np.asarray(p)
        sky = lambda q: np.stack(
            [
                np.cos(q[..., 4]) * np.cos(q[..., 3]),
                np.cos(q[..., 4]) * np.sin(q[..., 3]),
                np.sin(q[..., 4]),
            ],
            -1,
        )
        mis = np.asarray(signal(p[:n], jnp.asarray(back[:n]), f[:n]))
        return dict(
            f0=float(np.max(np.abs(back[..., 0] / pn[..., 0] - 1))),
            fdot=float(np.max(np.abs(back[..., 1] / pn[..., 1] - 1))),
            A=float(np.max(np.abs(back[..., 2] / pn[..., 2] - 1))),
            sky=float(np.max(np.linalg.norm(sky(back) - sky(pn), axis=-1))),
            sin_iota=float(np.max(np.abs(np.sin(back[..., 6]) - np.sin(pn[..., 6])))),
            z=float(
                np.max(np.abs(np.asarray(x_back)[..., 4:] - np.asarray(x)[..., 4:]))
            ),
            # the returned angles stand for the same signal only up to the four-fold choice
            phase_minus=float(
                np.max(
                    np.abs(
                        wrap(
                            (back[..., 7] - 2 * back[..., 5])
                            - (pn[..., 7] - 2 * pn[..., 5])
                        )
                    )
                )
            ),
            phase_plus=float(
                np.max(
                    np.abs(
                        wrap(
                            (back[..., 7] + 2 * back[..., 5])
                            - (pn[..., 7] + 2 * pn[..., 5])
                        )
                    )
                )
            ),
            signal_max=float(mis.max()),
            signal_median=float(np.median(mis)),
            signal_sources=int(n),
        )

    back64 = back_of(x, f)
    x32 = x.astype(jnp.float32)
    back32 = back_of(x32, f)
    out = dict(
        float64=errors(back64, flow_of(back64, f)),
        float32=errors(back32, flow_of(back32.astype(jnp.float64), f)),
        float32_dtype=str(back32.dtype),
    )

    # sources pushed onto and near the poles: |sin iota| = 1 - gap
    base = p[:n, :1]
    poles = []
    for sign in (1.0, -1.0):
        for gap in POLE_GAPS:
            q = base.at[..., 6].set(jnp.arcsin(sign * (1 - gap)))
            row = dict(sin_iota=sign * (1 - gap), gap=gap)
            for name, dtype in (("float64", jnp.float64), ("float32", jnp.float32)):
                back = back_of(flow_of(q, f[:n]).astype(dtype), f[:n]).astype(
                    jnp.float64
                )
                row[name] = dict(
                    sin_iota=float(
                        jnp.max(jnp.abs(jnp.sin(back[..., 6]) - jnp.sin(q[..., 6])))
                    ),
                    signal=float(jnp.max(signal(q, back, f[:n]))),
                )
            poles.append(row)
    out["poles"] = poles
    return out


def degeneracy(problem: LisaGB, d: dict) -> dict:
    """The four (psi, phi0) with one signal, the branch flow_to_physical returns, and the poles' surviving phase."""
    f, p, x = (jnp.asarray(d[k]) for k in ("f", "p", "x"))
    n = min(CHECK_SOURCES, p.shape[0])
    signal = jax.jit(jax.vmap(partial(mismatch, problem)))
    flow_of = jax.jit(jax.vmap(partial(physical_to_flow, problem)))
    base = p[:n, :1]

    def turned(q, d_psi, d_phi):
        return q.at[..., 5].add(d_psi).at[..., 7].add(d_phi)

    variants = []
    for k in range(4):
        q = turned(base, k * math.pi / 2, (k % 2) * math.pi)
        variants.append(
            dict(
                psi_step=k * math.pi / 2,
                phi0_step=(k % 2) * math.pi,
                signal=float(jnp.max(signal(base, q, f[:n]))),
                z=float(
                    jnp.max(jnp.abs(flow_of(q, f[:n]) - flow_of(base, f[:n]))[..., 4:])
                ),
            )
        )
    # a step that keeps neither phi±: a different signal, for scale
    wrong = float(jnp.median(signal(base, turned(base, math.pi / 4, 0.0), f[:n])))

    # which of the four the chart hands back
    back = np.asarray(jax.vmap(partial(flow_to_physical, problem))(x, f))
    pn = np.asarray(p)
    branch = (
        np.round(((back[..., 5] - pn[..., 5]) % TAU) / (math.pi / 2)).astype(int) % 4
    )
    phi_off = wrap(back[..., 7] - pn[..., 7] - (branch % 2) * math.pi)
    psi_centred = wrap(back[..., 5])

    # on a pole only one phase survives: sin iota = +1 keeps phi- = phi0 - 2 psi, -1 keeps phi+
    poles = {}
    for sign, name in ((1.0, "+1"), (-1.0, "-1")):
        q = base.at[..., 6].set(sign * math.pi / 2)
        keep = turned(q, SHIFT, 2 * sign * SHIFT)
        lose = turned(q, SHIFT, -2 * sign * SHIFT)
        poles[name] = dict(
            kept=float(jnp.max(signal(q, keep, f[:n]))),
            lost=float(jnp.median(signal(q, lose, f[:n]))),
        )
    return dict(
        variants=variants,
        wrong_step_median=wrong,
        sources=int(n),
        branch_counts=np.bincount(branch.ravel(), minlength=4).tolist(),
        branch_phi0_offset=float(np.max(np.abs(phi_off))),
        returned_psi=hist(back[..., 5].ravel(), 0, TAU, bins=96),
        returned_psi_centred=[float(psi_centred.min()), float(psi_centred.max())],
        returned_phi0=hist(back[..., 7].ravel(), 0, TAU, bins=96),
        pole_phase=poles,
        pole_shift=SHIFT,
    )


def phase_reference(problem: LisaGB, d: dict) -> dict:
    """What the data pins: the Fisher correlation of f0 with phi0 at the start against at mid-observation."""
    f, p = jnp.asarray(d["f"]), jnp.asarray(d["p"])
    n = min(CHECK_SOURCES, p.shape[0])
    t = problem.t_obs
    q = p[:n, 0]
    shift = jnp.pi * q[:, 0] * t + jnp.pi * q[:, 1] * t**2 / 4
    theta = jnp.stack(
        [
            q[:, 0] * t,
            jnp.log(q[:, 1]),
            q[:, 3],
            q[:, 4],
            jnp.log(q[:, 2]),
            jnp.sin(q[:, 6]),
            q[:, 5],
            q[:, 7],
        ],
        axis=-1,
    )
    out = {}
    for name, middle in (("start", False), ("middle", True)):
        th = theta.at[:, 7].add(-shift) if middle else theta
        fm = np.asarray(
            jax.jit(
                jax.vmap(
                    lambda a, b: fisher(
                        problem, partial(angles_to_p, problem, middle=middle), a, b
                    )
                )
            )(th, f[:n])
        )
        pairs = np.array([[*corr_of(m, 0, 7), *corr_of(m, 1, 7)] for m in fm])
        # std from the normalized inverse: fdot barely moves the narrow band's signal
        std_phi = np.sqrt(
            np.array([np.linalg.inv(normalized(m))[7, 7] / m[7, 7] for m in fm])
        )
        marginal = pairs[np.isfinite(pairs[:, 1]), 1]
        out[name] = dict(
            f0_phi_conditional=hist(pairs[:, 0], -1, 1),
            f0_phi_marginal=hist(marginal, -1, 1),
            f0_phi_conditional_median=float(np.median(pairs[:, 0])),
            f0_phi_marginal_median=float(np.nanmedian(pairs[:, 1])),
            fdot_phi_conditional_median=float(np.median(pairs[:, 2])),
            fdot_phi_marginal_median=float(np.nanmedian(pairs[:, 3])),
            singular=int(np.sum(~np.isfinite(pairs[:, 1]) | ~np.isfinite(std_phi))),
            phi_std_median=float(np.nanmedian(std_phi)),
            phi_std=std_phi.tolist(),
        )
    out["phi_std_ratio_median"] = float(
        np.nanmedian(
            np.array(out["start"]["phi_std"]) / np.array(out["middle"]["phi_std"])
        )
    )
    out["sources"] = int(n)

    # the shift itself and what float32 arithmetic would make of it
    f0 = np.asarray(p[..., 0]).ravel()
    fdot = np.asarray(p[..., 1]).ravel()
    s64 = math.pi * f0 * t + math.pi * fdot * t**2 / 4
    f32 = np.float32
    s32 = f32(math.pi) * f0.astype(f32) * f32(t) + f32(math.pi) * fdot.astype(
        f32
    ) * f32(t) ** 2 / f32(4)
    err32 = np.abs(wrap(s32.astype(np.float64) - s64))
    out["shift"] = dict(
        min=float(s64.min()),
        max=float(s64.max()),
        chirp_part_max=float(np.max(math.pi * fdot * t**2 / 4)),
        float32_error_median=float(np.median(err32)),
        float32_error_max=float(err32.max()),
    )
    return out


def singularity(problem: LisaGB, d: dict) -> dict:
    """The Fisher matrix of (ln A, sin iota, psi, phi0), of z± and of the flow's warped z± as sin iota nears ±1."""
    f = jnp.asarray(d["f"][0])
    p0 = jnp.asarray(d["p"][0, 0])
    x0 = physical_to_flow(problem, p0[None], f)[0]
    unit = amplitude_unit(problem, f)
    grid = np.sort(
        np.concatenate(
            [-(1 - np.array(FISHER_GAPS)), np.linspace(-0.95, 0.95, 39), 1 - np.array(FISHER_GAPS)]
        )
    )

    def angles(theta):
        log_a, s, psi, phi0 = theta
        return (
            p0.at[2]
            .set(jnp.exp(log_a))
            .at[5]
            .set(psi)
            .at[6]
            .set(jnp.arcsin(s))
            .at[7]
            .set(phi0)
        )

    def flow(theta):
        return flow_to_physical(problem, x0.at[4:].set(theta)[None], f)[0]

    def helicity(theta):
        re, im = theta[0::2], theta[1::2]
        r = jnp.hypot(re, im)
        warped = jnp.stack([re, im], -1) * (jnp.arcsinh(r) / r)[:, None]
        return flow(warped.ravel())

    rows = []
    z_fishers = []
    for s in grid:
        p = p0.at[6].set(jnp.arcsin(s))
        x = physical_to_flow(problem, p[None], f)[0]
        r = jnp.hypot(x[4::2], x[5::2])
        z = (jnp.stack([x[4::2], x[5::2]], -1) * (jnp.sinh(r) / r)[:, None]).ravel()
        fa = np.asarray(
            fisher(problem, angles, jnp.stack([jnp.log(p0[2]), s, p0[5], p0[7]]), f)
        )
        fz = np.asarray(fisher(problem, helicity, z, f))
        fx = np.asarray(fisher(problem, flow, x[4:], f))
        z_fishers.append(fz)
        rows.append(
            dict(
                sin_iota=float(s),
                psi_phi_marginal=corr_of(fa, 2, 3)[1],
                psi_phi_conditional=corr_of(fa, 2, 3)[0],
                cond_angles=condition(fa),
                cond_z=condition(fz),
                cond_flow=condition(fx),
            )
        )
    z_fishers = np.array(z_fishers)
    middle = z_fishers[np.argmin(np.abs(grid))]
    scale = np.abs(middle).max()

    # the signal is real-linear in z± at a fixed f0, fdot and sky
    zs = jnp.asarray(d["x"][:64, 1, 4:])
    zs = jnp.stack([zs[:, 0::2], zs[:, 1::2]], -1)
    rr = jnp.linalg.norm(zs, axis=-1, keepdims=True)
    zs = (zs * jnp.sinh(rr) / rr).reshape(-1, 4)
    za, zb = zs[:32], zs[32:]
    power = window_power(problem, f)
    h_of = jax.vmap(lambda z: clean_signal(problem, helicity(z)[None], f))
    norm = lambda h: jnp.sqrt(jnp.sum(jnp.abs(h) ** 2 / power, axis=(-2, -1)))
    ha, hb, hab = h_of(za), h_of(zb), h_of(za + zb)
    lam = 3.7
    return dict(
        source=dict(f0=float(p0[0]), fdot=float(p0[1]), unit=float(unit)),
        rows=rows,
        z_fisher_variation=float(np.abs(z_fishers - middle).max() / scale),
        z_fisher_variation_inner=float(np.abs(z_fishers[2:-2] - middle).max() / scale),
        linearity_sum=float(jnp.max(norm(hab - ha - hb) / (norm(ha) + norm(hb)))),
        linearity_scale=float(
            jnp.max(norm(h_of(lam * za) - lam * ha) / (lam * norm(ha)))
        ),
    )


def test_vectors(cfg) -> dict:
    """The chart at phase_reference='start' on a few sources, for the page's JS copy of it to check itself against."""
    problem = LisaGB(**{**cfg.problem, "phase_reference": "start"})
    key_f, key_p = jr.split(jr.key(1))
    f = sample_f(problem, key_f)
    p = jnp.concat(
        [
            sample_physical(problem, k, f)
            for k in jr.split(key_p, TEST_VECTORS // problem.n_sources)
        ]
    )
    p = p.at[:2, 6].set(jnp.arcsin(jnp.array([1 - 1e-4, -(1 - 1e-4)])))
    x = physical_to_flow(problem, p, f)
    back = flow_to_physical(problem, x, f)
    unit = amplitude_unit(problem, f)
    rows = []
    for pi, xi, bi in zip(np.asarray(p), np.asarray(x), np.asarray(back)):
        rows.append(
            dict(
                A=float(pi[2] / unit),
                sin_iota=float(np.sin(pi[6])),
                psi=float(pi[5]),
                phi0=float(pi[7]),
                z=[float(v) for v in xi[4:]],
                back=dict(
                    A=float(bi[2] / unit),
                    sin_iota=float(np.sin(bi[6])),
                    psi=float(bi[5]),
                    phi0=float(bi[7]),
                ),
            )
        )

    # h+ + i hx from a numpy copy of jaxgb's _get_gs amplitudes, against -2 (conj z- e^{iW} + z+ e^{-iW})
    omega = np.linspace(0, TAU, 64)
    worst = 0.0
    for pi, xi in zip(np.asarray(p), np.asarray(x)):
        amp, psi, iota, phi0 = pi[2], pi[5], pi[6], pi[7]
        cosiota = np.cos(np.pi / 2 - iota)
        aplus = -amp * (1 + cosiota**2)
        across = -2 * amp * cosiota
        plus = aplus * np.cos(2 * psi) + 1j * across * np.sin(2 * psi)
        cross = aplus * np.sin(2 * psi) - 1j * across * np.cos(2 * psi)
        carrier = np.exp(1j * (omega - phi0))
        w = (plus * carrier).real + 1j * (cross * carrier).real
        r = np.hypot(xi[4::2], xi[5::2])
        z = np.sinh(r) * np.exp(1j * np.arctan2(xi[5::2], xi[4::2])) * float(unit)
        model = -2 * (np.conj(z[1]) * np.exp(1j * omega) + z[0] * np.exp(-1j * omega))
        worst = max(worst, float(np.max(np.abs(w - model)) / amp))
    return dict(rows=rows, ellipse_identity=worst)


def main(out: str, windows: int) -> None:
    device = jax.devices()[0]
    print(f"device: {device.device_kind}  x64: {jax.config.jax_enable_x64}", flush=True)
    result = dict(device=device.device_kind, windows=windows, bands={})
    for band in BANDS:
        cfg = config.load(f"{band}-B")
        problem = LisaGB(**cfg.problem)
        start = time.perf_counter()
        d = jax.lax.map(
            partial(draw, problem), jr.split(jr.key(0), windows), batch_size=256
        )
        d = {k: np.asarray(v) for k, v in jax.block_until_ready(d).items()}
        seconds = {"draw": time.perf_counter() - start}
        b = dict(
            band=band,
            windows=windows,
            sources=int(d["snr"].size),
            problem=dict(
                n_sources=problem.n_sources,
                t_obs=problem.t_obs,
                phase_reference=problem.phase_reference,
            ),
        )
        for name, fn in (
            ("orientation", lambda: orientation(problem, d)),
            ("round_trip", lambda: round_trip(problem, d)),
            ("degeneracy", lambda: degeneracy(problem, d)),
            ("phase", lambda: phase_reference(problem, d)),
            ("singularity", lambda: singularity(problem, d)),
            ("test_vectors", lambda: test_vectors(cfg)),
        ):
            start = time.perf_counter()
            part = fn()
            seconds[name] = time.perf_counter() - start
            if name == "orientation":
                b.update(part)
            else:
                b[name] = part
            print(f"{band}: {name} in {seconds[name]:.1f} s", flush=True)
        b["seconds"] = seconds
        rt, ph, dg = b["round_trip"], b["phase"], b["degeneracy"]
        print(
            f"{band}: z std {[round(z['std'], 3) for z in b['flow']]};"
            f" round trip f64 signal {rt['float64']['signal_max']:.2e}, f32 {rt['float32']['signal_max']:.2e};"
            f" corr(f0, phi) start {ph['start']['f0_phi_conditional_median']:.3f}"
            f" middle {ph['middle']['f0_phi_conditional_median']:.3f};"
            f" degeneracy {max(v['signal'] for v in dg['variants']):.1e}",
            flush=True,
        )
        result["bands"][band] = b
    with open(out, "w") as fh:
        json.dump(result, fh)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else WINDOWS)
