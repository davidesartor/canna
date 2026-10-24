"""Draw LisaGB's prior for each band and summarise it for the prior pages: parameters, sky, snrs, chart coordinates.

Also per band: the windows and sources one run of total_steps draws, and the sky-averaged snr
constant C = <rho^2> S_n(f0) / (A^2 T) of a monochromatic source across f0. Once: the galactic
sky prior's density in the galactic, ecliptic and equatorial frames.

Run from the repo root (JAX: a job, not the login node):
    uv run --frozen python reports/prior/scripts/draw_prior.py reports/prior/data/prior.json
"""

import base64
import json
import math
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from einops import rearrange

from canna.lisa import config
from canna.lisa.chart import GALACTIC_SCALE, physical_to_flow
from canna.lisa.config import Config
from canna.lisa.network import COORD_RANGES
from canna.lisa.physics import (
    BULGE_SCALES,
    SKY_WEIGHTS,
    THICK_DISK_SCALES,
    THIN_DISK_SCALES,
    chirp_mass_log_density,
    chirp_mass_support,
    equatorial_to_galactic,
    fdot_from_chirp_mass,
    galactic_to_equatorial,
    sample_chirp_mass,
    sample_sky,
    sky_log_density,
)
from canna.lisa.problem import LisaGB, amplitude_unit, noise_psd, sample_f
from canna.lisa.simulator import sample_physical, snr, source_snrs

BANDS = ("narrow", "full")
WINDOWS = 8192
SCATTER = 3000
BINS = 48
QUANTILES = [0.01, 0.16, 0.5, 0.84, 0.99]
FLOW_NAMES = ["f0", "Mc", "l", "b", "Re z+", "Im z+", "Re z-", "Im z-"]
SKY_SOURCES = 8192
SKY_CHUNK = 256
SKY_GRID = 16
SKY_FLAT_BELOW = 1.5e-3  # [Hz]
MONOCHROMATIC_MC = 1e-3  # [Msun]: fdot T^2 below 1e-4 bins across the band
MAP_SIZE = (480, 240)
QUADRATURE_SIZE = (2880, 1440)
GC_RADII = (0.1, 0.2, 0.3, 0.5)  # [rad]
MC_CHECK_DRAWS = 2_000_000
MC_CHECK_BINS = 60
NOISE_F = np.geomspace(10**-4.3, 3e-2, 240)  # [Hz]

OBLIQUITY = math.radians(23.4392911)
ECLIPTIC_TO_EQUATORIAL = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, math.cos(OBLIQUITY), -math.sin(OBLIQUITY)],
        [0.0, math.sin(OBLIQUITY), math.cos(OBLIQUITY)],
    ]
)
# each frame's longitude range, the longitude at the centre of its map, and its latitude histogram [deg]
FRAMES = dict(
    galactic=dict(lon=(-math.pi, math.pi), lon0=0.0, lat_hist=(-30, 30, 60)),
    ecliptic=dict(lon=(0.0, 2 * math.pi), lon0=math.pi, lat_hist=(-90, 90, 72)),
    equatorial=dict(lon=(0.0, 2 * math.pi), lon0=math.pi, lat_hist=(-90, 90, 72)),
)


def rotate(
    matrix: np.ndarray, lon: np.ndarray, lat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    v = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    x, y, z = np.tensordot(matrix, v, axes=1)
    return np.arctan2(y, x), np.arctan2(z, np.hypot(x, y))


def to_equatorial(
    frame: str, lon: np.ndarray, lat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if frame == "galactic":
        return tuple(
            np.asarray(c)
            for c in galactic_to_equatorial(jnp.asarray(lon), jnp.asarray(lat))
        )
    if frame == "ecliptic":
        return rotate(ECLIPTIC_TO_EQUATORIAL, lon, lat)
    return lon, lat


def from_equatorial(
    frame: str, ra: np.ndarray, dec: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(RA, Dec) to the frame's (lon, lat), lon in the frame's range."""
    if frame == "galactic":
        lon, lat = (
            np.asarray(c)
            for c in equatorial_to_galactic(jnp.asarray(ra), jnp.asarray(dec))
        )
    elif frame == "ecliptic":
        lon, lat = rotate(ECLIPTIC_TO_EQUATORIAL.T, ra, dec)
    else:
        lon, lat = ra, dec
    lo = FRAMES[frame]["lon"][0]
    return np.mod(lon - lo, 2 * math.pi) + lo, lat


def log_density_per_steradian(
    frame: str, lon: np.ndarray, lat: np.ndarray
) -> np.ndarray:
    """log p_Omega at the frame's (lon, lat), through canna's sky_log_density on (RA, Dec)."""
    ra, dec = to_equatorial(frame, lon, lat)
    log_p = jax.jit(sky_log_density)(jnp.asarray(ra), jnp.asarray(dec))
    return np.asarray(log_p) - np.log(np.cos(dec))


def sky_frames() -> dict:
    """Per frame: the map of log10 p_Omega, the galactic centre, and the lon/lat CDFs on a quadrature grid."""
    out, cdfs = {}, {}
    for frame, spec in FRAMES.items():
        # map cells: column j at plot longitude -pi + (j + 1/2) 2 pi / W, increasing to the left; row i from the north
        w, h = MAP_SIZE
        plot_lon = -math.pi + (np.arange(w) + 0.5) * 2 * math.pi / w
        lat = math.pi / 2 - (np.arange(h) + 0.5) * math.pi / h
        lon_mesh, lat_mesh = np.meshgrid(spec["lon0"] - plot_lon, lat)
        log10_map = log_density_per_steradian(frame, lon_mesh, lat_mesh) / math.log(10)

        # midpoint quadrature of p(lon, lat) = p_Omega cos(lat): its integral and both marginal cdfs
        nw, nh = QUADRATURE_SIZE
        lon_edges = np.linspace(*spec["lon"], nw + 1)
        lat_edges = np.linspace(-math.pi / 2, math.pi / 2, nh + 1)
        lon_mesh, lat_mesh = np.meshgrid(
            (lon_edges[1:] + lon_edges[:-1]) / 2,
            (lat_edges[1:] + lat_edges[:-1]) / 2,
            indexing="ij",
        )
        cell = np.exp(log_density_per_steradian(frame, lon_mesh, lat_mesh)) * np.cos(
            lat_mesh
        )
        cell = cell * (lon_edges[1] - lon_edges[0]) * (lat_edges[1] - lat_edges[0])
        cdfs[frame] = dict(
            lon=(lon_edges, np.concatenate([[0], np.cumsum(cell.sum(1))])),
            lat=(lat_edges, np.concatenate([[0], np.cumsum(cell.sum(0))])),
        )
        centre = from_equatorial(
            frame, *to_equatorial("galactic", np.zeros(()), np.zeros(()))
        )
        out[frame] = dict(
            log10_map=log10_map,
            lon0=spec["lon0"],
            centre=[float(c) for c in centre],
            integral=float(cell.sum()),
        )

    # one shared colour scale, quantised to a byte per cell
    lo = min(f["log10_map"].min() for f in out.values())
    hi = max(f["log10_map"].max() for f in out.values())
    for f in out.values():
        q = np.round((f.pop("log10_map") - lo) / (hi - lo) * 255).astype(np.uint8)
        f["map"] = base64.b64encode(q.tobytes()).decode()
    sky = dict(
        frames=out,
        map=dict(
            width=MAP_SIZE[0],
            height=MAP_SIZE[1],
            log10_lo=float(lo),
            log10_hi=float(hi),
        ),
        components=dict(
            names=["bulge", "thick disk", "thin disk", "isotropic floor"],
            weights=SKY_WEIGHTS.tolist(),
            scales=[
                list(BULGE_SCALES),
                list(THICK_DISK_SCALES),
                list(THIN_DISK_SCALES),
                None,
            ],
        ),
        galactic_scale=list(GALACTIC_SCALE),
    )
    return sky, cdfs


def draw(problem: LisaGB, key: jax.Array) -> dict:
    """One window of the prior: its f, its sources and every derived quantity the page shows."""
    key_f, key_p = jr.split(key)
    f = sample_f(problem, key_f)
    p = sample_physical(problem, key_p, f)
    rho = source_snrs(problem, p, f)
    return dict(
        f=f,
        p=p,
        snr=rho,
        window_snr=snr(problem, p, f),
        unit=amplitude_unit(problem, f),
        x=physical_to_flow(problem, p, f),
    )


def dataset(cfg: Config, problem: LisaGB) -> dict:
    """What one run of the band's total_steps draws: fresh windows, their sources and the (x0, t) samples."""
    windows_per_step = cfg.batch_size // cfg.draws
    windows = cfg.total_steps * windows_per_step
    return dict(
        total_steps=cfg.total_steps,
        batch_size=cfg.batch_size,
        draws=cfg.draws,
        windows_per_step=windows_per_step,
        windows=windows,
        sources=windows * problem.n_sources,
        samples=cfg.total_steps * cfg.batch_size,
    )


def sky_average(problem: LisaGB) -> dict:
    """C(f0) = <rho^2> at A = amplitude_unit(f0), the same orientations at every f0, under an isotropic and the galactic sky."""
    keys = jr.split(jr.key(1), 6)
    n = SKY_SOURCES
    galactic = jax.vmap(sample_sky)(jr.split(keys[2], n))
    skies = dict(
        isotropic=(
            jr.uniform(keys[0], (n,), maxval=2 * jnp.pi),
            jnp.arcsin(jr.uniform(keys[1], (n,), minval=-1.0, maxval=1.0)),
        ),
        galactic=(galactic[:, 0], galactic[:, 1]),
    )
    psi = jr.uniform(keys[3], (n,), maxval=2 * jnp.pi)
    iota = jnp.arcsin(jr.uniform(keys[4], (n,), minval=-1.0, maxval=1.0))
    phi0 = jr.uniform(keys[5], (n,), maxval=2 * jnp.pi)
    grid = np.geomspace(*problem.f0_range, SKY_GRID)

    # each source alone in the window centred on it, so rho^2 = C(f0) for that orientation
    @jax.jit
    def kappa_squared(f0, lon, lat):
        ones = jnp.ones(n)
        unit = amplitude_unit(problem, f0)
        p = jnp.stack(
            [
                f0 * ones,
                MONOCHROMATIC_MC * ones,
                unit * ones,
                lon,
                lat,
                psi,
                iota,
                phi0,
            ],
            axis=-1,
        )
        chunks = rearrange(p, "(c b) d -> c b d", b=SKY_CHUNK)
        rho = jax.lax.map(partial(source_snrs, problem, f=f0), chunks)
        return rho.ravel() ** 2

    out = dict(
        f0=grid.tolist(),
        sources=n,
        flat_below=SKY_FLAT_BELOW,
        monochromatic_mc=MONOCHROMATIC_MC,
    )
    for name, (lon, lat) in skies.items():
        c = np.stack(
            [np.asarray(kappa_squared(jnp.asarray(f0), lon, lat)) for f0 in grid]
        )
        low = c[grid < SKY_FLAT_BELOW].mean(0)
        c_low = low.mean()
        ratio = c.mean(1) / c_low
        # paired with the low-f0 mean over the same orientations, so the ratio's error is the drift's alone
        out[name] = dict(
            C=c.mean(1).tolist(),
            C_se=(c.std(1) / math.sqrt(n)).tolist(),
            C_low=float(c_low),
            C_low_se=float(low.std() / math.sqrt(n)),
            ratio=ratio.tolist(),
            ratio_se=(
                (c - ratio[:, None] * low).std(1) / math.sqrt(n) / c_low
            ).tolist(),
        )
    return out


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


def chirp_mass_check(problem: LisaGB) -> dict:
    """sample_chirp_mass against chirp_mass_log_density, the MCMC's quadrature prior: counts in ln Mc and their chi^2."""
    draws = jax.lax.map(
        partial(sample_chirp_mass, mass_range=problem.mass_range),
        jr.split(jr.key(2), MC_CHECK_DRAWS),
        batch_size=65536,
    )
    grid, log_p = chirp_mass_log_density(problem.mass_range)
    p = np.exp(log_p)
    cdf = np.concatenate([[0.0], np.cumsum((p[1:] + p[:-1]) / 2 * np.diff(grid))])
    out = hist(
        np.log(np.asarray(draws)[:, 0]),
        grid[0],
        grid[-1],
        bins=MC_CHECK_BINS,
        cdf=lambda e: np.interp(e, grid, cdf / cdf[-1]),
    )
    observed, expected = np.array(out["counts"]), np.array(out["reference"])
    used = expected > 0
    chi2 = float(((observed - expected)[used] ** 2 / expected[used]).sum())
    dof = int(used.sum()) - 1
    return dict(
        draws=MC_CHECK_DRAWS,
        bins=MC_CHECK_BINS,
        chi2=chi2,
        dof=dof,
        chi2_per_dof=chi2 / dof,
        quadrature_integral=float(cdf[-1]),
        **out,
    )


def spread(values: np.ndarray) -> dict:
    return dict(
        mean=float(values.mean()),
        std=float(values.std()),
        quantiles=np.quantile(values, QUANTILES).tolist(),
    )


def summarise_sky(lon: np.ndarray, lat: np.ndarray, cdfs: dict) -> dict:
    """The drawn sky per frame against the quadrature marginals, and how much of it sits near the plane and the centre."""
    marginals, near_plane = {}, {}
    for frame, spec in FRAMES.items():
        frame_lon, frame_lat = from_equatorial(frame, lon, lat)
        lon_deg = np.rad2deg(spec["lon"])
        (lon_edges, lon_cdf), (lat_edges, lat_cdf) = (
            cdfs[frame]["lon"],
            cdfs[frame]["lat"],
        )
        lat_lo, lat_hi, lat_bins = spec["lat_hist"]
        marginals[frame] = dict(
            lon=hist(
                np.rad2deg(frame_lon),
                *lon_deg,
                bins=72,
                cdf=lambda e: np.interp(np.deg2rad(e), lon_edges, lon_cdf),
            ),
            lat=hist(
                np.rad2deg(frame_lat),
                lat_lo,
                lat_hi,
                bins=lat_bins,
                cdf=lambda e: np.interp(np.deg2rad(e), lat_edges, lat_cdf),
            ),
        )
        near_plane[frame] = float(np.mean(np.abs(frame_lat) < math.radians(10)))
    l, b = from_equatorial("galactic", lon, lat)
    from_centre = np.arccos(np.clip(np.cos(b) * np.cos(l), -1, 1))
    return dict(
        marginals=marginals,
        abs_lat_below_10deg=near_plane,
        within_rad_of_centre={
            str(r): float(np.mean(from_centre < r)) for r in GC_RADII
        },
    )


def summarise(
    name: str, cfg: Config, problem: LisaGB, d: dict, seconds: float, cdfs: dict
) -> dict:
    f0, mc, amp, lon, lat, psi, iota, phi0 = np.moveaxis(d["p"], -1, 0)
    rho, unit, x = d["snr"], d["unit"][:, None], d["x"]
    s = problem.n_sources
    # snr of a unit-amplitude source, in amplitude_unit: the O(1) sky and orientation factor
    kappa = rho / amp * unit
    fdot = np.asarray(fdot_from_chirp_mass(mc, f0))
    log_snr = np.log10(problem.snr_range)
    flat = lambda lo, hi: lambda v: np.full_like(v, 1 / (hi - lo))
    log_mc_grid, log_mc_density = chirp_mass_log_density(problem.mass_range)
    mc_lo, mc_hi = chirp_mass_support(problem.mass_range)
    # p(log10 Mc) = p(ln Mc) ln 10
    mc_reference = lambda v: np.interp(
        v * math.log(10), log_mc_grid, np.exp(log_mc_density), left=0, right=0
    ) * math.log(10)
    log_f0 = np.log10(problem.f0_range)

    physical = dict(
        frequency={
            "log10 f0 [Hz]": hist(np.log10(f0).ravel(), *log_f0, flat(*log_f0)),
            "log10 Mc [Msun]": hist(
                np.log10(mc).ravel(), math.log10(mc_lo), math.log10(mc_hi), mc_reference
            ),
            "log10 fdot [Hz/s]": hist(
                np.log10(fdot).ravel(), *np.quantile(np.log10(fdot), [0, 1])
            ),
        },
        amplitude={
            "log10 A": hist(np.log10(amp).ravel(), *np.quantile(np.log10(amp), [0, 1])),
        },
        orientation={
            "psi [rad]": hist(psi.ravel(), 0, 2 * math.pi, flat(0, 2 * math.pi)),
            "sin iota": hist(np.sin(iota).ravel(), -1, 1, flat(-1, 1)),
            "phi0 [rad]": hist(phi0.ravel(), 0, 2 * math.pi, flat(0, 2 * math.pi)),
        },
    )
    table = {
        k: spread(v)
        for k, v in {
            "f0 [mHz]": 1e3 * f0,
            "Mc [Msun]": mc,
            "A": amp,
            "fdot [Hz/s]": fdot,
            "snr": rho,
            "window snr": d["window_snr"],
            "kappa": kappa,
        }.items()
    }

    # the flow sees these: f0 is uniform on [-1, 1], the sky the galactic marginals over GALACTIC_SCALE; the rest have no closed form
    (l_edges, l_cdf), (b_edges, b_cdf) = (
        cdfs["galactic"]["lon"],
        cdfs["galactic"]["lat"],
    )
    flow_cdfs = [
        None,
        None,
        lambda e: np.interp(e * GALACTIC_SCALE[0], l_edges, l_cdf),
        lambda e: np.interp(e * GALACTIC_SCALE[1], b_edges, b_cdf),
    ] + [None] * 4
    flow = []
    for j, n in enumerate(FLOW_NAMES):
        v = x[..., j].ravel()
        lo, hi = (
            (-1.05, 1.05)
            if j == 0
            else (
                (-4.5, 4.5)
                if j == 1
                else (-6.0, 6.0) if j in (2, 3) else tuple(np.quantile(v, [0, 1]))
            )
        )
        reach_lo, reach_hi = COORD_RANGES[j]
        flow.append(
            dict(
                name=n,
                hist=hist(v, lo, hi, flat(-1, 1) if j == 0 else None, cdf=flow_cdfs[j]),
                beyond_reach=float(np.mean((v < reach_lo) | (v > reach_hi))),
                reach=[reach_lo, reach_hi],
                **spread(v),
            )
        )
    radius = np.hypot(x[..., 4::2], x[..., 5::2])
    corr = np.corrcoef(x.reshape(-1, 8), rowvar=False)

    # the amplitude band a sky-averaged source of snr in snr_range would sit in, against f0
    grid = np.geomspace(*problem.f0_range, 160)
    curve = np.asarray(jax.vmap(partial(amplitude_unit, problem))(jnp.asarray(grid)))

    rng = np.random.default_rng(0)
    pick = rng.choice(rho.size, size=min(SCATTER, rho.size), replace=False)
    rank = np.broadcast_to(np.arange(s), rho.shape)
    gal_l, gal_b = from_equatorial("galactic", lon, lat)
    ecl_lon, ecl_lat = from_equatorial("ecliptic", lon, lat)
    scatter = {
        k: np.asarray(v).ravel()[pick].tolist()
        for k, v in dict(
            f0=f0,
            A=amp,
            snr=rho,
            kappa=kappa,
            sin_iota=np.sin(iota),
            psi=psi,
            lon=lon,
            lat=lat,
            gal_l=gal_l,
            gal_b=gal_b,
            ecl_lon=ecl_lon,
            ecl_lat=ecl_lat,
            mc=mc,
            rank=rank,
        ).items()
    }

    return dict(
        band=name,
        windows=int(rho.shape[0]),
        sources=int(rho.size),
        seconds=seconds,
        problem=dict(
            n_sources=s,
            t_obs=problem.t_obs,
            response_points=problem.response_points,
            window_bins=problem.window_bins,
            f0_range=list(problem.f0_range),
            snr_range=list(problem.snr_range),
            mass_range=list(problem.mass_range),
            phase_reference=problem.phase_reference,
        ),
        snr_hist=hist(np.log10(rho).ravel(), *log_snr, flat(*log_snr)),
        snr_rank=[hist(np.log10(rho[:, r]), *log_snr) for r in range(s)],
        window_snr_hist=hist(
            np.log10(d["window_snr"]), log_snr[0], math.log10(2 * problem.snr_range[1])
        ),
        sorted_ok=bool(np.all(np.diff(rho, axis=-1) <= 1e-6 * rho[:, 1:])),
        in_range=float(
            np.mean(
                (rho >= problem.snr_range[0] * (1 - 1e-6))
                & (rho <= problem.snr_range[1] * (1 + 1e-6))
            )
        ),
        kappa_hist=hist(np.log10(kappa).ravel(), *np.quantile(np.log10(kappa), [0, 1])),
        physical=physical,
        sky=summarise_sky(lon.ravel(), lat.ravel(), cdfs),
        table=table,
        flow=flow,
        radius=[
            dict(
                name=n,
                hist=hist(radius[..., k].ravel(), 0, float(radius.max())),
                **spread(radius[..., k].ravel()),
            )
            for k, n in enumerate(["|z+|", "|z-|"])
        ],
        corr=corr.tolist(),
        amplitude_band=dict(
            f0=grid.tolist(),
            lo=(curve * problem.snr_range[0]).tolist(),
            hi=(curve * problem.snr_range[1]).tolist(),
        ),
        noise=dict(
            f=NOISE_F.tolist(),
            psd=np.asarray(noise_psd(problem, jnp.asarray(NOISE_F))).tolist(),
        ),
        scatter=scatter,
        dataset=dataset(cfg, problem),
        # <kappa^2> under the whole training prior: C at the window centre, over the prior's sky
        C_prior=float(np.mean(kappa**2)),
        C_prior_se=float(np.std(kappa**2) / math.sqrt(kappa.size)),
    )


def main(out: str) -> None:
    device = jax.devices()[0]
    print(f"device: {device.device_kind}  x64: {jax.config.jax_enable_x64}", flush=True)
    start = time.perf_counter()
    sky, cdfs = sky_frames()
    integrals = ", ".join(f"{k} {v['integral']:.6f}" for k, v in sky["frames"].items())
    print(
        f"sky: integrals {integrals}; galactic scale {GALACTIC_SCALE[0]:.4f}, {GALACTIC_SCALE[1]:.4f}"
        f" ({time.perf_counter() - start:.0f} s)",
        flush=True,
    )
    result = dict(device=device.device_kind, windows=WINDOWS, sky=sky, bands={})
    for band in BANDS:
        cfg = config.load(f"{band}-B")
        problem = LisaGB(**cfg.problem)
        start = time.perf_counter()
        d = jax.lax.map(
            partial(draw, problem), jr.split(jr.key(0), WINDOWS), batch_size=256
        )
        d = {k: np.asarray(v) for k, v in jax.block_until_ready(d).items()}
        seconds = time.perf_counter() - start
        result["bands"][band] = summarise(band, cfg, problem, d, seconds, cdfs)
        b = result["bands"][band]
        t, near = b["table"], b["sky"]["within_rad_of_centre"]
        beyond = ", ".join(
            f"{f['name']} {100 * f['beyond_reach']:.2f}%" for f in b["flow"][2:4]
        )
        print(
            f"{band}: {WINDOWS} windows in {seconds:.1f} s; snr median {t['snr']['quantiles'][2]:.1f},"
            f" kappa 16-84% {t['kappa']['quantiles'][1]:.2f}-{t['kappa']['quantiles'][3]:.2f};"
            f" within 0.1/0.5 rad of the centre {near['0.1']:.3f}/{near['0.5']:.3f};"
            f" beyond the embedding reach {beyond}",
            flush=True,
        )
        start = time.perf_counter()
        avg = sky_average(problem)
        avg["seconds"] = time.perf_counter() - start
        b["sky_average"] = avg
        b["chirp_mass_check"] = check = chirp_mass_check(problem)
        print(
            f"{band}: {check['draws']} chirp masses against the quadrature density,"
            f" chi2/dof {check['chi2']:.1f}/{check['dof']} = {check['chi2_per_dof']:.2f}"
            f" over {check['bins']} bins in ln Mc, {check['outside']} outside",
            flush=True,
        )
        n = b["dataset"]
        iso = avg["isotropic"]
        print(
            f"{band}: {n['windows']:.3g} windows, {n['sources']:.3g} sources per run;"
            f" C {iso['C_low']:.4f} +- {iso['C_low_se']:.4f} below {SKY_FLAT_BELOW * 1e3} mHz,"
            f" {iso['ratio'][-1]:.4f} of it at {avg['f0'][-1] * 1e3:.2f} mHz,"
            f" galactic sky {avg['galactic']['C_low']:.4f}, training prior {b['C_prior']:.4f}"
            f" ({avg['seconds']:.0f} s)",
            flush=True,
        )
    with open(out, "w") as fh:
        json.dump(result, fh)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
