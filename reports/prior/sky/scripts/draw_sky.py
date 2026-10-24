"""Draw the galactic sky prior and check it for the sky page: per component, in three frames, against the chart.

Run from the repo root (JAX: a job, not the login node):
    uv run --frozen python reports/prior/sky/scripts/draw_sky.py reports/prior/sky/data/sky.json
SKY_DRAWS and SKY_WINDOWS shrink the run for a smoke test.
"""

import base64
import json
import math
import os
import sys
import time

import astropy.units as u
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from astropy.coordinates import BarycentricMeanEcliptic, SkyCoord

from canna.lisa import config
from canna.lisa.chart import GALACTIC_SCALE, physical_to_flow
from canna.lisa.physics import (
    BULGE_SCALES,
    EQUATORIAL_TO_GALACTIC,
    SKY_CATALOGUE_SOURCES,
    SKY_WEIGHTS,
    THICK_DISK_SCALES,
    THIN_DISK_SCALES,
    equatorial_to_galactic,
    galactic_scales,
    galactic_to_equatorial,
    sample_sky,
    sky_log_density,
)
from canna.lisa.problem import LisaGB, sample_f
from canna.lisa.simulator import sample_physical

BANDS = ("narrow", "full")
DRAWS = int(os.environ.get("SKY_DRAWS", 2**20))
WINDOWS = int(os.environ.get("SKY_WINDOWS", 8192))
ASTROPY_DRAWS = 100_000
BINS = 48
QUANTILES = [0.01, 0.16, 0.5, 0.84, 0.99]
FLOW_NAMES = ["f0", "fdot", "l", "b", "Re z+", "Im z+", "Re z-", "Im z-"]
MAP_SIZE = (480, 240)
QUADRATURE_SIZE = (2880, 1440)
GC_RADII = (0.1, 0.2, 0.3, 0.5)  # [rad]
WRAP_DELTAS = (0.05, 0.1, 0.25, 0.5)  # [rad] from l = ±pi
CENTRE_STEP = 0.005  # [rad] of the from-centre cdf grid
NAMES = ["bulge", "thick disk", "thin disk", "isotropic floor"]
PARTS = ["sum"] + NAMES

OBLIQUITY = math.radians(23.4392911)
ECLIPTIC_TO_EQUATORIAL = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, math.cos(OBLIQUITY), -math.sin(OBLIQUITY)],
        [0.0, math.sin(OBLIQUITY), math.cos(OBLIQUITY)],
    ]
)
# Hipparcos (ESA 1997, vol. 1 §1.5.3): the north galactic pole and the galactic centre in ICRS [deg]
HIPPARCOS_NGP = (192.85948, 27.12825)
HIPPARCOS_CENTRE = (266.40500, -28.93617)
# each frame's longitude range, the longitude at the centre of its map, and its latitude histogram [deg]
FRAMES = dict(
    galactic=dict(lon=(-math.pi, math.pi), lon0=0.0, lat_hist=(-30, 30, 60)),
    ecliptic=dict(lon=(0.0, 2 * math.pi), lon0=math.pi, lat_hist=(-90, 90, 72)),
    equatorial=dict(lon=(0.0, 2 * math.pi), lon0=math.pi, lat_hist=(-90, 90, 72)),
)


def unit_vector(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def angle_between(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Angle between unit vectors along axis 0, from the chord: exact near zero, unlike arccos."""
    return 2 * np.arcsin(np.clip(np.linalg.norm(v - w, axis=0) / 2, 0, 1))


def rotate(
    matrix: np.ndarray, lon: np.ndarray, lat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = np.tensordot(matrix, unit_vector(lon, lat), axes=1)
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


def component_densities(frame: str, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Each component's weighted density per steradian, w_k p_k, at the frame's (lon, lat): physics' mixture term by term."""
    l, b = from_equatorial("galactic", *to_equatorial(frame, lon, lat))
    x = np.stack([l, np.sin(b)], axis=-1)
    half_width = np.array([math.pi, 1.0])
    scales = np.array(BULGE_SCALES)
    bulge = (np.exp(-0.5 * (x / scales) ** 2) / (math.sqrt(2 * math.pi) * scales)).prod(
        -1
    )
    thick, thin = (
        (np.exp(-np.abs(x) / s) / (2 * s * (1 - np.exp(-half_width / s)))).prod(-1)
        for s in (np.array(THICK_DISK_SCALES), np.array(THIN_DISK_SCALES))
    )
    floor = np.full_like(l, 1 / (4 * math.pi))
    return SKY_WEIGHTS.reshape(4, *[1] * l.ndim) * np.stack([bulge, thick, thin, floor])


def sky_frames() -> tuple[dict, dict, dict]:
    """Per frame: maps of log10 p_Omega (sum and components), the galactic centre, and per-part quadrature cdfs."""
    out, cdfs, checks = {}, {}, {}
    maps = {}
    for frame, spec in FRAMES.items():
        # map cells: column j at plot longitude -pi + (j + 1/2) 2 pi / W, increasing to the left; row i from the north
        w, h = MAP_SIZE
        plot_lon = -math.pi + (np.arange(w) + 0.5) * 2 * math.pi / w
        lat = math.pi / 2 - (np.arange(h) + 0.5) * math.pi / h
        lon_mesh, lat_mesh = np.meshgrid(spec["lon0"] - plot_lon, lat)
        log_sum = log_density_per_steradian(frame, lon_mesh, lat_mesh)
        parts = component_densities(frame, lon_mesh, lat_mesh)
        with np.errstate(divide="ignore"):
            maps[frame] = dict(
                sum=log_sum / math.log(10),
                **{n: np.log10(parts[k]) for k, n in enumerate(NAMES[:3])},
            )
        map_err = float(np.max(np.abs(np.log(parts.sum(0)) - log_sum)))

        # midpoint quadrature of p(lon, lat) = p_Omega cos(lat): integrals and marginal cdfs, canna's sum and each part
        nw, nh = QUADRATURE_SIZE
        lon_edges = np.linspace(*spec["lon"], nw + 1)
        lat_edges = np.linspace(-math.pi / 2, math.pi / 2, nh + 1)
        lon_mesh, lat_mesh = np.meshgrid(
            (lon_edges[1:] + lon_edges[:-1]) / 2,
            (lat_edges[1:] + lat_edges[:-1]) / 2,
            indexing="ij",
        )
        area = (
            np.cos(lat_mesh)
            * (lon_edges[1] - lon_edges[0])
            * (lat_edges[1] - lat_edges[0])
        )
        log_sum = log_density_per_steradian(frame, lon_mesh, lat_mesh)
        parts = component_densities(frame, lon_mesh, lat_mesh)
        quadrature_err = float(np.max(np.abs(np.log(parts.sum(0)) - log_sum)))
        cells = dict(
            sum=np.exp(log_sum) * area,
            **{n: parts[k] * area for k, n in enumerate(NAMES)},
        )
        cdfs[frame] = {
            part: dict(
                lon=(lon_edges, np.concatenate([[0], np.cumsum(cell.sum(1))])),
                lat=(lat_edges, np.concatenate([[0], np.cumsum(cell.sum(0))])),
            )
            for part, cell in cells.items()
        }
        if frame == "galactic":
            # the angle from the galactic centre of every quadrature cell, binned into cdfs per part
            from_centre = np.arccos(np.clip(np.cos(lat_mesh) * np.cos(lon_mesh), -1, 1))
            edges = np.arange(0, math.pi + CENTRE_STEP, CENTRE_STEP)
            cdfs["from_centre"] = dict(
                edges=edges,
                **{
                    part: np.concatenate(
                        [
                            [0],
                            np.cumsum(
                                np.histogram(from_centre, edges, weights=cell)[0]
                            ),
                        ]
                    )
                    for part, cell in cells.items()
                },
            )
        centre = from_equatorial(
            frame, *to_equatorial("galactic", np.zeros(()), np.zeros(()))
        )
        # the other frames' equators traced in this frame: the galactic plane, the ecliptic, the celestial equator
        trace = np.linspace(0, 2 * math.pi, 361)
        planes = {
            other: np.stack(
                from_equatorial(
                    frame, *to_equatorial(other, trace, np.zeros_like(trace))
                )
            ).T.tolist()
            for other in FRAMES
            if other != frame
        }
        out[frame] = dict(
            lon0=spec["lon0"],
            centre=[float(c) for c in centre],
            planes=planes,
            integral=float(cells["sum"].sum()),
            component_integrals={
                n: float(cells[n].sum() / SKY_WEIGHTS[k]) for k, n in enumerate(NAMES)
            },
        )
        checks[frame] = dict(map_log_err=map_err, quadrature_log_err=quadrature_err)

    # one shared colour scale from the sum, quantised to a byte per cell; byte 0 marks a part below the scale
    floor_log10 = math.log10(SKY_WEIGHTS[3] / (4 * math.pi))
    lo = min(min(m["sum"].min() for m in maps.values()), floor_log10)
    hi = max(m["sum"].max() for m in maps.values())
    for frame, m in maps.items():
        encoded = {}
        for part, log10_map in m.items():
            q = 1 + np.round((np.clip(log10_map, lo, hi) - lo) / (hi - lo) * 254)
            encoded[part] = base64.b64encode(
                np.where(log10_map < lo, 0, q).astype(np.uint8).tobytes()
            ).decode()
        out[frame]["map"] = encoded.pop("sum")
        out[frame]["component_maps"] = encoded
    sky = dict(
        frames=out,
        map=dict(
            width=MAP_SIZE[0],
            height=MAP_SIZE[1],
            log10_lo=float(lo),
            log10_hi=float(hi),
            encoding="byte 0: below log10_lo; 1..255: log10_lo..log10_hi",
            floor_log10=floor_log10,
        ),
        components=dict(
            names=NAMES,
            weights=SKY_WEIGHTS.tolist(),
            scales=[
                list(BULGE_SCALES),
                list(THICK_DISK_SCALES),
                list(THIN_DISK_SCALES),
                None,
            ],
            catalogue_sources=SKY_CATALOGUE_SOURCES,
        ),
        galactic_scale=list(GALACTIC_SCALE),
        quadrature=list(QUADRATURE_SIZE),
    )
    return sky, cdfs, checks


def hist(values: np.ndarray, lo: float, hi: float, bins=BINS, cdf=None) -> dict:
    """Counts on equal bins over [lo, hi], with the prior's expected counts from a cdf at the edges."""
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    out = dict(
        edges=edges.tolist(),
        counts=counts.tolist(),
        outside=int(values.size - counts.sum()),
    )
    if cdf is not None:
        out["reference"] = (np.diff(cdf(edges)) * values.size).tolist()
    return out


def spread(values: np.ndarray) -> dict:
    return dict(
        mean=float(values.mean()),
        std=float(values.std()),
        quantiles=np.quantile(values, QUANTILES).tolist(),
    )


def frame_marginals(lon: np.ndarray, lat: np.ndarray, cdf: dict, spec: dict) -> dict:
    (lon_edges, lon_cdf), (lat_edges, lat_cdf) = cdf["lon"], cdf["lat"]
    total = lon_cdf[-1]
    lat_lo, lat_hi, lat_bins = spec["lat_hist"]
    return dict(
        lon=hist(
            np.rad2deg(lon),
            *np.rad2deg(spec["lon"]),
            bins=72,
            cdf=lambda e: np.interp(np.deg2rad(e), lon_edges, lon_cdf / total),
        ),
        lat=hist(
            np.rad2deg(lat),
            lat_lo,
            lat_hi,
            bins=lat_bins,
            cdf=lambda e: np.interp(np.deg2rad(e), lat_edges, lat_cdf / total),
        ),
    )


def summarise_sky(lon: np.ndarray, lat: np.ndarray, cdfs: dict) -> dict:
    """The drawn sky per frame against the quadrature marginals, and how much of it sits near the plane and the centre."""
    marginals, near_plane = {}, {}
    for frame, spec in FRAMES.items():
        frame_lon, frame_lat = from_equatorial(frame, lon, lat)
        marginals[frame] = frame_marginals(
            frame_lon, frame_lat, cdfs[frame]["sum"], spec
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


def component_of(key: jax.Array) -> jax.Array:
    """The mixture component sample_sky picks for this key: the same split, the same choice."""
    return jr.choice(jr.split(key, 4)[0], SKY_WEIGHTS.size, p=jnp.asarray(SKY_WEIGHTS))


def bulge_candidate(key: jax.Array) -> jax.Array:
    """The bulge's untruncated (l, sin b) that sample_sky draws for this key, before its clip."""
    return jnp.asarray(BULGE_SCALES) * jr.normal(jr.split(key, 4)[1], (2,))


def draws_summary(
    ra: np.ndarray, dec: np.ndarray, comp: np.ndarray, cdfs: dict
) -> dict:
    """Per frame and part: marginals against the quadrature, the share near the plane, near the centre."""
    masks = dict(
        sum=np.ones_like(comp, bool), **{n: comp == k for k, n in enumerate(NAMES)}
    )
    marginals, near_plane = {}, {}
    for frame, spec in FRAMES.items():
        lon, lat = from_equatorial(frame, ra, dec)
        marginals[frame] = {
            part: frame_marginals(lon[m], lat[m], cdfs[frame][part], spec)
            for part, m in masks.items()
        }
        near_plane[frame] = {}
        for part, m in masks.items():
            lat_edges, lat_cdf = cdfs[frame][part]["lat"]
            cut = np.interp(
                [-math.radians(10), math.radians(10)], lat_edges, lat_cdf / lat_cdf[-1]
            )
            near_plane[frame][part] = dict(
                draws=float(np.mean(np.abs(lat[m]) < math.radians(10))),
                quadrature=float(cut[1] - cut[0]),
            )
    l, b = from_equatorial("galactic", ra, dec)
    from_centre = np.arccos(np.clip(np.cos(b) * np.cos(l), -1, 1))
    edges = cdfs["from_centre"]["edges"]
    centre, within = {}, {}
    for part, m in masks.items():
        quad = cdfs["from_centre"][part] / cdfs["from_centre"][part][-1]
        drawn = np.searchsorted(np.sort(from_centre[m]), edges, side="right") / m.sum()
        centre[part] = dict(
            draws=drawn.tolist(),
            quadrature=quad.tolist(),
            ks=float(np.max(np.abs(drawn - quad))),
        )
        within[part] = {
            str(r): dict(
                draws=float(np.mean(from_centre[m] < r)),
                quadrature=float(np.interp(r, edges, quad)),
            )
            for r in GC_RADII
        }
    return dict(
        n=int(comp.size),
        component_counts=[int((comp == k).sum()) for k in range(len(NAMES))],
        marginals=marginals,
        abs_lat_below_10deg=near_plane,
        within_rad_of_centre=within,
        from_centre=dict(edges=edges.tolist(), **centre),
    )


def rotation_check(ra: np.ndarray, dec: np.ndarray) -> dict:
    """EQUATORIAL_TO_GALACTIC against orthonormality, its own inverse, Hipparcos and astropy."""
    m = EQUATORIAL_TO_GALACTIC
    to_gal = jax.jit(equatorial_to_galactic)
    to_eq = jax.jit(galactic_to_equatorial)
    l, b = (np.asarray(c) for c in to_gal(jnp.asarray(ra), jnp.asarray(dec)))
    ra_back, dec_back = (np.asarray(c) for c in to_eq(jnp.asarray(l), jnp.asarray(b)))
    l_back, b_back = (
        np.asarray(c) for c in to_gal(jnp.asarray(ra_back), jnp.asarray(dec_back))
    )
    eq_round = angle_between(unit_vector(ra, dec), unit_vector(ra_back, dec_back))
    gal_round = angle_between(unit_vector(l, b), unit_vector(l_back, b_back))

    # the galactic centre and north pole through canna, against the Hipparcos definition
    centre = np.rad2deg([float(c) for c in to_eq(jnp.zeros(()), jnp.zeros(()))])
    pole = np.rad2deg(
        [float(c) for c in to_eq(jnp.zeros(()), jnp.asarray(math.pi / 2))]
    )
    arcsec = lambda a, ref: float(
        np.rad2deg(
            angle_between(unit_vector(*np.deg2rad(a)), unit_vector(*np.deg2rad(ref)))
        )
        * 3600
    )

    # astropy's ICRS -> Galactic and ICRS -> BarycentricMeanEcliptic(J2000) on a subset
    n = min(ASTROPY_DRAWS, ra.size)
    icrs = SkyCoord(ra=ra[:n] * u.rad, dec=dec[:n] * u.rad, frame="icrs")
    gal = icrs.galactic
    gal_err = angle_between(
        unit_vector(l[:n], b[:n]), unit_vector(gal.l.rad, gal.b.rad)
    )
    ecl = icrs.transform_to(BarycentricMeanEcliptic(equinox="J2000"))
    ecl_lon, ecl_lat = rotate(ECLIPTIC_TO_EQUATORIAL.T, ra[:n], dec[:n])
    ecl_err = angle_between(
        unit_vector(ecl_lon, ecl_lat), unit_vector(ecl.lon.rad, ecl.lat.rad)
    )
    return dict(
        matrix=m.tolist(),
        orthonormal_max_err=float(np.max(np.abs(m @ m.T - np.eye(3)))),
        det=float(np.linalg.det(m)),
        ecliptic_orthonormal_max_err=float(
            np.max(
                np.abs(ECLIPTIC_TO_EQUATORIAL @ ECLIPTIC_TO_EQUATORIAL.T - np.eye(3))
            )
        ),
        round_trip_rad=dict(
            equatorial_max=float(eq_round.max()),
            equatorial_median=float(np.median(eq_round)),
            galactic_max=float(gal_round.max()),
            galactic_median=float(np.median(gal_round)),
        ),
        centre_radec_deg=centre.tolist(),
        centre_hipparcos_deg=list(HIPPARCOS_CENTRE),
        centre_err_arcsec=arcsec(centre, HIPPARCOS_CENTRE),
        pole_radec_deg=pole.tolist(),
        pole_hipparcos_deg=list(HIPPARCOS_NGP),
        pole_err_arcsec=arcsec(pole, HIPPARCOS_NGP),
        astropy_draws=n,
        astropy_galactic_max_arcsec=float(np.rad2deg(gal_err.max()) * 3600),
        astropy_ecliptic_max_arcsec=float(np.rad2deg(ecl_err.max()) * 3600),
        obliquity_deg=math.degrees(OBLIQUITY),
    )


def lisa_frame(ra: np.ndarray, dec: np.ndarray, comp: np.ndarray) -> dict:
    """Where the galactic plane sits for LISA: the planes' tilts, the centre's ecliptic latitude, the prior's |beta|."""
    galactic_pole = EQUATORIAL_TO_GALACTIC[2]
    ecliptic_pole = ECLIPTIC_TO_EQUATORIAL[:, 2]
    tilt = lambda v, w: math.degrees(math.acos(abs(float(v @ w))))
    centre = EQUATORIAL_TO_GALACTIC[0]
    centre_lon, centre_lat = rotate(
        ECLIPTIC_TO_EQUATORIAL.T, math.atan2(centre[1], centre[0]), math.asin(centre[2])
    )
    _, beta = rotate(ECLIPTIC_TO_EQUATORIAL.T, ra, dec)
    abs_beta = np.rad2deg(np.abs(beta))
    masks = dict(
        sum=np.ones_like(comp, bool), **{n: comp == k for k, n in enumerate(NAMES)}
    )
    return dict(
        galactic_to_ecliptic_tilt_deg=tilt(galactic_pole, ecliptic_pole),
        galactic_to_equator_tilt_deg=tilt(galactic_pole, np.array([0.0, 0.0, 1.0])),
        centre_ecliptic_deg=[
            float(np.rad2deg(np.mod(centre_lon, 2 * math.pi))),
            float(np.rad2deg(centre_lat)),
        ],
        abs_beta={
            part: dict(
                hist=hist(abs_beta[m], 0, 90, bins=45),
                median=float(np.median(abs_beta[m])),
                below_10deg=float(np.mean(abs_beta[m] < 10)),
                below_30deg=float(np.mean(abs_beta[m] < 30)),
                cos_beta_mean=float(np.mean(np.cos(beta[m]))),
                cos_beta_median=float(np.median(np.cos(beta[m]))),
            )
            for part, m in masks.items()
        },
        isotropic=dict(
            below_10deg=math.sin(math.radians(10)),
            below_30deg=math.sin(math.radians(30)),
            cos_beta_mean=math.pi / 4,
            median_deg=math.degrees(math.asin(0.5)),
        ),
    )


def chart_check(
    ra: np.ndarray, dec: np.ndarray, comp: np.ndarray, bulge: np.ndarray, cdfs: dict
) -> dict:
    """GALACTIC_SCALE's whitening, each part's width in flow units, the wrap at l = ±pi."""
    l, b = from_equatorial("galactic", ra, dec)
    scale = np.array(GALACTIC_SCALE)
    masks = dict(
        sum=np.ones_like(comp, bool), **{n: comp == k for k, n in enumerate(NAMES)}
    )
    second_moment = np.array([np.mean(l**2), np.mean(b**2)])
    parts = {}
    for part, m in masks.items():
        x = np.stack([l[m] / scale[0], b[m] / scale[1]])
        parts[part] = dict(
            share=float(m.mean()),
            flow_std=x.std(1).tolist(),
            variance_share=(
                np.array([np.sum(l[m] ** 2), np.sum(b[m] ** 2)])
                / (second_moment * l.size)
            ).tolist(),
        )

    # mass within delta of l = ±pi, where the flow's l jumps from +pi / scale to -pi / scale
    l_edges, l_cdf = cdfs["galactic"]["sum"]["lon"]
    l_cdf = l_cdf / l_cdf[-1]
    near_wrap = {}
    for delta in WRAP_DELTAS:
        lo, hi = np.interp([-math.pi + delta, math.pi - delta], l_edges, l_cdf)
        near_wrap[str(delta)] = dict(
            draws=float(np.mean(np.abs(l) > math.pi - delta)),
            quadrature=float(lo + 1 - hi),
            by_part={
                part: float(np.mean(np.abs(l[m]) > math.pi - delta))
                for part, m in masks.items()
            },
        )
    plane = lambda lon: float(
        np.exp(log_density_per_steradian("galactic", np.asarray(lon), np.zeros(())))
    )
    return dict(
        galactic_scale=scale.tolist(),
        galactic_scale_recomputed=list(galactic_scales()),
        draws_std=[float(l.std()), float(b.std())],
        wrap_flow=[math.pi / scale[0], math.pi / 2 / scale[1]],
        parts=parts,
        near_wrap=near_wrap,
        plane_density=dict(
            centre=plane(0.0),
            quarter=plane(math.pi / 2),
            anticentre=plane(math.pi),
            floor=float(SKY_WEIGHTS[3] / (4 * math.pi)),
        ),
        clip=dict(
            bulge_draws=int((comp == 0).sum()),
            bulge_sin_b_beyond_1=int(np.sum((comp == 0) & (np.abs(bulge[:, 1]) > 1))),
            bulge_l_beyond_pi=int(
                np.sum((comp == 0) & (np.abs(bulge[:, 0]) > math.pi))
            ),
            log10_p_sin_b_beyond_1=float(
                (math.log(2) + jax.scipy.special.log_ndtr(-1 / BULGE_SCALES[1]))
                / math.log(10)
            ),
            log10_p_l_beyond_pi=float(
                (math.log(2) + jax.scipy.special.log_ndtr(-math.pi / BULGE_SCALES[0]))
                / math.log(10)
            ),
        ),
    )


def band_sky(problem: LisaGB, cdfs: dict) -> dict:
    """The training prior's own windows: sky summary, l and b as the flow sees them, per slot, and their correlations."""

    def draw(key: jax.Array) -> dict:
        key_f, key_p = jr.split(key)
        f = sample_f(problem, key_f)
        p = sample_physical(problem, key_p, f)
        return dict(p=p, x=physical_to_flow(problem, p, f))

    start = time.perf_counter()
    d = jax.lax.map(draw, jr.split(jr.key(0), WINDOWS), batch_size=256)
    p, x = (np.asarray(v) for v in jax.block_until_ready((d["p"], d["x"])))
    seconds = time.perf_counter() - start
    lon, lat = p[..., 3], p[..., 4]

    (l_edges, l_cdf), (b_edges, b_cdf) = (
        cdfs["galactic"]["sum"]["lon"],
        cdfs["galactic"]["sum"]["lat"],
    )
    flow_cdfs = {
        2: lambda e: np.interp(e * GALACTIC_SCALE[0], l_edges, l_cdf / l_cdf[-1]),
        3: lambda e: np.interp(e * GALACTIC_SCALE[1], b_edges, b_cdf / b_cdf[-1]),
    }
    flow = []
    for j in (2, 3):
        v = x[..., j].ravel()
        flow.append(
            dict(
                name=FLOW_NAMES[j],
                hist=hist(v, -6.0, 6.0, cdf=flow_cdfs[j]),
                **spread(v),
            )
        )
    corr = np.corrcoef(x.reshape(-1, 8), rowvar=False)
    _, gal_b = from_equatorial("galactic", lon, lat)
    slots = [
        dict(
            l=spread(x[:, k, 2]),
            b=spread(x[:, k, 3]),
            abs_b_below_10deg=float(np.mean(np.abs(gal_b[:, k]) < math.radians(10))),
        )
        for k in range(problem.n_sources)
    ]
    return dict(
        windows=WINDOWS,
        sources=int(lon.size),
        seconds=seconds,
        sky=summarise_sky(lon.ravel(), lat.ravel(), cdfs),
        flow=flow,
        corr_names=FLOW_NAMES,
        corr_l=corr[2].tolist(),
        corr_b=corr[3].tolist(),
        slots=slots,
    )


def main(out: str) -> None:
    device = jax.devices()[0]
    print(
        f"device: {device.device_kind}  x64: {jax.config.jax_enable_x64}  draws {DRAWS}  windows {WINDOWS}",
        flush=True,
    )
    start = time.perf_counter()
    sky, cdfs, map_checks = sky_frames()
    integrals = ", ".join(f"{k} {v['integral']:.6f}" for k, v in sky["frames"].items())
    print(
        f"sky: integrals {integrals} ({time.perf_counter() - start:.0f} s)", flush=True
    )

    start = time.perf_counter()
    keys = jr.split(jr.key(3), DRAWS)
    ra, dec = (np.asarray(c) for c in jax.jit(jax.vmap(sample_sky))(keys).T)
    comp = np.asarray(jax.jit(jax.vmap(component_of))(keys))
    bulge = np.asarray(jax.jit(jax.vmap(bulge_candidate))(keys))
    print(f"draws: {DRAWS} in {time.perf_counter() - start:.1f} s", flush=True)

    result = dict(
        device=device.device_kind,
        draws=DRAWS,
        sky=sky,
        sky_draws=draws_summary(ra, dec, comp, cdfs),
        checks=dict(frames=map_checks, rotation=rotation_check(ra, dec)),
        lisa=lisa_frame(ra, dec, comp),
        chart=chart_check(ra, dec, comp, bulge, cdfs),
        bands={},
    )
    r = result["checks"]["rotation"]
    print(
        f"rotation: |MM^T - I| {r['orthonormal_max_err']:.1e}, round trip {r['round_trip_rad']['equatorial_max']:.1e} rad,"
        f' centre {r["centre_err_arcsec"]:.3f}", astropy {r["astropy_galactic_max_arcsec"]:.3f}"',
        flush=True,
    )
    for band in BANDS:
        cfg = config.load(f"{band}-B")
        problem = LisaGB(**cfg.problem)
        b = result["bands"][band] = band_sky(problem, cdfs)
        print(
            f"{band}: {WINDOWS} windows in {b['seconds']:.1f} s; |b| < 10° {b['sky']['abs_lat_below_10deg']['galactic']:.4f};"
            f" flow std "
            + ", ".join(f"{f['name']} {f['std']:.3f}" for f in b["flow"]),
            flush=True,
        )
    with open(out, "w") as file:
        json.dump(result, file)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
