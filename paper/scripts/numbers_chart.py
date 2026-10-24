"""The numbers in appendix/chart.tex, from the lisa chart, physics, configs and the prior diagnostic."""

import json
import math

import numpy as np

import style
from canna.lisa import config
from canna.lisa.chart import GALACTIC_SCALE
from canna.lisa.network import COORD_RANGES, Z_REACH
from canna.lisa.physics import (
    BULGE_SCALES,
    SKY_WEIGHTS,
    THICK_DISK_SCALES,
    THIN_DISK_SCALES,
)
from canna.lisa.problem import LisaGB

EDGE_DEG = 10.0
prior = json.loads((style.REPO / "reports/prior/data/prior.json").read_text())
bands = {band: prior["bands"][band] for band in ("narrow", "full")}
problems = {
    band: LisaGB(**config.load(f"{band}-B").problem) for band in ("narrow", "full")
}
narrow, full = problems["narrow"], problems["full"]
assert (narrow.mass_range, narrow.n_sources, narrow.snr_range) == (
    full.mass_range,
    full.n_sources,
    full.snr_range,
)
assert narrow.phase_reference == full.phase_reference == "middle"


def marginal_mass(axis: int, inside: float) -> float:
    """Prior mass of the sky mixture's (l, sin b) marginal on axis beyond |x| > inside."""
    half_width = (math.pi, 1.0)[axis]
    x = np.linspace(-half_width, half_width, 400001)
    bulge = np.exp(-0.5 * (x / BULGE_SCALES[axis]) ** 2) / (
        np.sqrt(2 * np.pi) * BULGE_SCALES[axis]
    )
    thick, thin = (
        np.exp(-np.abs(x) / s) / (2 * s * (1 - np.exp(-half_width / s)))
        for s in (THICK_DISK_SCALES[axis], THIN_DISK_SCALES[axis])
    )
    floor = np.full_like(x, 1 / (2 * half_width))
    density = SKY_WEIGHTS @ np.stack([bulge, thick, thin, floor])
    return float(np.trapezoid(np.where(np.abs(x) > inside, density, 0.0), x))


edge = math.radians(EDGE_DEG)
seam_mass = marginal_mass(0, math.pi - edge)
pole_mass = marginal_mass(1, math.cos(edge))

# the prior pushed through the chart, per band, from the prior diagnostic
flow = {band: {c["name"]: c for c in data["flow"]} for band, data in bands.items()}
z_names = ("Re z+", "Im z+", "Re z-", "Im z-")
z_std = [flow[band][name]["std"] for band in flow for name in z_names]
unit_std = [flow[band][name]["std"] for band in flow for name in ("Mc", "l", "b")]
max_corr = max(
    float(np.max(np.abs(np.asarray(data["corr"]) - np.eye(8))))
    for data in bands.values()
)
kappa = {band: data["table"]["kappa"]["quantiles"] for band, data in bands.items()}

# kappa against the sky-averaged inclination law, over the drawn narrow-band sources
scatter = bands["narrow"]["scatter"]
cos_iota, kappa_drawn = np.asarray(scatter["sin_iota"]), np.asarray(scatter["kappa"])
sky_constant = bands["narrow"]["sky_average"]["galactic"]["C_low"]
kappa_law = np.sqrt(sky_constant * (1 + 6 * cos_iota**2 + cos_iota**4) / 3.2)
kappa_law_rms = float(np.sqrt(np.mean((kappa_drawn / kappa_law - 1) ** 2)))

# one unit of x_f in frequency bins, and float32's spacing there in the full band
bins_per_unit = {
    band: (p.window_bins - p.response_points) / 2 for band, p in problems.items()
}
f32_bins_full = bins_per_unit["full"] * float(np.spacing(np.float32(1.0)))

reach = COORD_RANGES[1][1]
assert all(r == (-reach, reach) for r in COORD_RANGES[1:4])
numbers = {
    "ChartCoords": "8",
    "ChartSources": f"{full.n_sources}",
    "ChartMcMedian": f"{math.exp(float(np.ravel(full.log_mc_mean)[0])):.2f}",
    "ChartMcLogStd": f"{float(np.ravel(full.log_mc_std)[0]):.2f}",
    "ChartSigmaL": f"{GALACTIC_SCALE[0]:.3f}",
    "ChartSigmaB": f"{GALACTIC_SCALE[1]:.3f}",
    "ChartSigmaLDeg": f"{math.degrees(GALACTIC_SCALE[0]):.0f}",
    "ChartSigmaBDeg": f"{math.degrees(GALACTIC_SCALE[1]):.1f}",
    "ChartSeamX": f"{math.pi / GALACTIC_SCALE[0]:.1f}",
    "ChartPoleX": f"{math.pi / 2 / GALACTIC_SCALE[1]:.1f}",
    "ChartEdgeDeg": f"{EDGE_DEG:g}",
    "ChartSeamMass": f"{100 * seam_mass:.2f}",
    "ChartPoleMass": f"{100 * pole_mass:.2f}",
    "ChartReach": f"{reach:g}",
    "ChartZReach": f"{Z_REACH:.2f}",
    "ChartSnrHigh": f"{full.snr_range[1]:g}",
    "ChartBeyondL": f"{100 * max(flow[b]['l']['beyond_reach'] for b in flow):.1f}",
    "ChartBeyondB": f"{100 * max(flow[b]['b']['beyond_reach'] for b in flow):.1f}",
    "ChartUnitStdLow": f"{min(unit_std):.2f}",
    "ChartUnitStdHigh": f"{max(unit_std):.2f}",
    "ChartZStdLow": f"{min(z_std):.2f}",
    "ChartZStdHigh": f"{max(z_std):.2f}",
    "ChartMaxCorr": f"{max_corr:.3f}",
    "ChartKappaNarrowLow": f"{kappa['narrow'][0]:.2f}",
    "ChartKappaNarrowHigh": f"{kappa['narrow'][-1]:.2f}",
    "ChartKappaFullLow": f"{kappa['full'][0]:.2f}",
    "ChartKappaFullHigh": f"{kappa['full'][-1]:.2f}",
    "ChartSkyConstant": f"{sky_constant:.2f}",
    "ChartKappaLawRms": f"{100 * kappa_law_rms:.0f}",
    "ChartPriorDraws": f"{bands['narrow']['sources']}",
    "ChartBinsPerUnitNarrow": f"{bins_per_unit['narrow']:g}",
    "ChartBinsPerUnitFull": f"{bins_per_unit['full']:g}",
    "ChartFThirtyTwoBinsFull": f"{f32_bins_full:.1e}".replace("e-0", "e-"),
}
for key, value in numbers.items():
    print(f"{key:28s} {value}")
style.save_numbers("chart", numbers)
