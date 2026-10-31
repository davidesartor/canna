"""The numbers in appendix/chart.tex, from the lisa chart, physics, configs and the prior reports."""

import json
import math

import numpy as np

import style
from canna.lisa import config
from canna.lisa.chart import GALACTIC_SCALE
from canna.lisa.physics import (
    BULGE_SCALES,
    FDOT_STD,
    SKY_WEIGHTS,
    SOURCE_SLOPE,
    THICK_DISK_SCALES,
    THIN_DISK_SCALES,
    log10_fdot_mean,
)
from canna.lisa.problem import LisaGB, f0_window

EDGE_DEG = 10.0
BANDS = ("narrow", "full")
prior = {
    name: json.loads((style.REPO / f"reports/prior/{name}/data/{name}.json").read_text())["bands"]
    for name in ("f0-fdot-amplitude", "sky", "polarization")
}
draws = {
    band: np.load(style.REPO / f"reports/prior/f0-fdot-amplitude/data/{band}.npz")
    for band in BANDS
}
problems = {band: LisaGB(**config.load(f"{band}-B").problem) for band in BANDS}
narrow, full = problems["narrow"], problems["full"]
assert narrow.n_sources == full.n_sources
assert narrow.phase_reference == full.phase_reference == "middle"


def chart_f0_fdot(problem: LisaGB, d) -> tuple[np.ndarray, np.ndarray]:
    """x_f and x_fdot of the drawn sources, as physical_to_flow maps them."""
    lo, hi = (np.log(np.asarray(e))[:, None] for e in f0_window(problem, d["f"]))
    f0 = 10.0 ** d["log_f0"]
    x_fdot = (d["log_fdot"] - np.asarray(log10_fdot_mean(f0))) / FDOT_STD
    return (2 * np.log(f0) - lo - hi) / (hi - lo), x_fdot


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

# the prior pushed through the chart, per band: x_f, x_fdot from the source draws, the rest from the reports
flow = {
    band: {c["name"]: c for name in ("sky", "polarization") for c in prior[name][band]["flow"]}
    for band in BANDS
}
z_names = ("Re z+", "Im z+", "Re z-", "Im z-")
z_std = [flow[band][name]["std"] for band in BANDS for name in z_names]
x_f0_fdot = {band: chart_f0_fdot(problems[band], draws[band]) for band in BANDS}
x_f_std = [float(x_f0_fdot[band][0].std()) for band in BANDS]
unit_std = [float(x_f0_fdot[band][1].std()) for band in BANDS] + [
    flow[band][name]["std"] for band in BANDS for name in ("l", "b")
]
max_corr = max(
    float(np.max(np.abs(np.asarray(data["corr"]) - np.eye(8))))
    for data in prior["polarization"].values()
)

# SNR per unit amplitude, rho u / A, over the drawn sources of each band
scatters = {band: prior["polarization"][band]["scatter"] for band in BANDS}
kappas = {
    band: np.asarray(s["snr"]) / 10 ** np.asarray(s["log_amp"]) for band, s in scatters.items()
}
kappa = {band: np.quantile(k, [0.01, 0.99]) for band, k in kappas.items()}

# kappa against the sky-averaged inclination law, over the drawn narrow-band sources
cos_iota, kappa_drawn = np.asarray(scatters["narrow"]["sin_iota"]), kappas["narrow"]
sky_constant = prior["f0-fdot-amplitude"]["narrow"]["sky_average"]["galactic"]["C_low"]
kappa_law = np.sqrt(sky_constant * (1 + 6 * cos_iota**2 + cos_iota**4) / 3.2)
kappa_law_rms = float(np.sqrt(np.mean((kappa_drawn / kappa_law - 1) ** 2)))

# one unit of x_f in frequency bins, and float32's spacing there in the full band
bins_per_unit = {
    band: (p.window_bins - p.response_points) / 2 for band, p in problems.items()
}
f32_bins_full = bins_per_unit["full"] * float(np.spacing(np.float32(1.0)))

numbers = {
    "ChartCoords": "8",
    "ChartSources": f"{full.n_sources}",
    "ChartFdotSlope": f"{SOURCE_SLOPE[0]:.2f}",
    "ChartFdotStd": f"{FDOT_STD:.2f}",
    "ChartXfStd": f"{np.mean(x_f_std):.3f}",
    "ChartSigmaL": f"{GALACTIC_SCALE[0]:.3f}",
    "ChartSigmaB": f"{GALACTIC_SCALE[1]:.3f}",
    "ChartSigmaLDeg": f"{math.degrees(GALACTIC_SCALE[0]):.0f}",
    "ChartSigmaBDeg": f"{math.degrees(GALACTIC_SCALE[1]):.1f}",
    "ChartSeamX": f"{math.pi / GALACTIC_SCALE[0]:.1f}",
    "ChartPoleX": f"{math.pi / 2 / GALACTIC_SCALE[1]:.1f}",
    "ChartEdgeDeg": f"{EDGE_DEG:g}",
    "ChartSeamMass": f"{100 * seam_mass:.2f}",
    "ChartPoleMass": f"{100 * pole_mass:.2f}",
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
    "ChartPriorDraws": f"{prior['polarization']['narrow']['sources']}",
    "ChartBinsPerUnitNarrow": f"{bins_per_unit['narrow']:g}",
    "ChartBinsPerUnitFull": f"{bins_per_unit['full']:g}",
    "ChartFThirtyTwoBinsFull": f"{f32_bins_full:.1e}".replace("e-0", "e-"),
}
for key, value in numbers.items():
    print(f"{key:28s} {value}")
style.save_numbers("chart", numbers)
