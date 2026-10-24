"""Numbers of appendix/priors.tex: the prior over a window, its symmetries and the training distribution."""

import json
import math

import numpy as np
from scipy.stats import beta

import style

from canna.lisa import config
from canna.lisa.physics import (
    BULGE_SCALES,
    PRIMARY_MEANS,
    PRIMARY_STDS,
    PRIMARY_WEIGHTS,
    SKY_CATALOGUE_SOURCES,
    SKY_WEIGHTS,
    THICK_DISK_SCALES,
    THIN_DISK_SCALES,
    chirp_mass_log_density,
    chirp_mass_support,
    fdot_from_chirp_mass,
    galactic_to_equatorial,
)
from canna.lisa.problem import WINDOW_RESPONSES, WINDOW_STRIDE, LisaGB

BANDS = ("narrow", "full")
PLANE_DEG = 10


def sci(x: float, digits: int = 1) -> str:
    mantissa, exponent = f"{x:.{digits}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def f0_marginal_ratio(problem: LisaGB, grid: np.ndarray) -> np.ndarray:
    """Exact p(ln f0) of sample_f + f0_window over the log-uniform density, on a grid of ln(f0 T)."""
    low, high = (f * problem.t_obs for f in problem.f0_range)
    span = math.log(high / low)
    bins, guard = problem.window_bins, problem.response_points // 2
    lo, hi = problem.window_index_range
    index = np.arange(lo, hi + 1)
    centre = index * WINDOW_STRIDE + bins / 2
    u_of = lambda c: np.clip(np.log(np.maximum(c, 1e-300) / low) / span, 0.0, 1.0)
    upper = np.where(index == hi, np.inf, centre + WINDOW_STRIDE / 2)
    lower = np.where(index == lo, 0.0, centre - WINDOW_STRIDE / 2)
    prob = u_of(upper) - u_of(lower)
    a = np.log(np.maximum(index * WINDOW_STRIDE + guard, low))
    b = np.log(np.minimum(index * WINDOW_STRIDE + bins - guard, high))
    height = prob / (b - a)
    density = np.zeros(grid.size + 1)
    np.add.at(density, np.searchsorted(grid, a), height)
    np.add.at(density, np.searchsorted(grid, b, side="right"), -height)
    assert abs(prob.sum() - 1) < 1e-9
    return np.cumsum(density)[:-1] * span


numbers = {}
problems, cfgs = {}, {}
with open(style.REPO / "reports/prior/data/prior.json") as file:
    report = json.load(file)
reported, reported_windows = report["bands"], report["windows"]
for band in BANDS:
    cfgs[band] = config.load(f"{band}-B")
    problems[band] = LisaGB(**cfgs[band].problem)
narrow, full = problems["narrow"], problems["full"]
for name in ("n_sources", "snr_range", "mass_range", "t_obs"):
    assert np.all(np.asarray(getattr(narrow, name)) == np.asarray(getattr(full, name)))
assert narrow.f0_range[0] == full.f0_range[0]
n_sources, (snr_lo, snr_hi), mass_range = (
    full.n_sources,
    full.snr_range,
    full.mass_range,
)
mc_lo, mc_hi = chirp_mass_support(mass_range)

# the window and its interior, per band
for band, problem in problems.items():
    Band = band.capitalize()
    # the f0 marginal near each band edge, against the distance from it in units of R
    low, high = (f * problem.t_obs for f in problem.f0_range)
    R = problem.response_points
    distance = np.linspace(0, 6, 6001)
    bottom = f0_marginal_ratio(problem, np.log(low + R * distance))
    top = f0_marginal_ratio(problem, np.log(high - R * distance)[::-1])[::-1]
    extent = max(
        distance[np.flatnonzero(np.abs(r - 1) > 0.01).max()] for r in (bottom, top)
    )
    assert extent < distance[-1] - 0.1
    print(
        band,
        "ratio at f_lo",
        bottom[0],
        "at f_hi",
        top[0],
        "max",
        bottom.max(),
        top.max(),
        "1% extent [R]",
        extent,
    )
    lo, hi = problem.window_index_range
    numbers |= {
        f"Priors{Band}R": f"{problem.response_points}",
        f"Priors{Band}W": f"{problem.window_bins}",
        f"Priors{Band}HighMhz": f"{problem.f0_range[1] * 1e3:.3g}",
        f"Priors{Band}Windows": f"{hi - lo + 1}",
        f"Priors{Band}EdgeExtent": f"{extent:.1f}",
        f"Priors{Band}AmpLow": sci(reported[band]["table"]["A"]["quantiles"][0]),
        f"Priors{Band}AmpHigh": sci(reported[band]["table"]["A"]["quantiles"][-1]),
    }

# mode and median of the induced chirp mass, from p(log Mc) on its uniform log grid
log_mc, log_p = chirp_mass_log_density(mass_range)
p = np.exp(log_p)
cdf = np.cumsum(p) / p.sum()
mc_mode = math.exp(log_mc[np.argmax(p / np.exp(log_mc))])
mc_median = math.exp(np.interp(0.5, cdf, log_mc))
check = reported["full"]["chirp_mass_check"]

# the galactic centre in the equatorial frame jaxgb reads
ra, dec = (math.degrees(float(a)) for a in galactic_to_equatorial(0.0, 0.0))

# the k-th loudest of S log-uniform SNRs: u = ln(rho/rho_lo)/ln(rho_hi/rho_lo) ~ Beta(S-k+1, k)
slot_medians = [
    snr_lo * (snr_hi / snr_lo) ** beta.median(n_sources - k + 1, k)
    for k in range(1, n_sources + 1)
]
for k, rank in enumerate(reported["full"]["snr_rank"]):
    edges, counts = np.asarray(rank["edges"]), np.cumsum(rank["counts"])
    print(
        f"slot {k + 1}: Beta median {slot_medians[k]:.1f}, prior.json median",
        10 ** np.interp(counts[-1] / 2, counts, edges[1:]),
    )

# the chirp mass's Fisher precision against its prior, per slot (reports/mcmc)
with open(style.REPO / "reports/mcmc/data/fisher-diagonal.json") as file:
    fisher = json.load(file)
mc = fisher["params"].index("Mc")
gain = {}
for band in BANDS:
    data = fisher["bands"][band]
    prior_mc = np.asarray(data["log10_prior_diag"])[mc :: len(fisher["params"])]
    precision_mc = np.asarray([i["log10_precision_diag"] for i in data["injections"]])[
        :, mc :: len(fisher["params"])
    ]
    gain[band] = np.median(precision_mc - prior_mc, axis=0)
    print(
        band,
        "Mc log10 precision over prior, median per slot",
        gain[band],
        "prior",
        prior_mc,
    )
    numbers[f"Priors{band.capitalize()}FisherInjections"] = f"{len(data['injections'])}"

# share of the drawn prior sources whose chirp drift over the observation exceeds one bin
for band, problem in problems.items():
    scatter = reported[band]["scatter"]
    drift_bins = (
        np.asarray(
            fdot_from_chirp_mass(np.asarray(scatter["mc"]), np.asarray(scatter["f0"]))
        )
        * problem.t_obs**2
    )
    numbers[f"Priors{band.capitalize()}DriftShare"] = (
        f"{100 * np.mean(drift_bins > 1):.0f}"
    )

# the training distribution each band's run sees
draws = cfgs["full"].draws
observations = cfgs["full"].batch_size // draws
assert all(
    c.draws == draws and c.batch_size // c.draws == observations for c in cfgs.values()
)
for band, cfg in cfgs.items():
    Band = band.capitalize()
    windows = cfg.total_steps * observations
    assert reported[band]["dataset"]["windows"] == windows, reported[band]["dataset"]
    numbers |= {
        f"Priors{Band}Steps": f"{cfg.total_steps}",
        f"Priors{Band}Observations": sci(windows),
        f"Priors{Band}SourcesSeen": sci(windows * n_sources),
    }
    path = f"reports/benchmarks/step-time-memory/data/a100-bf16backbone-{band}.jsonl"
    with open(style.REPO / path) as file:
        (step,) = [
            r
            for r in map(json.loads, file)
            if r["config"] == f"{band}-B"
            and r["batch"] == r["micro"] == observations
            and "step_s" in r
        ]
    numbers |= {
        f"Priors{Band}StepMs": f"{1e3 * step['step_s']:.0f}",
        f"Priors{Band}DrawMs": f"{float(f'{1e3 * step["draw_s"]:.2g}'):g}",
        f"Priors{Band}DrawShare": f"{100 * step['draw_s'] / step['step_s']:.0f}",
    }
numbers["PriorsStepChip"] = step["chip"].removeprefix("NVIDIA ")

window_snr = reported["full"]["table"]["window snr"]["quantiles"]
numbers |= {
    "PriorsSources": f"{n_sources}",
    "PriorsCoords": "8",
    "PriorsStride": f"{WINDOW_STRIDE}",
    "PriorsInteriorResponses": f"{WINDOW_RESPONSES - 1}",
    "PriorsBandLowMhz": f"{full.f0_range[0] * 1e3:.1f}",
    "PriorsTobsYears": f"{full.t_obs / (365.25 * 86400):.0f}",
    "PriorsMassLow": f"{mass_range[0]:g}",
    "PriorsMassHigh": f"{mass_range[1]:g}",
    "PriorsMcLow": f"{mc_lo:.3f}",
    "PriorsMcHigh": f"{mc_hi:.3f}",
    "PriorsPrimaryWeights": ", ".join(f"{x:g}" for x in PRIMARY_WEIGHTS),
    "PriorsPrimaryMeans": ", ".join(f"{x:g}" for x in PRIMARY_MEANS),
    "PriorsPrimaryStds": ", ".join(f"{x:g}" for x in PRIMARY_STDS),
    "PriorsMcMode": f"{mc_mode:.2f}",
    "PriorsMcMedian": f"{mc_median:.2f}",
    "PriorsMcCheckDraws": sci(check["draws"], 0),
    "PriorsMcCheckChi": f"{check['chi2_per_dof']:.2f}",
    "PriorsMcCheckDof": f"{check['dof']}",
    "PriorsSnrLow": f"{snr_lo:.0f}",
    "PriorsSnrHigh": f"{snr_hi:.0f}",
    "PriorsSlotMedians": ", ".join(f"{x:.0f}" for x in slot_medians),
    "PriorsPriorWindows": f"{reported_windows}",
    "PriorsWindowSnrMedian": f"{window_snr[2]:.0f}",
    "PriorsWindowSnrLow": f"{window_snr[0]:.0f}",
    "PriorsWindowSnrHigh": f"{window_snr[-1]:.0f}",
    "PriorsBulgeScales": ", ".join(f"{x:.2g}" for x in BULGE_SCALES),
    "PriorsThickScales": ", ".join(f"{x:.2g}" for x in THICK_DISK_SCALES),
    "PriorsThinScales": ", ".join(f"{x:.2g}" for x in THIN_DISK_SCALES),
    "PriorsBulgeWeight": f"{100 * SKY_WEIGHTS[0]:.0f}",
    "PriorsThickWeight": f"{100 * SKY_WEIGHTS[1]:.0f}",
    "PriorsThinWeight": f"{100 * SKY_WEIGHTS[2]:.0f}",
    "PriorsFloorWeight": f"{100 * SKY_WEIGHTS[3]:.1f}",
    "PriorsCatalogue": f"{SKY_CATALOGUE_SOURCES / 10**6:.0f}e6",
    "PriorsCentreRa": f"{ra:.1f}",
    "PriorsCentreDec": f"{dec:.1f}",
    "PriorsPlaneDeg": f"{PLANE_DEG}",
    "PriorsPlaneShare": f"{100 * reported['full']['sky'][f'abs_lat_below_{PLANE_DEG}deg']['galactic']:.0f}",
    "PriorsPermutations": f"{math.factorial(n_sources)}",
    "PriorsCopies": f"{math.factorial(n_sources) * 4**n_sources}",
    "PriorsNarrowMcGain": f"{gain['narrow'].max():.1g}",
    "PriorsFullMcGainLow": f"{gain['full'].min():.1f}",
    "PriorsFullMcGainHigh": f"{gain['full'].max():.1f}",
    "PriorsBatch": f"{cfgs['full'].batch_size}",
    "PriorsDraws": f"{draws}",
    "PriorsObservationsPerStep": f"{observations}",
    "PriorsPasses": f"{draws + 2}",
}

for k, v in numbers.items():
    print(f"{k} = {v}")
style.save_numbers("priors", numbers)
