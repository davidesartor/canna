"""Numbers of appendix/priors.tex: the prior over a window, its symmetries and the training distribution."""

import json
import math

import numpy as np
from scipy.special import ndtr

import style

from canna.lisa import config
from canna.lisa.physics import (
    BULGE_SCALES,
    SKY_CATALOGUE_SOURCES,
    SKY_WEIGHTS,
    SOURCE_COVARIANCE,
    SOURCE_MEAN,
    THICK_DISK_SCALES,
    THIN_DISK_SCALES,
    galactic_to_equatorial,
)
from canna.lisa.problem import WINDOW_RESPONSES, WINDOW_STRIDE, LisaGB

BANDS = ("narrow", "full")
PLANE_DEG = 10
VISIBLE_SNR = 7.0
F0_MEAN, F0_STD = SOURCE_MEAN[0], math.sqrt(SOURCE_COVARIANCE[0, 0])


def sci(x: float, digits: int = 1) -> str:
    mantissa, exponent = f"{x:.{digits}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def percent(share: float) -> str:
    """A share in percent: two significant figures below 10%, one decimal above."""
    return (
        "0"
        if share == 0
        else f"{100 * share:.2g}" if share < 0.1 else f"{100 * share:.1f}"
    )


def f0_cdf(f0_t: np.ndarray, t_obs: float) -> np.ndarray:
    """The catalogue's f0 marginal CDF at f0 T, untruncated."""
    return ndtr((np.log10(np.maximum(f0_t, 1e-300) / t_obs) - F0_MEAN) / F0_STD)


numbers = {}
problems, cfgs = {}, {}
prior_report = json.loads(
    (
        style.REPO / "reports/prior/f0-fdot-amplitude/data/f0-fdot-amplitude.json"
    ).read_text()
)
prior = prior_report["prior"]
sky = json.loads((style.REPO / "reports/prior/sky/data/sky.json").read_text())["bands"]
for band in BANDS:
    cfgs[band] = config.load(f"{band}-B")
    problems[band] = LisaGB(**cfgs[band].problem)
narrow, full = problems["narrow"], problems["full"]
for name in ("n_sources", "t_obs"):
    assert getattr(narrow, name) == getattr(full, name)
assert narrow.f0_range[0] == full.f0_range[0]
n_sources = full.n_sources
assert np.allclose(prior["mean"], SOURCE_MEAN) and np.allclose(
    prior["cov"], SOURCE_COVARIANCE
)

# the window, its interior and its sources, per band
for band, problem in problems.items():
    Band = band.capitalize()
    report = prior_report["bands"][band]
    assert report["problem"]["response_points"] == problem.response_points
    low, high = (f * problem.t_obs for f in problem.f0_range)
    R = problem.response_points
    band_mass = f0_cdf(high, problem.t_obs) - f0_cdf(low, problem.t_obs)
    cut_mass = 1 - ndtr((prior["cut"] - F0_MEAN) / F0_STD)

    # the catalogue's windows: visible sources, empty windows, drift and span, where the centres fall
    z = np.load(style.REPO / f"reports/prior/f0-fdot-amplitude/data/{band}.npz")
    visible = z["snr"] >= VISIBLE_SNR
    log_lo, log_hi = (math.log10(f) for f in problem.f0_range)
    top = lambda f: (np.log10(f) >= (log_lo + log_hi) / 2).mean()

    # training's windows, centred by the visible prior at k = visible_snr
    assert problem.visible_snr == report["training_visible_snr"] == VISIBLE_SNR
    cut = np.load(style.REPO / f"reports/prior/f0-fdot-amplitude/data/{band}-cut.npz")
    (k,) = np.flatnonzero(cut["visible_snr"] == VISIBLE_SNR)
    lo, hi = problem.window_index_range
    numbers |= {
        f"Priors{Band}R": f"{R}",
        f"Priors{Band}W": f"{problem.window_bins}",
        f"Priors{Band}HighMhz": f"{problem.f0_range[1] * 1e3:.3g}",
        f"Priors{Band}Windows": f"{hi - lo + 1}",
        f"Priors{Band}CatalogueShare": percent(band_mass / cut_mass),
        f"Priors{Band}Visible": percent(visible.mean()),
        f"Priors{Band}Empty": percent((visible.sum(1) == 0).mean()),
        f"Priors{Band}WithVisible": percent(visible.any(1).mean()),
        f"Priors{Band}TrainingWithVisible": percent(
            (cut["snr"][k] >= VISIBLE_SNR).any(1).mean()
        ),
        f"Priors{Band}TrainingTopShare": percent(top(cut["f"][k])),
        f"Priors{Band}DriftOver": percent((z["drift"] > 1).mean()),
        f"Priors{Band}DriftMedian": f"{np.median(z['drift']):.1g}",
        f"Priors{Band}SpanMax": f"{z['span'].max():.0f}",
        f"Priors{Band}TopShare": percent(top(z["f"])),
    }

# where the conditional mean amplitude reaches the SNR-7 amplitude of a sky- and orientation-averaged source
curves = prior_report["bands"]["full"]["curves"]
cross = np.flatnonzero(np.asarray(curves["amp_mean"]) >= np.asarray(curves["amp_snr7"]))
numbers["PriorsAmpCrossMhz"] = f"{10 ** curves['log_f0'][cross[0]] * 1e3:.2g}"

# the catalogue Gaussian in (log10 f0, log10 fdot, log10 A) and its conditionals on f0
for k, name in enumerate(("F", "Fdot", "Amp")):
    numbers |= {
        f"PriorsMean{name}": f"{prior['mean'][k]:.2f}",
        f"PriorsStd{name}": f"{prior['std'][k]:.2f}",
    }
for k, name in enumerate(("Fdot", "Amp")):
    numbers |= {
        f"PriorsSlope{name}": f"{prior['slope'][k]:.2f}",
        f"PriorsIntercept{name}": f"{prior['intercept'][k]:.2f}",
        f"PriorsResidual{name}": f"{prior['residual_std'][k]:.2f}",
    }
corr = np.asarray(prior["corr"])
numbers |= {
    "PriorsCorrFFdot": f"{corr[0, 1]:.2f}",
    "PriorsCorrFAmp": f"{corr[0, 2]:.2f}",
    "PriorsCorrFdotAmp": f"{corr[1, 2]:.2f}",
    "PriorsResidualCorr": f"{prior['residual_corr']:.2f}",
    "PriorsCatalogueCutMhz": f"{10 ** prior['cut'] * 1e3:g}",
    "PriorsVisibleSnr": f"{VISIBLE_SNR:.0f}",
    "PriorsPriorWindows": f"{prior_report['bands']['full']['windows']}",
    "PriorsCutWindows": f"{prior_report['bands']['full']['cut']['windows']}",
    "PriorsCandidates": f"{prior_report['bands']['full']['cut']['candidates']}",
}

# the galactic centre in the equatorial frame jaxgb reads
ra, dec = (math.degrees(float(a)) for a in galactic_to_equatorial(0.0, 0.0))

# the training distribution each band's run sees
draws = cfgs["full"].draws
observations = cfgs["full"].batch_size // draws
assert all(
    c.draws == draws and c.batch_size // c.draws == observations for c in cfgs.values()
)
for band, cfg in cfgs.items():
    Band = band.capitalize()
    windows = cfg.total_steps * observations
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

numbers |= {
    "PriorsSources": f"{n_sources}",
    "PriorsCoords": "8",
    "PriorsStride": f"{WINDOW_STRIDE}",
    "PriorsInteriorResponses": f"{WINDOW_RESPONSES - 1}",
    "PriorsBandLowMhz": f"{full.f0_range[0] * 1e3:.1f}",
    "PriorsTobsYears": f"{full.t_obs / (365.25 * 86400):.0f}",
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
    "PriorsPlaneShare": f"{100 * sky['full']['sky'][f'abs_lat_below_{PLANE_DEG}deg']['galactic']:.0f}",
    "PriorsPermutations": f"{math.factorial(n_sources)}",
    "PriorsCopies": f"{math.factorial(n_sources) * 4**n_sources}",
    "PriorsBatch": f"{cfgs['full'].batch_size}",
    "PriorsDraws": f"{draws}",
    "PriorsObservationsPerStep": f"{observations}",
}


for k, v in numbers.items():
    print(f"{k} = {v}")
style.save_numbers("priors", numbers)
