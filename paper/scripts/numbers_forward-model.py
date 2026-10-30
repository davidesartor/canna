"""The numbers in appendix/forward-model.tex, from the lisa problem, physics, configs and the forward-model and response-span reports."""

import json
import math

import numpy as np

import style

from canna.lisa import config
from canna.lisa.physics import (
    ANTENNA_HARMONICS,
    EARTH_ORBIT_SPEED,
    SPEED_OF_LIGHT,
    SOURCE_INTERCEPT,
    SOURCE_SLOPE,
    YEAR,
    response_span,
)
from canna.lisa.problem import WINDOW_RESPONSES, WINDOW_STRIDE, LisaGB

WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight")
ORDINALS = ("zeroth", "first", "second", "third", "fourth", "fifth", "sixth")
REPORT = style.REPO / "reports/diagnostics/forward-model/data"
SPAN = style.REPO / "reports/prior/response-span/data"
F0_BINS = 20


def sci(x: float, digits: int = 1) -> str:
    """x as a.be<exponent>, for siunitx."""
    if x == 0:
        return "0"
    exponent = math.floor(math.log10(abs(x)))
    mantissa = x / 10**exponent
    if round(mantissa, digits) >= 10:
        mantissa, exponent = mantissa / 10, exponent + 1
    return f"{mantissa:.{digits}f}e{exponent}"


def two_figures(x: float) -> str:
    """x to two significant figures, without an exponent."""
    return f"{x:.{max(0, 1 - math.floor(math.log10(abs(x))))}f}"


def percent(x: float) -> str:
    """x in percent, to two significant figures."""
    return two_figures(100 * x)


def pow2(q: float) -> int:
    """The smallest power of two holding q bins."""
    return 1 << max(1, math.ceil(math.log2(max(q, 2))))


def quantile(a: np.ndarray, w: np.ndarray, y: float) -> float:
    """The smallest value with at least a share y of the weight w at or below it."""
    order = np.argsort(a, kind="stable")
    cumulative = np.cumsum(w[order])
    at = np.searchsorted(cumulative, y * cumulative[-1] * (1 - 1e-12))
    return float(a[order][min(at, a.size - 1)])


bands = {name: config.load(f"{name}-B").problem for name in ("narrow", "full")}
problems = {name: LisaGB(**cfg) for name, cfg in bands.items()}
full = problems["full"]
t_obs = full.t_obs
velocity = EARTH_ORBIT_SPEED / SPEED_OF_LIGHT
arm_seconds = full.response.arm_length / SPEED_OF_LIGHT
f_lowest = min(p.f0_range[0] for p in problems.values())
f_highest = max(p.f0_range[1] for p in problems.values())

# each term of the half-span at a frequency, in bins, for the prior's mean fdot there
fdot_mean = lambda f: 10 ** (SOURCE_INTERCEPT[0] + SOURCE_SLOPE[0] * math.log10(f))
doppler = lambda f: f * velocity * t_obs
drift = lambda f: fdot_mean(f) * t_obs**2
antenna = ANTENNA_HARMONICS * t_obs / YEAR

# the mean drift overtakes the Doppler sideband where fdot T^2 = f v T, fdot going as f^slope
f_cross = (velocity / (10 ** SOURCE_INTERCEPT[0] * t_obs)) ** (
    1 / (SOURCE_SLOPE[0] - 1)
)
assert f_cross > f_highest, "the text has the crossing beyond both bands"

numbers = {
    "ForwardModelVelocity": f"{EARTH_ORBIT_SPEED / 1e3:.1f}",
    "ForwardModelVelocityRatio": sci(velocity),
    "ForwardModelArmGm": f"{full.response.arm_length / 1e9:.1f}",
    "ForwardModelArmSeconds": f"{arm_seconds:.2f}",
    "ForwardModelFStar": f"{1e3 / (2 * math.pi * arm_seconds):.1f}",
    "ForwardModelTobsYears": f"{t_obs / YEAR:g}",
    "ForwardModelBinNanoHz": f"{1e9 / t_obs:.1f}",
    "ForwardModelSources": f"{full.n_sources}",
    "ForwardModelSourcesWord": WORDS[full.n_sources],
    "ForwardModelWindowResponses": f"{WINDOW_RESPONSES}",
    "ForwardModelInteriorResponses": f"{WINDOW_RESPONSES - 1}",
    "ForwardModelWindowStride": f"{WINDOW_STRIDE}",
    "ForwardModelAntennaHarmonics": f"{ANTENNA_HARMONICS}",
    "ForwardModelAntennaOrdinal": ORDINALS[ANTENNA_HARMONICS],
    "ForwardModelAntennaBins": f"{antenna:.0f}",
    "ForwardModelChirpCross": f"{f_cross * 1e3:.2g}",
    "ForwardModelFdotSlope": f"{SOURCE_SLOPE[0]:.2f}",
    "ForwardModelBandLow": f"{f_lowest * 1e3:g}",
    "ForwardModelBandHigh": f"{f_highest * 1e3:g}",
}

# per band: the window, the mean-fdot span at its top, its share of the log band
for name, problem in problems.items():
    Name = name.capitalize()
    top = problem.f0_range[1]
    lo, hi = problem.window_index_range
    numbers |= {
        f"ForwardModel{Name}Low": f"{problem.f0_range[0] * 1e3:g}",
        f"ForwardModel{Name}Top": f"{top * 1e3:.4g}",
        f"ForwardModel{Name}Doppler": f"{doppler(top):.0f}",
        f"ForwardModel{Name}Chirp": (
            f"{drift(top):.0f}" if drift(top) >= 1 else f"{drift(top):.2f}"
        ),
        f"ForwardModel{Name}Span": f"{response_span(top, fdot_mean(top), t_obs)}",
        f"ForwardModel{Name}R": f"{problem.response_points}",
        f"ForwardModel{Name}K": f"{problem.window_bins}",
        f"ForwardModel{Name}Interior": f"{(WINDOW_RESPONSES - 1) * problem.response_points}",
        f"ForwardModel{Name}Windows": f"{hi - lo + 1}",
        f"ForwardModel{Name}Fraction": f"{100 * math.log(top / problem.f0_range[0]) / math.log(f_highest / f_lowest):.0f}",
    }

# R(x, y): the smallest R keeping a share x of a source's above-noise power for a share y of the sources the data see
span = json.loads((SPAN / "response-span.json").read_text())
z = np.load(SPAN / "response-span.npz")
assert span["response_points"] == full.response_points
assert span["narrow_response_points"] == problems["narrow"].response_points
assert span["narrow_f0_max"] == problems["narrow"].f0_range[1]
draws = span["amplitudes"]
keep = span["keep"]
source_w = z["f0_weight"] * z["seen"]
w = np.repeat(source_w / draws, draws)
width = z["width"].reshape(w.size, -1)
need = lambda x, y, rows=slice(None): pow2(
    quantile(width[rows, keep.index(x)], w[rows], y)
)
assert need(0.99, 0.99) == need(1.0, 0.999) == full.response_points
narrow = np.repeat(z["f0"] < span["narrow_f0_max"], draws)
assert need(1.0, 0.999, narrow) <= problems["narrow"].response_points

# the lowest f0 bin from which keeping 99% for 99% of its sources needs more than full-B's R
edges = np.linspace(*np.log10(span["f0_range"]), F0_BINS + 1)
bins = np.repeat(
    np.clip(np.digitize(np.log10(z["f0"]), edges) - 1, 0, F0_BINS - 1), draws
)
wider = [
    b
    for b in range(F0_BINS)
    if w[bins == b].sum() > 0 and need(0.99, 0.99, bins == b) > full.response_points
]
numbers |= {
    "ForwardModelSpanSources": f"{span['sources']}",
    "ForwardModelSpanSeen": percent(source_w.mean()),
    "ForwardModelSpanNeed": f"{need(0.99, 0.99)}",
    "ForwardModelSpanNeedAll": f"{need(1.0, 0.999)}",
    "ForwardModelSpanMedian": f"{quantile(width[:, keep.index(1.0)], w, 0.5):.0f}",
    "ForwardModelSpanTail": f"{quantile(width[:, keep.index(1.0)], w, 0.9999):.0f}",
    "ForwardModelSpanWiderMhz": f"{10 ** edges[wider[0]] * 1e3:.2g}",
    "ForwardModelSpanWiderShare": percent(w[bins >= wider[0]].sum() / w.sum()),
    "ForwardModelSpanBound": f"{span['bound_points']}",
    "ForwardModelNarrowSpanShare": percent(w[narrow].sum() / w.sum()),
}

# each band's R-bin response against a long reference, as reports/diagnostics/forward-model measured it
points = json.loads((REPORT / "response-points.json").read_text())
numbers["ForwardModelReferencePoints"] = f"{points['reference_points']}"
for name in problems:
    Name = name.capitalize()
    assert points[f"{name}_response_points"] == problems[name].response_points
    numbers |= {
        f"ForwardModel{Name}TruncSources": f"{points[f'{name}_sources']}",
        f"ForwardModel{Name}TruncRelMedian": percent(points[f"{name}_rel_median"]),
        f"ForwardModel{Name}TruncRelMax": percent(points[f"{name}_rel_max"]),
        f"ForwardModel{Name}TruncRelOnBin": percent(
            points[f"{name}_rel_median_on_bin"]
        ),
        f"ForwardModel{Name}TruncRelOffBin": percent(
            points[f"{name}_rel_median_off_bin"]
        ),
        f"ForwardModel{Name}TruncOutsideMax": percent(points[f"{name}_outside_max"]),
        f"ForwardModel{Name}TruncRhoResMedian": two_figures(
            points[f"{name}_rho_res_median"]
        ),
        f"ForwardModel{Name}TruncRhoResOverOne": f"{100 * points[f'{name}_rho_res_over_1']:.0f}",
        f"ForwardModel{Name}LadderSlope": f"{points[f'{name}_ladder_slope']:.2f}",
        f"ForwardModel{Name}TruncRhoDev": sci(points[f"{name}_rho_ratio_max_dev"]),
    }

style.save_numbers("forward-model", numbers)
