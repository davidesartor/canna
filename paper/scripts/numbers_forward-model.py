"""The numbers in appendix/forward-model.tex, from the lisa problem, physics, configs and the forward-model report."""

import json
import math

import style

from canna.lisa import config
from canna.lisa.physics import (
    ANTENNA_HARMONICS,
    EARTH_ORBIT_SPEED,
    SPEED_OF_LIGHT,
    YEAR,
    chirp_mass_support,
    fdot_from_chirp_mass,
    response_span,
)
from canna.lisa.problem import WINDOW_RESPONSES, WINDOW_STRIDE, LisaGB

WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight")
ORDINALS = ("zeroth", "first", "second", "third", "fourth", "fifth", "sixth")
REPORT = style.REPO / "reports/diagnostics/forward-model/data"


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


bands = {name: config.load(f"{name}-B").problem for name in ("narrow", "full")}
problems = {name: LisaGB(**cfg) for name, cfg in bands.items()}
full = problems["full"]
t_obs = full.t_obs
mc_hi = chirp_mass_support(full.mass_range)[1]
velocity = EARTH_ORBIT_SPEED / SPEED_OF_LIGHT
arm_seconds = full.response.arm_length / SPEED_OF_LIGHT
f_lowest = min(p.f0_range[0] for p in problems.values())
f_highest = max(p.f0_range[1] for p in problems.values())

# each term of the heaviest binary's half-span at a frequency, in bins
doppler = lambda f: f * velocity * t_obs
drift = lambda f: float(fdot_from_chirp_mass(mc_hi, f)) * t_obs**2
antenna = ANTENNA_HARMONICS * t_obs / YEAR

# the chirp overtakes the Doppler sideband where fdot T^2 = f v T, fdot going as f^(11/3)
f_cross = (velocity / (float(fdot_from_chirp_mass(mc_hi, 1.0)) * t_obs)) ** (3 / 8)

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
    "ForwardModelMassMax": f"{full.mass_range[1]:g}",
    "ForwardModelMcMax": f"{mc_hi:.3f}",
    "ForwardModelChirpCross": f"{f_cross * 1e3:.0f}",
    "ForwardModelQuadruple": f"{4 ** (3 / 11):.2f}",
    "ForwardModelBandLow": f"{f_lowest * 1e3:g}",
    "ForwardModelBandHigh": f"{f_highest * 1e3:g}",
}

# per band: the window, the span at its top, the highest f0 its R holds, its share of the log band
for name, problem in problems.items():
    Name = name.capitalize()
    top = problem.f0_range[1]
    holds = LisaGB(
        **{
            **bands[name],
            "f0_range": (1e-4, 1.0),
            "response_points": problem.response_points,
        }
    ).f0_range[1]
    lo, hi = problem.window_index_range
    numbers |= {
        f"ForwardModel{Name}Low": f"{problem.f0_range[0] * 1e3:g}",
        f"ForwardModel{Name}Top": f"{top * 1e3:.4g}",
        f"ForwardModel{Name}Doppler": f"{doppler(top):.0f}",
        f"ForwardModel{Name}Chirp": f"{drift(top):.0f}",
        f"ForwardModel{Name}Span": f"{response_span(top, mc_hi, t_obs)}",
        f"ForwardModel{Name}Holds": f"{holds * 1e3:.4g}",
        f"ForwardModel{Name}R": f"{problem.response_points}",
        f"ForwardModel{Name}K": f"{problem.window_bins}",
        f"ForwardModel{Name}Interior": f"{(WINDOW_RESPONSES - 1) * problem.response_points}",
        f"ForwardModel{Name}Windows": f"{hi - lo + 1}",
        f"ForwardModel{Name}Fraction": f"{100 * math.log(top / problem.f0_range[0]) / math.log(f_highest / f_lowest):.0f}",
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
    }

style.save_numbers("forward-model", numbers)
