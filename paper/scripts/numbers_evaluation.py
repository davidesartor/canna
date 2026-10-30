"""The numbers in appendix/evaluation.tex: the eval, scorecard and validation settings, the test power and mock ladder of reports/posterior-eval, and the Fisher-forecast report."""

import ast
import json
import math

import numpy as np

import style

from canna.lisa import config, scorecard, validate
from canna.lisa.injections import N_FISHER_DRAWS, N_PER_WINDOW, N_POSTERIOR, N_WINDOWS
from canna.lisa.mcmc import FDOT_SIGMAS
from canna.lisa.physics import FDOT_STD, log10_fdot_mean

WORDS = "zero one two three four five six seven eight nine ten".split()

narrow, full = config.load("narrow-B"), config.load("full-B")
for field in ("seed", "ode_steps", "n_random", "n_validation", "weights"):
    assert getattr(narrow, field) == getattr(full, field), field
for field in ("n_sources", "t_obs"):
    assert narrow.problem[field] == full.problem[field], field
assert scorecard.SNR_BANDS == validate.SNR_BANDS
cfg, problem = full, full.problem
n_sources, t_obs = problem["n_sources"], problem["t_obs"]
# the visible sources' SNR span: 7, where reports/prior counts a source visible, to the full-B prior's 99.99% quantile
SNR_QUANTILE = 0.9999
prior_snr = np.load(style.REPO / "reports/prior/f0-fdot-amplitude/data/full.npz")["snr"]
snr_lo, snr_hi = 7.0, float(np.quantile(prior_snr, SNR_QUANTILE))
ideal = lambda snr: math.sqrt(3) / (math.pi * snr)
windows = N_WINDOWS * N_PER_WINDOW + cfg.n_random

# the narrow band's largest chirp drift over the mission, in bins: top f0, fdot at the prior's ceiling
f0_top = narrow.problem["f0_range"][1]
drift = 10 ** float(log10_fdot_mean(f0_top) + FDOT_SIGMAS * FDOT_STD) * t_obs**2

with open(style.REPO / "reports/mcmc/data/fisher-diagonal.json") as file:
    fisher = json.load(file)
assert fisher["fisher_draws"] == N_FISHER_DRAWS
bands = fisher["bands"]
injections = [i for b in bands.values() for i in b["injections"]]
spans = [i["precision_decades"] for i in injections]
hessian = np.array([i["log10_hessian_diag"] for i in injections]).reshape(
    len(injections), n_sources, -1
)
mc_negative = np.isnan(hessian[..., 1])
ratio_by_band = {
    band: np.concatenate(
        [np.divide(i["f0_width_bins"], i["ideal_width_bins"]) for i in b["injections"]]
    )
    for band, b in bands.items()
}
ratio = np.concatenate(list(ratio_by_band.values()))
floored = np.array([i["floored_eigenvalues"] for i in injections])
window_snr = {
    band: [b["injections"][0]["window_snr"], b["injections"][-1]["window_snr"]]
    for band, b in bands.items()
}
source_snr = np.concatenate([i["source_snr"] for i in injections])
print(
    "f0 Fisher / ideal: median",
    np.median(ratio),
    "range",
    ratio.min(),
    ratio.max(),
    "floored",
    floored,
)

# f0 at the start of the observation (jaxgb's t0 = 0): its width with a free quadratic phase over the centred-origin one
gram = lambda n: np.linalg.inv(1 / (np.arange(n)[:, None] + np.arange(n) + 1))
start_factor = math.sqrt(gram(3)[1, 1] / gram(2)[1, 1])
# the drift in bins resolvable with f0 free: the quadratic-phase coefficient's width on a centred origin
drift_resolution = math.sqrt(1 / (1 / 80 - 1 / 144)) / math.pi

# size and power of the validation tests on the exact-posterior toy, per injected error
toy = json.loads(
    (style.REPO / "reports/posterior-eval/test-power/data/toy.json").read_text()
)
coords = ["f0", "mc", "lon", "lat", "re_zp", "im_zp", "re_zm", "im_zm"]
errors = {
    "Null": "null",
    "Phase": "z+ phase from prior",
    "Sky": "sky from prior",
    "Shuffle": "dependence shuffled",
    "Mixed": "10% prior draws",
    "Narrow": "narrow ×0.8",
    "Wide": "wide ×1.25",
    "Shift": "f0 shift 0.5σ",
}
power = {}
for row, error in errors.items():
    rates = toy["power"][error]
    coord_rates = [rates[f"slot {c}"] for c in coords]
    power |= {
        f"EvaluationPower{row}Coord": f"{min(coord_rates):.2f}--{max(coord_rates):.2f}",
        f"EvaluationPower{row}Overlap": f"{rates['slot overlap']:.2f}",
        f"EvaluationPower{row}Snr": f"{rates['slot own_snr']:.2f}",
        f"EvaluationPower{row}Like": f"{rates['joint log_likelihood']:.2f}",
        f"EvaluationPower{row}Energy": f"{rates['score joint_energy']:.2f}",
    }
# a score has no null of its own: its null rate is undefined
power["EvaluationPowerNullEnergy"] = "--"

# the multivariate rank tests over a window: TARP, band depth, average rank
window = lambda test, rows: [toy["power"][errors[row]][f"joint {test}"] for row in rows]
widths, marginal_exact = ["Narrow", "Wide"], ["Phase", "Sky", "Shuffle"]
power |= {
    "EvaluationPowerTarpWidth": f"{min(window('tarp', widths)):.2f}--{max(window('tarp', widths)):.2f}",
    "EvaluationPowerTarpBlind": f"{max(window('tarp', marginal_exact)):.2f}",
    "EvaluationPowerDepthShuffle": f"{window('band_depth', ['Shuffle'])[0]:.2f}",
    "EvaluationPowerDepthMixed": f"{window('band_depth', ['Mixed'])[0]:.2f}",
    "EvaluationPowerAverageShift": f"{window('average_rank', ['Shift'])[0]:.2f}",
    "EvaluationPowerAverageWidth": f"{max(window('average_rank', widths)):.2f}",
}

# the mock ladder of reports/posterior-eval: known posteriors through the scorecard, validation and reference comparison
EVAL = style.REPO / "reports/posterior-eval"
read = lambda path: json.loads((EVAL / path).read_text())
MOCKS = {
    "Fisher": "fisher",
    "Wide": "wide",
    "Narrow": "narrow",
    "Biased": "biased",
    "Prior": "prior",
    "Phases": "phases-prior",
    "Floor": "loud-floor",
}
# the ladder's construction constants, the mock script's top-level literals
constants = {}
for node in ast.parse((EVAL / "mock-flows/scripts/mock_eval.py").read_text()).body:
    if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Constant, ast.Tuple)):
        target = node.targets[0]
        names = target.elts if isinstance(target, ast.Tuple) else [target]
        values = ast.literal_eval(node.value)
        constants |= dict(zip([n.id for n in names], values if len(names) > 1 else [values]))
assert tuple(MOCKS.values()) == constants["MOCKS"]
p_value = lambda p: f"{p:.3f}" if p >= 0.001 else "$<$0.001"
ratio_text = lambda r: (
    f"\\num{{{r:.1e}}}".replace("e+0", "e")
    if r >= 1000
    else f"{r:.2f}" if r < 10 else f"{r:.1f}" if r < 100 else f"{r:.0f}"
)
loud = lambda bands: next(
    b["ratio"]
    for b in bands
    if b["subset"] == "all" and b["lo"] == scorecard.SNR_BANDS[3][0]
)
dataspace = {
    band: {c["name"]: c for c in read(f"data-space-checks/data/mocks-{band}.json")["cases"]}
    for band in ("narrow", "full")
}
proper = read("proper-scores/data/mocks-narrow.json")
ladder = {}
for key, mock in MOCKS.items():
    card = read(f"fisher-scorecard/data/mock-narrow-{mock}.json")
    tests = dataspace["narrow"][mock]["ladder"]
    reference = read(f"reference-comparison/data/mock-{mock}.json")
    assert reference["config"] == "narrow-B"
    # median flow/MCMC width per parameter over the per-source chains, over the eight physical parameters
    sources = reference["widths"]["sources"]
    medians = [np.median([s["flow"][j] / s["mcmc"][j] for s in sources]) for j in range(8)]
    ladder |= {
        f"EvaluationMock{key}Ideal": ratio_text(loud(card["bands"])),
        f"EvaluationMock{key}Cover": f"{np.mean(card['rows']['in68']):.2f}",
        f"EvaluationMock{key}Coord": p_value(tests["coords_min"]),
        f"EvaluationMock{key}Tarp": p_value(tests["tarp"]),
        f"EvaluationMock{key}Overlap": p_value(tests["overlap"]),
        f"EvaluationMock{key}Like": p_value(tests["log_likelihood"]),
        f"EvaluationMock{key}Energy": f"{proper['ratios'][mock]['energy']['ratio']:.2f}",
        f"EvaluationMock{key}Mcmc": f"{ratio_text(min(medians))}--{ratio_text(max(medians))}",
    }

# the loud floor reads conservative by coverage: in68 below and above the second SNR band edge
floor_rows = read("fisher-scorecard/data/mock-narrow-loud-floor.json")["rows"]
floor_in68 = np.array(floor_rows["in68"])
quiet = np.array(floor_rows["snr"]) < scorecard.SNR_BANDS[2][0]
# its coordinate ranks in the loud strata (Mc excepted: the prior width caps its floor)
floor_cal = read("calibration/data/narrow-loud-floor.json")
loud_strata = floor_cal["strata"][2:4]
assert loud_strata == ["40-100", ">=100"]
floor_loud_p = max(
    floor_cal["cells"][s][c]["pooled"]["p"] for s in loud_strata for c in coords if c != "mc"
)
ladder |= {
    "EvaluationMockWide": f"{constants['WIDE']:g}",
    "EvaluationMockNarrow": f"{constants['NARROW']:g}",
    "EvaluationMockFloorBins": f"{constants['FLOOR_BINS']:g}",
    "EvaluationMockFloorSnr": f"{math.sqrt(3) / (math.pi * constants['FLOOR_BINS']):.1f}",
    "EvaluationMockInjections": f"{dataspace['narrow']['fisher']['n']}",
    "EvaluationMockInjectionsFull": f"{dataspace['full']['fisher']['n']}",
    "EvaluationMockScorecardSources": f"{card['n_rows']}",
    "EvaluationMockReferenceSources": f"{len(sources)}",
    "EvaluationMockFisherIdealFull": f"{loud(read('fisher-scorecard/data/mock-full-fisher.json')['bands']):.2f}",
    "EvaluationMockPriorCoverFull": f"{np.mean(read('fisher-scorecard/data/mock-full-prior.json')['rows']['in68']):.2f}",
    "EvaluationMockFloorCoverQuiet": f"{floor_in68[quiet].mean():.2f}",
    "EvaluationMockFloorCoverLoud": f"{floor_in68[~quiet].mean():.2f}",
    "EvaluationMockFloorLoudP": p_value(floor_loud_p),
}
for name, value in ladder.items():
    print(name, value)

exponent = math.floor(math.log10(ideal(snr_hi)))
numbers = {
    # the eval injections
    "EvaluationFrequenciesWord": WORDS[N_WINDOWS],
    "EvaluationPerFrequencyWord": WORDS[N_PER_WINDOW],
    "EvaluationInjections": f"{N_WINDOWS * N_PER_WINDOW}",
    "EvaluationSourcesBoth": f"{len(source_snr)}",
    "EvaluationNarrowSnrLo": f"{window_snr['narrow'][0]:.0f}",
    "EvaluationNarrowSnrHi": f"{window_snr['narrow'][1]:.0f}",
    "EvaluationFullSnrLo": f"{window_snr['full'][0]:.0f}",
    "EvaluationFullSnrHi": f"{window_snr['full'][1]:.0f}",
    "EvaluationSourceSnrLo": f"{source_snr.min():.1f}",
    "EvaluationSourceSnrHi": f"{source_snr.max():.0f}",
    "EvaluationPosterior": f"{N_POSTERIOR}",
    "EvaluationOdeSteps": f"{cfg.ode_steps}",
    "EvaluationPermutations": f"{math.factorial(n_sources)}",
    "EvaluationNSources": WORDS[n_sources],
    # the scorecard
    "EvaluationRandom": f"{cfg.n_random}",
    "EvaluationWindows": f"{windows}",
    "EvaluationWindowSources": f"{windows * n_sources}",
    "EvaluationDraws": f"{scorecard.N_DRAWS}",
    "EvaluationFound": f"{scorecard.FOUND_BINS:g}",
    "EvaluationBandOne": f"{scorecard.SNR_BANDS[1][0]:g}",
    "EvaluationBandTwo": f"{scorecard.SNR_BANDS[2][0]:g}",
    "EvaluationBandThree": f"{scorecard.SNR_BANDS[3][0]:g}",
    "EvaluationSnrLo": f"{snr_lo:g}",
    "EvaluationSnrHi": f"{snr_hi:.0f}",
    "EvaluationSnrQuantile": f"{100 * SNR_QUANTILE:g}",
    "EvaluationIdealLo": f"{ideal(snr_lo):.2f}",
    "EvaluationIdealHi": f"{ideal(snr_hi) / 10**exponent:.1f}e{exponent}",
    # the validation
    "EvaluationValidation": f"{cfg.n_validation}",
    "EvaluationValidationDraws": f"{validate.N_DRAWS}",
    "EvaluationRankValues": f"{validate.N_DRAWS + 1}",
    "EvaluationGammaSims": f"{validate.GAMMA_SIMS}",
    "EvaluationBandLevel": f"{100 * (1 - validate.ALPHA):g}",
    "EvaluationJointDim": f"{n_sources * len(validate.COORDS)}",
    # size and power of the validation tests
    "EvaluationPowerInjections": f"{toy['injections']:,}".replace(",", "\\,"),
    "EvaluationPowerDraws": f"{toy['draws']}",
    "EvaluationPowerChunk": f"{toy['chunk']}",
    "EvaluationPowerAlpha": f"{100 * toy['alpha']:g}",
    "EvaluationPowerMixShare": errors["Mixed"].split("%")[0],
    **power,
    # the mock ladder
    **ladder,
    # the Fisher forecast
    "EvaluationFisherDraws": f"{N_FISHER_DRAWS}",
    "EvaluationFisherDecades": f"{fisher['median_precision_decades']:.0f}",
    "EvaluationFisherDecadesLo": f"{min(spans):.0f}",
    "EvaluationFisherDecadesHi": f"{max(spans):.0f}",
    "EvaluationFisherInjections": f"{len(injections)}",
    "EvaluationMcNegative": f"{mc_negative.sum()}",
    "EvaluationMcNegativeInjections": f"{mc_negative.any(-1).sum()}",
    "EvaluationFlooredInjections": f"{(floored > 0).sum()}",
    "EvaluationFlooredMax": f"{floored.max()}",
    "EvaluationFisherRatioMedian": f"{np.median(ratio):.2f}",
    "EvaluationFisherRatioLo": f"{ratio.min():.2f}",
    "EvaluationFisherRatioHi": f"{ratio.max():.1f}",
    "EvaluationFisherRatioNarrow": f"{np.median(ratio_by_band['narrow']):.2f}",
    "EvaluationFisherRatioNarrowHi": f"{ratio_by_band['narrow'].max():.1f}",
    "EvaluationFisherRatioFull": f"{np.median(ratio_by_band['full']):.1f}",
    "EvaluationStartFactor": f"{start_factor:.0f}",
    "EvaluationDriftResolution": f"{drift_resolution:.1f}",
    "EvaluationNarrowFzeroTop": f"{f0_top * 1e3:.2f}",
    "EvaluationFdotSigmas": f"{FDOT_SIGMAS:g}",
    "EvaluationChirpDrift": f"{drift:.2f}",
}
style.save_numbers("evaluation", numbers)
