"""The numbers in appendix/evaluation.tex: the eval, scorecard and validation settings, and the Fisher-forecast report."""

import json
import math

import numpy as np

import style

from canna.lisa import config, scorecard, validate
from canna.lisa.injections import N_CANDIDATES, N_FISHER_DRAWS, N_POSTERIOR, N_QUANTILES
from canna.lisa.physics import chirp_mass_support, fdot_from_chirp_mass

WORDS = "zero one two three four five six seven eight nine ten".split()

narrow, full = config.load("narrow-B"), config.load("full-B")
for field in ("seed", "ode_steps", "n_random", "n_validation", "weights"):
    assert getattr(narrow, field) == getattr(full, field), field
for field in ("n_sources", "t_obs", "snr_range", "mass_range"):
    assert narrow.problem[field] == full.problem[field], field
assert scorecard.SNR_BANDS == validate.SNR_BANDS
cfg, problem = full, full.problem
n_sources, t_obs = problem["n_sources"], problem["t_obs"]
snr_lo, snr_hi = problem["snr_range"]
ideal = lambda snr: math.sqrt(3) / (math.pi * snr)
quantiles = np.linspace(1 / N_QUANTILES, 1, N_QUANTILES)
windows = N_QUANTILES + cfg.n_random

# the narrow band's largest chirp drift over the mission, in bins: top f0, top chirp mass
mc_top = chirp_mass_support(tuple(narrow.problem["mass_range"]))[1]
f0_top = narrow.problem["f0_range"][1]
drift = float(fdot_from_chirp_mass(mc_top, f0_top)) * t_obs**2

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
    (style.REPO / "reports/diagnostics/posterior-validation/data/toy.json").read_text()
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
# a score has no null of its own: its null rate is zero by construction
power["EvaluationPowerNullEnergy"] = "--"

exponent = math.floor(math.log10(ideal(snr_hi)))
numbers = {
    # the eval injections
    "EvaluationCandidates": f"{N_CANDIDATES}",
    "EvaluationQuantilesWord": WORDS[N_QUANTILES],
    "EvaluationQuantiles": f"{quantiles[0]:.1f}, {quantiles[1]:.1f}, \\dots, {quantiles[-1]:.1f}",
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
    "EvaluationSnrHi": f"{snr_hi:g}",
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
    **power,
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
    "EvaluationMcTop": f"{mc_top:.2f}",
    "EvaluationChirpDrift": f"{drift:.2f}",
}
style.save_numbers("evaluation", numbers)
