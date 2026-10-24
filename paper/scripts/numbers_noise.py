"""The numbers in appendix/noise.tex, from the lisa problem and the prior, forward-model and strub-source reports."""

import json
import math

import numpy as np

import style

import jax.numpy as jnp
from canna.lisa.physics import SPEED_OF_LIGHT, YEAR
from canna.lisa import config
from canna.lisa.problem import (
    LisaGB,
    noise_psd,
    single_link_noise,
    sky_averaged_sensitivity,
)

WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight")


def sci(value: float, digits: int = 1) -> str:
    exponent = math.floor(math.log10(abs(value)))
    return f"{value / 10**exponent:.{digits}f}e{exponent}"


narrow = LisaGB(**config.load("narrow-B").problem)
full = LisaGB(**config.load("full-B").problem)
for field in ("t_obs", "oms_noise", "acceleration_noise", "n_sources", "snr_range"):
    assert getattr(narrow, field) == getattr(full, field), field
arm_length = full.response.arm_length
f_star = SPEED_OF_LIGHT / (2 * math.pi * arm_length)
f = jnp.geomspace(*full.f0_range, 4001)

# A, E, T from Babak et al.'s S_X (eq. 19) and S_XY (eq. 129 over the TDI-2 factor 4 sin^2 2x), against noise_psd
x = 2 * math.pi * f * arm_length / SPEED_OF_LIGHT
p_oms, p_acc = single_link_noise(full, f)
s_oms = p_oms * (2 * math.pi * f / SPEED_OF_LIGHT) ** 2
s_acc = p_acc / ((2 * math.pi * f) ** 2 * SPEED_OF_LIGHT**2)
s_x = 16 * jnp.sin(x) ** 2 * (s_oms + (3 + jnp.cos(2 * x)) * s_acc)
s_xy = -8 * jnp.sin(x) ** 2 * jnp.cos(x) * (s_oms + 4 * s_acc)
babak = jnp.stack([s_x - s_xy, s_x - s_xy, s_x + 2 * s_xy], axis=-1)
psd = noise_psd(full, f)
psd_gap = float(jnp.max(jnp.abs(psd / babak - 1)))

# Robson et al. eq. 13's acceleration factor 2 (1 + cos^2(f/f*)) against the 4 of their eq. 1, over the band
robson_13 = (
    (10 / (3 * arm_length**2))
    * (p_oms + 2 * (1 + jnp.cos(f / f_star) ** 2) * p_acc / (2 * math.pi * f) ** 4)
    * (1 + 0.6 * (f / f_star) ** 2)
)
robson_gap = float(jnp.max(jnp.abs(robson_13 / sky_averaged_sensitivity(full, f) - 1)))

# T against A/E noise at the band's ends
t_over_a = np.asarray(psd[:, 2] / psd[:, 0])

# C and its drift with f0, isotropic sky, as reports/prior measured it
with open(style.REPO / "reports/prior/data/prior.json") as file:
    sky_average = json.load(file)["bands"]["full"]["sky_average"]
isotropic, galactic = sky_average["isotropic"], sky_average["galactic"]
f0_grid, ratio = np.array(sky_average["f0"]), np.array(isotropic["ratio"])
flat = f0_grid < sky_average["flat_below"]

# units, TDI and noise-generator checks, as reports/diagnostics/forward-model measured them
with open(
    style.REPO / "reports/diagnostics/forward-model/data/forward-model.json"
) as file:
    forward = json.load(file)

# SNR and MCMC checks on the Strub et al. source, as reports/diagnostics/strub-source measured them
with open(
    style.REPO / "reports/diagnostics/strub-source/data/strub_source.json"
) as file:
    strub = json.load(file)
pulls = {
    key.removeprefix("noiseless_pull_"): value
    for key, value in strub.items()
    if key.startswith("noiseless_pull_")
}
prior_pulled = ["log10_A", "cos_iota"]
response_points_checked = [
    int(key.removeprefix("snr_rp")) for key in strub if key.startswith("snr_rp")
] + [strub["full_b_response_points"]]
strub_cos_half_width = (
    math.cos(strub["strub_iota_median"] - strub["strub_iota_minus"])
    - math.cos(strub["strub_iota_median"] + strub["strub_iota_plus"])
) / 2
sky_ratio_below_mhz = float(
    next(k for k in forward if k.startswith("sky_ratio_min_below_"))
    .removeprefix("sky_ratio_min_below_")
    .removesuffix("mHz")
)
snr_window_spread = max(
    abs(strub[key] / strub["snr"] - 1)
    for key in strub
    if key.startswith("snr_rp") or key == "snr_full_b"
)

numbers = {
    # instrument and band
    "NoiseOms": f"{full.oms_noise:g}",
    "NoiseAcc": f"{full.acceleration_noise:g}",
    "NoiseArmGm": f"{arm_length / 1e9:.1f}",
    "NoiseArmSeconds": f"{arm_length / SPEED_OF_LIGHT:.2f}",
    "NoiseTransfer": f"{f_star * 1e3:.2f}",
    "NoiseXTop": f"{2 * math.pi * full.f0_range[1] * arm_length / SPEED_OF_LIGHT:.2f}",
    "NoiseTobsYears": f"{full.t_obs / YEAR:g}",
    "NoiseSourcesWord": WORDS[full.n_sources],
    "NoiseBandLow": f"{full.f0_range[0] * 1e3:g}",
    "NoiseBandHigh": f"{full.f0_range[1] * 1e3:g}",
    "NoiseNarrowHigh": f"{narrow.f0_range[1] * 1e3:.2f}",
    "NoiseKNarrow": f"{narrow.window_bins}",
    "NoiseKFull": f"{full.window_bins}",
    "NoiseInputsNarrow": f"{6 * narrow.window_bins}",
    "NoiseInputsFull": f"{6 * full.window_bins}",
    "NoiseSnrLow": f"{full.snr_range[0]:g}",
    "NoiseSnrHigh": f"{full.snr_range[1]:g}",
    "NoisePsdGap": sci(psd_gap),
    "NoiseTOverALow": sci(float(t_over_a[0])),
    "NoiseTOverAHigh": f"{float(t_over_a[-1]):.2f}",
    # sky-averaged sensitivity and the chart's amplitude unit
    "NoiseRobsonPower": f"{100 * robson_gap:.1f}",
    "NoiseRobsonAmplitude": f"{100 * (math.sqrt(1 + robson_gap) - 1):.1f}",
    "NoiseC": f"{isotropic['C_low']:.2f}",
    "NoiseCError": f"{isotropic['C_low_se']:.2f}",
    "NoiseCGalactic": f"{galactic['C_low']:.2f}",
    "NoiseCFlat": f"{math.ceil(1000 * np.abs(ratio[flat] - 1).max()) / 10:.1f}",
    "NoiseCFlatBelow": f"{sky_average['flat_below'] * 1e3:.1f}",
    "NoiseCRise": f"{100 * (ratio[-1] - 1):.0f}",
    "NoiseCTop": f"{f0_grid[-1] * 1e3:g}",
    "NoiseCSources": f"{sky_average['sources']}",
    "NoiseSqrtC": f"{math.sqrt(isotropic['C_low']):.1f}",
    # units
    "NoiseSkyDraws": f"{forward['sky_draws']}",
    "NoiseSkySlope": f"{forward['sky_slope']:.4f}",
    "NoiseSkyFitLow": f"{forward['sky_grid_lo_hz'] * 1e3:.1f}",
    "NoiseSkyFitTop": f"{forward['sky_fit_hi_hz'] * 1e3:.1f}",
    "NoiseSkyRatioLow": f"{forward['sky_ratio_min_below_4mHz']:.3f}",
    "NoiseSkyRatioHigh": f"{forward['sky_ratio_max_below_4mHz']:.3f}",
    "NoiseSkySlopeLength": f"{forward['sky_slope_length']:.2f}",
    # noise generator
    "NoiseGeneratorWindows": f"{forward['noise_draws']}",
    "NoiseGeneratorBins": f"{forward['noise_window_bins']}",
    "NoiseGeneratorAbs": f"{forward['noise_abs2']:.4f}",
    "NoiseGeneratorVarRe": f"{forward['noise_var_re']:.4f}",
    "NoiseGeneratorVarIm": f"{forward['noise_var_im']:.4f}",
    "NoiseGeneratorCorr": sci(forward["noise_corr_re_im"]),
    "NoiseGeneratorCorrMax": sci(
        max(
            abs(forward[k])
            for k in ("noise_corr_adjacent", "noise_corr_a_e", "noise_corr_a_t")
        )
    ),
    "NoiseGeneratorCorrFloor": sci(forward["noise_corr_floor"]),
    "NoiseGeneratorTilt": f"{max(forward[f'noise_psd_tilt_{c}'] for c in 'aet'):.4f}",
    "NoiseGeneratorTiltFloor": f"{forward['noise_psd_tilt_floor']:.4f}",
    "NoiseGeneratorDof": f"{forward['noise_nn_dof']}",
    "NoiseGeneratorNnMean": f"{forward['noise_nn_mean']:.1f}",
    "NoiseGeneratorNnVar": f"{forward['noise_nn_var']:.0f}",
    "NoiseGeneratorNnVarExpected": f"{2 * forward['noise_nn_dof']}",
    "NoiseGeneratorKs": f"{forward['noise_nn_ks_p']:.2f}",
    # TDI convention
    "NoiseTdiSources": f"{forward['tdi_sources']}",
    "NoiseTdiLow": sci(forward["tdi_rel_max_low"]),
    "NoiseTdiLowTop": f"{forward['tdi_low_hz'] * 1e3:g}",
    "NoiseTdiHigh": sci(forward["tdi_rel_median_high"]),
    "NoiseTdiHighMax": sci(forward["tdi_rel_max_high"]),
    "NoiseTdiHighLo": f"{forward['tdi_high_lo_hz'] * 1e3:g}",
    "NoiseTdiHighHi": f"{forward['tdi_high_hi_hz'] * 1e3:g}",
    "NoiseTdiSlope": f"{forward['tdi_rel_loglog_slope']:.2f}",
    "NoiseTdiPowerMax": f"{100 * forward['tdi_power_dev_max']:.1f}",
    # the Strub et al. source
    "NoiseStrubSnr": f"{strub['snr']:.3f}",
    "NoiseStrubSnrTheirs": f"{strub['snr_strub']:g}",
    "NoiseStrubSnrGap": f"{100 * strub['snr_relative_gap']:.2f}",
    "NoiseStrubSnrUnrotated": f"{strub['snr_unrotated_rp64']:.3f}",
    "NoiseStrubRLow": f"{min(response_points_checked)}",
    "NoiseStrubRHigh": f"{max(response_points_checked)}",
    "NoiseStrubWindowSpread": sci(snr_window_spread),
    "NoiseStrubNoT": sci(abs(strub["snr_ae_rp64"] / strub["snr_rp64"] - 1), 0),
    "NoiseStrubSigmaFzero": f"{strub['seed0_sigma_f0_nhz']:.2f}",
    "NoiseStrubFzeroPlus": f"{strub['seed0_f0_plus_nhz']:.2f}",
    "NoiseStrubFzeroMinus": f"{strub['seed0_f0_minus_nhz']:.2f}",
    "NoiseStrubFzeroPlusTheirs": f"{strub['strub_f0_plus_nhz']:g}",
    "NoiseStrubFzeroMinusTheirs": f"{strub['strub_f0_minus_nhz']:g}",
    "NoiseStrubCorrFzeroFdot": f"{strub['seed0_corr_f0_log10_fdot']:.2f}",
    "NoiseStrubCorrCosIotaAmp": f"{strub['seed0_corr_cos_iota_log10_A']:.2f}",
    "NoiseStrubFdotPrior": f"{strub['seed0_log10_fdot_std_over_prior_std']:.2f}",
    "NoiseStrubBoxHalfWidth": f"{strub['box_cos_iota_half_width68']:.2f}",
    "NoiseStrubFullHalfWidth": f"{strub['seed0_cos_iota_half_width68']:.2f}",
    "NoiseStrubRealisations": f"{strub['sweep_count']}",
    "NoiseStrubGain": f"{strub['sweep_gain_obs_mean']:.2f}",
    "NoiseStrubGainError": f"{strub['sweep_gain_obs_sem']:.2f}",
    "NoiseStrubGainScatter": f"{strub['sweep_gain_obs_std']:.2f}",
    "NoiseStrubGainTheirs": f"{strub['strub_gain']:.2f}",
    "NoiseStrubRhat": f"{strub['noiseless_rhat_max']:.4f}",
    "NoiseStrubPullMax": f"{max(abs(v) for k, v in pulls.items() if k not in prior_pulled):.2f}",
    "NoiseStrubPullAmp": f"{pulls['log10_A']:+.1f}",
    "NoiseStrubPullCosIota": f"{pulls['cos_iota']:+.1f}",
    "NoiseStrubRhoMle": f"{strub['noiseless_rho_mle_refined']:.5f}",
    "NoiseStrubRhoOpt": f"{strub['noiseless_rho_opt']:.5f}",
    "NoiseStrubBoxFzeroPlus": f"{strub['box_f0_plus_nhz']:.2f}",
    "NoiseStrubBoxFzeroMinus": f"{strub['box_f0_minus_nhz']:.2f}",
    "NoiseStrubBoxSigmas": f"{strub['box_half_width_lon'] / strub['box_fisher_sigma_lon']:.0f}",
    "NoiseStrubFullHalfWidthTheirs": f"{strub_cos_half_width:.2f}",
    "NoiseStrubSweepHalfWidth": f"{strub['sweep_f0_half_width68_nhz_median']:.2f}",
    "NoiseStrubHalfWidthTheirs": f"{(strub['strub_f0_plus_nhz'] + strub['strub_f0_minus_nhz']) / 2:.2f}",
    "NoiseStrubTruthIota": f"{strub['iota']:g}",
    "NoiseSkyRatioBelow": f"{sky_ratio_below_mhz:g}",
    "NoiseStrubFzeroMhz": f"{strub['f0'] * 1e3:.5f}",
    "NoiseStrubLogFdot": f"{strub['log10_fdot']:g}",
    "NoiseStrubAmp": sci(strub["amplitude"], 2),
    "NoiseStrubLon": f"{strub['ecliptic_lon']:g}",
    "NoiseStrubLat": f"{strub['ecliptic_lat']:g}",
    "NoiseStrubPsi": f"{strub['psi']:g}",
    "NoiseStrubPhi": f"{strub['phi0']:g}",
    "NoiseStrubMarginals": WORDS[len(pulls)],
    "NoiseStrubCentred": WORDS[len(pulls) - len(prior_pulled)],
    "NoiseStrubBoxFzeroSigmas": f"{strub['box_half_width_log_f0'] / strub['box_fisher_sigma_log_f0']:.0f}",
    "NoiseStrubBoxFzeroHalfWidth": f"{strub['box_half_width_log_f0'] * strub['f0'] * 1e9:.2f}",
}
style.save_numbers("noise", numbers)
