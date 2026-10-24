"""The numbers in appendix/flow-matching.tex: the lisa config, loss and sampler, and the Gaussian model of one coordinate."""

import inspect
import json
import math
from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import style
from canna.lisa import config, sample, train
from canna.lisa.network import (
    LOSS_GROUPS,
    LOSS_TIME_BINS,
    LOSS_TIME_DECADES,
    TIME_DECADES,
)
from canna.lisa.problem import WINDOW_STRIDE, LisaGB, f0_window

cfg = config.load("full-B")
narrow = config.load("narrow-B")
for field in (
    "batch_size",
    "draws",
    "time_power",
    "late_time_mix",
    "late_time_floor",
    "ode_steps",
):
    assert getattr(cfg, field) == getattr(narrow, field), field
assert cfg.network["loss_weights"] and narrow.network["loss_weights"]
mix, floor, p, steps = (
    cfg.late_time_mix,
    cfg.late_time_floor,
    cfg.time_power,
    cfg.ode_steps,
)
assert p == 1
S0 = 1 / math.sqrt(3)  # std of the f0 coordinate, uniform on [-1, 1] under the prior


def tfrac(x: float) -> str:
    fraction = Fraction(x).limit_denominator(64)
    assert abs(float(fraction) - x) < 1e-12, x
    return rf"\tfrac{{{fraction.numerator}}}{{{fraction.denominator}}}"


def sci(x: float) -> str:
    exponent = math.floor(math.log10(x))
    mantissa = round(x / 10**exponent)
    if mantissa == 10:
        mantissa, exponent = 1, exponent + 1
    return f"{mantissa}e{exponent}"


def percent(x: float) -> str:
    return f"{100 * x:.0f}"


# where the training draws land: Pr(1 - s < eps) under the mix and under a power clock alone
share_mix = lambda eps: (1 - mix) * eps ** (1 / p) + mix * max(
    0.0, 1 - math.log(eps) / math.log(floor)
)
share_power = lambda eps, power: eps ** (1 / power)

# the self-conditioning coin is jr.bernoulli at its default p
assert "jr.bernoulli(key_hat)" in inspect.getsource(train.train_epoch)
self_condition_p = inspect.signature(jr.bernoulli).parameters["p"].default


def excess_density(eps, sigma, sigma_m, s0=S0):
    """Excess flow-matching loss at 1 - s = eps of a N(0, sigma_m^2) model of a N(0, sigma^2) posterior."""
    s = 1 - eps
    m2 = eps**2 * s0**2 + s**2 * sigma**2
    mm2 = eps**2 * s0**2 + s**2 * sigma_m**2
    g = (s * sigma**2 - eps * s0**2) / m2
    gm = (s * sigma_m**2 - eps * s0**2) / mm2
    return m2 * (gm - g) ** 2


def excess(sigma_m, sigma, late_share, s0=S0):
    """Quadrature in z = -ln(1 - s) of the excess under the clock: uniform s, plus late_share log-uniform down to floor."""
    z = np.linspace(0, 60, 600_001)
    eps = np.exp(-z)
    weight = (1 - late_share) * eps + late_share * (z <= -math.log(floor)) / -math.log(
        floor
    )
    return np.trapezoid(excess_density(eps, sigma, sigma_m, s0) * weight, z)


uniform_asymptote = lambda sigma_m, s0=S0: math.pi / 4 * s0 * sigma_m
mix_asymptote = lambda sigma_m, s0=S0: (1 - mix) * uniform_asymptote(
    sigma_m, s0
) + mix * s0**2 * (math.log(sigma_m / (s0 * floor)) - 0.5) / -math.log(floor)

# the asymptotes against quadrature, for a true posterior far narrower than the model
check_uniform = excess(1e-3 * S0, 1e-7 * S0, 0.0) / uniform_asymptote(1e-3 * S0)
check_mix = excess(1e-2 * S0, 1e-6 * S0, mix) / mix_asymptote(1e-2 * S0)


# what one e-fold of narrowing is worth at sigma_m = 1e-3 s0, the mix against the uniform clock
def slope(late_share, sigma_m=1e-3 * S0, sigma=1e-7 * S0, h=0.05):
    return (
        excess(sigma_m * math.exp(h), sigma, late_share)
        - excess(sigma_m * math.exp(-h), sigma, late_share)
    ) / (2 * h)


gain = slope(mix) / slope(0.0)


# the ideal f0 width in chart units, from the Fisher bound of a sinusoid with unknown amplitude and phase
def band(name: str) -> dict[str, float]:
    run = config.load(name)
    problem = LisaGB(**run.problem)
    lo, hi = problem.window_index_range
    index = (lo + hi) // 2
    f = (index * WINDOW_STRIDE + problem.window_bins / 2) / problem.t_obs
    low, high = (float(x) for x in f0_window(problem, jnp.asarray(f)))
    bins_per_unit = problem.t_obs * math.sqrt(low * high) * math.log(high / low) / 2
    snr_lo, snr_hi = run.problem["snr_range"]
    ideal = lambda snr: math.sqrt(3) / (math.pi * snr) / bins_per_unit / S0
    return {
        "bins_per_unit": bins_per_unit,
        "weak": ideal(snr_lo),
        "loud": ideal(snr_hi),
        "snr_lo": snr_lo,
        "snr_hi": snr_hi,
    }


bands = {"Narrow": band("narrow-B"), "Full": band("full-B")}
assert bands["Narrow"]["snr_hi"] == bands["Full"]["snr_hi"]


class GaussianFlow(eqx.Module):
    """The exact endpoint E[x1 | x_s] of a N(0, s0^2) base and a N(0, sigma^2) posterior, as a LisaFlow."""

    sigma: float
    s0: float = S0
    prediction: str = eqx.field(static=True, default="x")
    self_condition: bool = eqx.field(static=True, default=True)

    def encode(self, y, f):
        return jnp.zeros(())

    def __call__(self, u, t, y, f, x_hat, cache):
        s = 1 - (1 - t) ** p
        m2 = (1 - s) ** 2 * self.s0**2 + s**2 * self.sigma**2
        return s * self.sigma**2 / m2 * u


def sampled_width(
    sigma_scaled: float, ode_steps: int = steps, n: int = 1 << 16
) -> float:
    """The width sample_posterior returns for the exact Gaussian endpoint, over the true width."""
    u = S0 * jr.normal(jr.key(0), (n, 1))
    flow = GaussianFlow(sigma=sigma_scaled * S0)
    x = sample.sample_posterior(flow, u, jnp.zeros(()), jnp.zeros(()), ode_steps, p)
    return float(jnp.std(x) / (sigma_scaled * S0))


width_wide = sampled_width(0.1)
width_narrow_weak = sampled_width(bands["Narrow"]["weak"])
width_full_weak = sampled_width(bands["Full"]["weak"])
width_narrow_weak_more = sampled_width(bands["Narrow"]["weak"], ode_steps=4 * steps)
width_narrow_weak_most = sampled_width(bands["Narrow"]["weak"], ode_steps=16 * steps)
width_full_weak_most = sampled_width(bands["Full"]["weak"], ode_steps=16 * steps)
last_mid = (1 / (2 * steps)) ** p

# sampling cost, measured in reports/benchmarks/sampling-cost
FLOW_DRAWS = 256
records = [
    json.loads(line)
    for line in (style.REPO / "reports/benchmarks/sampling-cost/data/a100.jsonl")
    .read_text()
    .splitlines()
]
cost = {
    (r["config"], r["draws"]): r["best_s"]
    for r in records
    if r["ode_steps"] == steps and r["prediction"] == "x"
}

# the MCMC reference on 2 CPU cores: per source (narrowband) and per window (joint), over chains
mcmc = json.loads(
    (style.REPO / "reports/benchmarks/sampling-cost/data/mcmc.json").read_text()
)
mcmc_source_min = np.array([chain["seconds"] for chain in mcmc["narrowband"]]) / 60
mcmc_window_h = np.array([chain["seconds"] for chain in mcmc["joint"]]) / 3600
gap_decades = math.log10(np.median(mcmc_window_h) * 3600 / cost["full-B", FLOW_DRAWS])

numbers = {
    "FlowMatchingDraws": f"{cfg.draws}",
    "FlowMatchingCostDraws": f"{FLOW_DRAWS}",
    "FlowMatchingWindows": f"{cfg.batch_size // cfg.draws}",
    "FlowMatchingBatch": f"{cfg.batch_size}",
    "FlowMatchingGroups": f"{len(LOSS_GROUPS)}",
    "FlowMatchingLossBins": f"{LOSS_TIME_BINS}",
    "FlowMatchingLossDecades": f"{-LOSS_TIME_DECADES:g}",
    "FlowMatchingLossBinWidth": f"{LOSS_TIME_DECADES / LOSS_TIME_BINS:g}",
    "FlowMatchingTimeDecades": f"{-TIME_DECADES:g}",
    "FlowMatchingTimePower": f"{p:g}",
    "FlowMatchingLateMix": tfrac(mix),
    "FlowMatchingLateFloor": f"{round(math.log10(floor))}",
    "FlowMatchingPerDecade": tfrac(mix / -math.log10(floor)),
    "FlowMatchingShareHundredth": percent(share_mix(1e-2)),
    "FlowMatchingShareThousandth": percent(share_mix(1e-3)),
    "FlowMatchingShareBelowFloor": sci((1 - mix) * floor ** (1 / p)),
    "FlowMatchingShareBelowLossFloor": sci(
        (1 - mix) * (10**-LOSS_TIME_DECADES) ** (1 / p)
    ),
    "FlowMatchingFirstHalfMix": percent(1 - share_mix(0.5)),
    "FlowMatchingFirstHalfWarp": percent(1 - share_power(0.5, 3)),
    "FlowMatchingBelowFloorWarp": percent(share_power(floor, 3)),
    "FlowMatchingSelfCondition": tfrac(self_condition_p),
    "FlowMatchingOdeSteps": f"{steps}",
    "FlowMatchingOdeStepsMore": f"{4 * steps}",
    "FlowMatchingEvaluations": f"{2 * steps + 1}",
    "FlowMatchingLastMid": tfrac(last_mid),
    "FlowMatchingCheckUniform": f"{check_uniform:.3f}",
    "FlowMatchingCheckMix": f"{check_mix:.2f}",
    "FlowMatchingMixGain": f"{gain:.0f}",
    "FlowMatchingSnrLow": f"{bands['Full']['snr_lo']:g}",
    "FlowMatchingSnrHigh": f"{bands['Full']['snr_hi']:g}",
    "FlowMatchingWidthWide": f"{width_wide:.2f}",
    "FlowMatchingWidthNarrowWeak": f"{width_narrow_weak:.2f}",
    "FlowMatchingWidthNarrowWeakMore": f"{width_narrow_weak_more:.2f}",
    "FlowMatchingWidthFullWeak": sci(width_full_weak),
    "FlowMatchingOdeStepsMost": f"{16 * steps}",
    "FlowMatchingWidthNarrowWeakMost": f"{width_narrow_weak_most:.2f}",
    "FlowMatchingWidthFullWeakMost": f"{width_full_weak_most:.2f}",
    "FlowMatchingCostNarrow": f"{cost['narrow-B', FLOW_DRAWS]:.2f}",
    "FlowMatchingCostFull": f"{cost['full-B', FLOW_DRAWS]:.2f}",
    "FlowMatchingCostSmallest": f"{min(cost['narrow-XS', FLOW_DRAWS], cost['full-XS', FLOW_DRAWS]):.2f}",
    "FlowMatchingCostLargest": f"{cost['full-XL', FLOW_DRAWS]:.2f}",
    "FlowMatchingCostDrawsMore": f"{4 * FLOW_DRAWS}",
    "FlowMatchingCostFullMore": f"{cost['full-B', 4 * FLOW_DRAWS]:.2f}",
    "FlowMatchingMcmcSourceLow": f"{mcmc_source_min.min():.1f}",
    "FlowMatchingMcmcSourceHigh": f"{mcmc_source_min.max():.1f}",
    "FlowMatchingMcmcWindowLow": f"{mcmc_window_h.min():.1f}",
    "FlowMatchingMcmcWindowHigh": f"{mcmc_window_h.max():.1f}",
    "FlowMatchingMcmcGapDecades": f"{gap_decades:.1f}",
}
for name, b in bands.items():
    numbers[f"FlowMatching{name}BinsPerUnit"] = f"{b['bins_per_unit']:.0f}"
    numbers[f"FlowMatching{name}Weak"] = sci(b["weak"])
    numbers[f"FlowMatching{name}Loud"] = sci(b["loud"])

print("bands", bands)
print("checks", check_uniform, check_mix, "gain", gain)
print(
    "sampled widths",
    width_wide,
    width_narrow_weak,
    width_narrow_weak_more,
    width_narrow_weak_most,
    width_full_weak_most,
    width_full_weak,
)
style.save_numbers("flow-matching", numbers)
