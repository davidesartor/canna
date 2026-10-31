"""The numbers in appendix/flow-matching.tex: the lisa config, loss and sampler, and the Gaussian model of one coordinate."""

import json
import math
from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import style
from canna.lisa import config, sample
from canna.lisa.network import (
    COORDS,
    LOSS_GROUPS,
    LOSS_SNR_BINS,
    LOSS_TIME_BINS,
    LOSS_TIME_DECADES,
    TIME_DECADES,
    reference_frame,
)
from canna.lisa.train import FRAME_DECAY
from canna.lisa.problem import WINDOW_STRIDE, LisaGB, f0_window

cfg = config.load("full-B")
narrow = config.load("narrow-B")
for field in ("batch_size", "draws", "ode_steps"):
    assert getattr(cfg, field) == getattr(narrow, field), field
steps = cfg.ode_steps
S0 = 1 / math.sqrt(3)  # std of the f0 coordinate, uniform on [-1, 1] under the prior: the scale without a frame


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


def marginal(eps, sigma, s0):
    """x_s's variance m² and the optimal field's slope g of a N(0, s0²) base and a N(0, sigma²) posterior."""
    s = 1 - eps
    m2 = eps**2 * s0**2 + s**2 * sigma**2
    return m2, (s * sigma**2 - eps * s0**2) / m2


def excess_density(eps, sigma, sigma_m, s0=S0):
    """Excess velocity MSE at 1 - s = eps of a N(0, sigma_m²) model of a N(0, sigma²) posterior."""
    m2, g = marginal(eps, sigma, s0)
    _, gm = marginal(eps, sigma_m, s0)
    return m2 * (gm - g) ** 2


def excess_density_nll(eps, sigma, sigma_m, s0=S0):
    """Excess Gaussian NLL of x1 (eq:loss, twice the NLL) at 1 - s = eps of the same model, mean and variance its own."""
    m2, g = marginal(eps, sigma, s0)
    mm2, gm = marginal(eps, sigma_m, s0)
    var = eps**2 * s0**2 * sigma**2 / m2
    var_m = eps**2 * s0**2 * sigma_m**2 / mm2
    mean_error2 = eps**2 * m2 * (gm - g) ** 2
    return np.log(var_m / var) + (var + mean_error2) / var_m - 1


def excess(sigma_m, sigma, density=excess_density, s0=S0):
    """Quadrature in z = -ln(1 - s) of the excess under uniform s."""
    z = np.linspace(0, 60, 600_001)
    eps = np.exp(-z)
    return np.trapezoid(density(eps, sigma, sigma_m, s0) * eps, z)


uniform_asymptote = lambda sigma_m, s0=S0: math.pi / 4 * s0 * sigma_m
nll_asymptote = lambda sigma_m, sigma: math.log(sigma_m**2 / sigma**2) + sigma**2 / sigma_m**2 - 1

# the asymptotes against quadrature, for a true posterior far narrower than the model
check_uniform = excess(1e-3 * S0, 1e-7 * S0) / uniform_asymptote(1e-3 * S0)
check_nll = excess(1e-3 * S0, 1e-7 * S0, excess_density_nll) / nll_asymptote(1e-3, 1e-7)


# the visible sources' SNR span: 7, where reports/prior counts a source visible, to the prior's 99.99% quantile
SNR_VISIBLE, SNR_QUANTILE = 7.0, 0.9999


def centre_window(name: str) -> tuple[LisaGB, float]:
    """The config's problem and the centre frequency of its middle window."""
    problem = LisaGB(**config.load(name).problem)
    lo, hi = problem.window_index_range
    return problem, ((lo + hi) // 2 * WINDOW_STRIDE + problem.window_bins / 2) / problem.t_obs


# the ideal f0 width in prior units, from the Fisher bound of a sinusoid with unknown amplitude and phase
def band(name: str) -> dict[str, float]:
    problem, f = centre_window(name)
    low, high = (float(x) for x in f0_window(problem, jnp.asarray(f)))
    bins_per_unit = problem.t_obs * math.sqrt(low * high) * math.log(high / low) / 2
    prior = style.REPO / f"reports/prior/f0-fdot-amplitude/data/{name.removesuffix('-B')}.npz"
    snr_lo, snr_hi = SNR_VISIBLE, float(np.quantile(np.load(prior)["snr"], SNR_QUANTILE))
    ideal = lambda snr: math.sqrt(3) / (math.pi * snr) / bins_per_unit / S0
    return {
        "bins_per_unit": bins_per_unit,
        "weak": ideal(snr_lo),
        "loud": ideal(snr_hi),
        "snr_lo": snr_lo,
        "snr_hi": snr_hi,
    }


bands = {"Narrow": band("narrow-B"), "Full": band("full-B")}


class GaussianFlow(eqx.Module):
    """The exact mean and log std of z1 | z_t of a N(0, 1) base and a N(0, sigma^2) z1, as a LisaFlow whose frame is the prior's."""

    sigma: float
    n_sources: int = eqx.field(static=True)

    def encode(self, y, f):
        return jnp.zeros((self.n_sources, 2 * COORDS)), None

    def __call__(self, z, t, frame, f, cache):
        m2 = (1 - t) ** 2 + t**2 * self.sigma**2
        log_std = jnp.log((1 - t) * self.sigma) - 0.5 * jnp.log(m2)
        return jnp.concat([t * self.sigma**2 / m2 * z, jnp.broadcast_to(log_std, z.shape)], axis=-1)


problem, f = centre_window("narrow-B")


def sampled_width(sigma: float, ode_steps: int = steps, n: int = 1 << 13) -> float:
    """The width of z1 sample_posterior returns for the exact Gaussian z1 | z_t, over the true width."""
    z = jr.normal(jr.key(0), (n, problem.n_sources, COORDS))
    flow = GaussianFlow(sigma, problem.n_sources)
    x = sample.sample_posterior(problem, flow, z, jnp.zeros(()), jnp.asarray(f), ode_steps)
    # the coordinates past f0 have prior mean 0: x / sigma_frame is z1 without cancellation
    _, log_sigma = reference_frame(problem, flow.encode(None, f)[0], jnp.asarray(f))
    return float(jnp.std(x[..., 1:] / jnp.exp(log_sigma[..., 1:])) / sigma)


# without a frame the base is the prior and the f0 width is sigma / s0; the frame divides it out, to about one
width_frame = sampled_width(1.0)
width_wide = sampled_width(0.1)
width_narrow_weak = sampled_width(bands["Narrow"]["weak"])
width_full_weak = sampled_width(bands["Full"]["weak"])
width_narrow_weak_more = sampled_width(bands["Narrow"]["weak"], ode_steps=4 * steps)
width_narrow_weak_most = sampled_width(bands["Narrow"]["weak"], ode_steps=16 * steps)
width_full_weak_most = sampled_width(bands["Full"]["weak"], ode_steps=16 * steps)
last_mid = 1 / (2 * steps)

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
    if r["ode_steps"] == steps
}

# the MCMC reference on 2 CPU cores: one joint chain per injection (window), over chains
mcmc = json.loads(
    (style.REPO / "reports/benchmarks/sampling-cost/data/mcmc.json").read_text()
)
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
    "FlowMatchingOdeSteps": f"{steps}",
    "FlowMatchingOdeStepsMore": f"{4 * steps}",
    "FlowMatchingEvaluations": f"{2 * steps}",
    "FlowMatchingLastMid": tfrac(last_mid),
    "FlowMatchingCheckUniform": f"{check_uniform:.3f}",
    "FlowMatchingCheckNll": f"{check_nll:.3f}",
    "FlowMatchingSnrLow": f"{SNR_VISIBLE:g}",
    "FlowMatchingSnrQuantile": f"{100 * SNR_QUANTILE:g}",
    "FlowMatchingNarrowSnrHigh": f"{bands['Narrow']['snr_hi']:.0f}",
    "FlowMatchingFullSnrHigh": f"{bands['Full']['snr_hi']:.0f}",
    "FlowMatchingWidthWide": f"{width_wide:.2f}",
    "FlowMatchingWidthFrame": f"{width_frame:.3f}",
    "FlowMatchingFrameDecay": f"{FRAME_DECAY:g}",
    "FlowMatchingSnrBins": f"{LOSS_SNR_BINS}",
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
    "FlowMatchingMcmcWindowLow": f"{mcmc_window_h.min():.1f}",
    "FlowMatchingMcmcWindowHigh": f"{mcmc_window_h.max():.1f}",
    "FlowMatchingMcmcGapDecades": f"{gap_decades:.1f}",
}
for name, b in bands.items():
    numbers[f"FlowMatching{name}BinsPerUnit"] = f"{b['bins_per_unit']:.0f}"
    numbers[f"FlowMatching{name}Weak"] = sci(b["weak"])
    numbers[f"FlowMatching{name}Loud"] = sci(b["loud"])

print("bands", bands)
print("checks", check_uniform, check_nll)
print(
    "sampled widths",
    width_frame,
    width_wide,
    width_narrow_weak,
    width_narrow_weak_more,
    width_narrow_weak_most,
    width_full_weak_most,
    width_full_weak,
)
style.save_numbers("flow-matching", numbers)
