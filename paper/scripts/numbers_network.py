"""Numbers of appendix/network.tex: LisaFlow's constants, token counts, sizes and parameter counts, from canna.lisa."""

import math

import style

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from canna.lisa import config, network
from canna.lisa.network import LisaFlow
from canna.lisa.problem import LisaGB

WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
SIZES = ("XS", "S", "B", "L", "XL")
BANDS = ("narrow", "full")
count = lambda tree: sum(
    math.prod(a.shape) for a in jax.tree.leaves(tree) if hasattr(a, "shape")
)

# every band and size, built abstractly (shapes only) from its config
configs, problems, flows = {}, {}, {}
for band in BANDS:
    for size in SIZES:
        cfg = config.load(f"{band}-{size}")
        problem = LisaGB(**cfg.problem)
        flows[band, size] = eqx.filter_eval_shape(
            lambda: LisaFlow(
                problem,
                **cfg.network,
                dtype=jnp.float32,
                param_dtype=jnp.float32,
                key=jr.key(cfg.seed),
            )
        )
        configs[band, size], problems[band] = cfg, problem

numbers = {}
for size in SIZES:
    cfg = configs["full", size]
    d, h, n = (cfg.network[k] for k in ("hidden_dim", "num_heads", "num_blocks"))
    backbone = count(flows["full", size].backbone)
    # 16 D^2 per stream (data, sources, sinks) per block, H query gains; the last block drops 10 D^2 of the data and of the sink stream
    assert backbone == n * (48 * d**2 + h) - 20 * d**2, (size, backbone)
    assert count(flows["narrow", size].backbone) == backbone, size
    key = size.capitalize()
    numbers |= {
        f"Network{key}Width": f"{d}",
        f"Network{key}Heads": f"{h}",
        f"Network{key}Blocks": f"{n}",
        f"Network{key}Backbone": str(backbone),
        f"Network{key}TotalNarrow": str(count(flows["narrow", size])),
        f"Network{key}TotalFull": str(count(flows["full", size])),
    }
    print(
        size,
        d,
        h,
        n,
        backbone,
        count(flows["narrow", size]),
        count(flows["full", size]),
    )

# what every size shares, read off full-B
flow, cfg = flows["full", "B"], configs["full", "B"]
head_dims = {
    cfg.network["hidden_dim"] // cfg.network["num_heads"] for cfg in configs.values()
}
assert len(head_dims) == 1
head_dim = head_dims.pop()
assert len(flow.rope_periods) * 2 == head_dim
assert flow.rope_periods[0] == 2.0 and flow.rope_periods[-1] == 2.0 * flow.window_bins
assert flow.y_periods[0] == 2 * flow.patch
assert network.Z_REACH == network.COORD_RANGES[-1][1]
assert all(r == network.COORD_RANGES[1] for r in network.COORD_RANGES[1:4])
assert flow.prediction == "x" and flow.self_condition
sinks, sources = flow.s_slot.shape[0], problems["full"].n_sources
assert all(configs[b, "B"].network.get("loss_weights") for b in BANDS)

# the backbone's FF width, read off the built block
expand = flow.backbone.last.mlp_x.linear2.weight.shape[1] // cfg.network["hidden_dim"]

# bf16 keeps nmant + 1 significant bits: rounding moves v by up to 2^-(nmant + 1) |v|, past one unit of whitened noise above 2^(nmant + 1)
bf16_bits = jnp.finfo(jnp.bfloat16).nmant + 1

# one encoding serves every draw of an observation in training, and every evaluation of a draw's path in sampling
evals = 2 * cfg.ode_steps + 1

numbers |= {
    "NetworkHeadDim": f"{head_dim}",
    "NetworkRopePairs": f"{head_dim // 2}",
    "NetworkPatch": f"{flow.patch}",
    "NetworkTokenScalars": f"{3 * flow.patch * 3}",
    "NetworkAmplitudeScalars": f"{3 * flow.patch}",
    "NetworkPhaseScalars": f"{2 * 3 * flow.patch}",
    "NetworkSinksWord": WORDS[sinks],
    "NetworkSourcesWord": WORDS[sources],
    "NetworkCoords": f"{len(network.COORD_RANGES)}",
    "NetworkCoordsWord": WORDS[len(network.COORD_RANGES)],
    "NetworkExpand": f"{expand}",
    "NetworkAmplitudeBins": f"{network.STFT_AMPLITUDE_BINS}",
    "NetworkPhaseBins": f"{network.STFT_PHASE_BINS}",
    "NetworkDataReach": f"{network.DATA_REACH:.0f}",
    "NetworkFzeroBins": f"{network.F0_BINS}",
    "NetworkCoordBins": f"{network.COORD_BINS}",
    "NetworkCoordReach": f"{network.COORD_RANGES[1][1]:.0f}",
    "NetworkZReach": f"{math.sinh(network.Z_REACH):.0f}",
    "NetworkHarmonics": f"{network.HARMONICS}",
    "NetworkHarmonicScalars": f"{4 * (network.HARMONICS - 1)}",
    "NetworkSourceScalars": f"{len(network.COORD_RANGES) + 4 * (network.HARMONICS - 1)}",
    "NetworkFzeroFinest": f"1/{1 / network.F0_FINEST_PERIOD:.0f}",
    "NetworkTokenPeriod": f"{flow.y_periods[0]:.0f}",
    "NetworkTimeDecades": f"{network.TIME_DECADES:.0f}",
    "NetworkSinusoidFreqs": f"{network.SINUSOID_FREQS}",
    "NetworkSinusoidDecades": f"{network.SINUSOID_DECADES:.0f}",
    "NetworkSlotScale": f"{0.02:g}",
    "NetworkDraws": f"{cfg.draws}",
    "NetworkSampleEvals": f"{evals}",
    "NetworkBfBits": f"{bf16_bits}",
    "NetworkBfUnity": f"{2**bf16_bits}",
    "NetworkSnrTop": f"{problems['full'].snr_range[1]:.0f}",
}
for band in BANDS:
    key, problem, flow = band.capitalize(), problems[band], flows[band, "B"]
    assert problem.window_bins % flow.patch == 0
    numbers |= {
        f"NetworkBins{key}": f"{problem.window_bins}",
        f"NetworkTokens{key}": f"{problem.window_bins // flow.patch}",
        f"NetworkFzeroPeriods{key}": f"{len(flow.f0_periods)}",
        f"NetworkTokenPeriods{key}": f"{len(flow.y_periods)}",
    }
saved = [count(flows["full", s]) - count(flows["narrow", s]) for s in SIZES]
numbers |= {
    "NetworkNarrowSavedMin": str(min(saved)),
    "NetworkNarrowSavedMax": str(max(saved)),
}
style.save_numbers("network", numbers)
