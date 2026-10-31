"""Numbers of appendix/network.tex: LisaFlow's constants, token counts, sizes and parameter counts, from canna.lisa."""

import json
import math

import numpy as np

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
# the loudest sources: the full band's 99.99% quantile of the prior's source snr
SNR_QUANTILE = 0.9999
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
    # 16 D^2 per stream (data, sources, sinks, frame registers) per block, H query gains; the last block drops 10 D^2 of the data and of the sink stream
    assert backbone == n * (64 * d**2 + h) - 20 * d**2, (size, backbone)
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
rope_bins = [period * flow.t_obs for period in flow.rope_periods]
assert math.isclose(rope_bins[0], 2.0) and math.isclose(rope_bins[-1], 2.0 * flow.window_bins)
sinks, sources = flow.s_slot.shape[0], problems["full"].n_sources
assert flow.y_embed.in_features == 3 * flow.patch * 3
assert flow.x_unembed.linear2.out_features == 2 * network.COORDS
assert flow.frame_unembed.out_features == 2 * network.COORDS
assert flow.frame_slot.shape[0] == sources

# the backbone's FF width, read off the built block
expand = flow.backbone.last.mlp_x.linear2.weight.shape[1] // cfg.network["hidden_dim"]

# bf16 keeps nmant + 1 significant bits: rounding moves v by up to 2^-(nmant + 1) |v|, past one unit of whitened noise above 2^(nmant + 1)
bf16_bits = jnp.finfo(jnp.bfloat16).nmant + 1

# one encoding serves every draw of an observation in training, and every evaluation of a draw's path in sampling
evals = 2 * cfg.ode_steps + 1
prior = np.load(style.REPO / "reports/prior/f0-fdot-amplitude/data/full.npz")
snr_top = float(np.quantile(prior["snr"], SNR_QUANTILE))

numbers |= {
    "NetworkHeadDim": f"{head_dim}",
    "NetworkRopePairs": f"{head_dim // 2}",
    "NetworkPatch": f"{flow.patch}",
    "NetworkTokenScalars": f"{3 * flow.patch * 3}",
    "NetworkAmplitudeScalars": f"{3 * flow.patch}",
    "NetworkPhaseScalars": f"{2 * 3 * flow.patch}",
    "NetworkSinksWord": WORDS[sinks],
    "NetworkSourcesWord": WORDS[sources],
    "NetworkCoords": f"{network.COORDS}",
    "NetworkCoordsWord": WORDS[network.COORDS],
    "NetworkHeadOutputs": f"{2 * network.COORDS}",
    "NetworkExpand": f"{expand}",
    "NetworkHarmonics": f"{network.HARMONICS}",
    "NetworkHarmonicScalars": f"{4 * (network.HARMONICS - 1)}",
    "NetworkSourceScalars": f"{flow.x_embed.in_features}",
    "NetworkFreqFinest": f"1/{1 / network.FREQ_FINEST_PERIOD:.0f}",
    "NetworkTimeDecades": f"{network.TIME_DECADES:.0f}",
    "NetworkSinusoidFreqs": f"{network.SINUSOID_FREQS}",
    "NetworkSinusoidDecades": f"{network.SINUSOID_DECADES:.0f}",
    "NetworkDraws": f"{cfg.draws}",
    "NetworkSampleEvals": f"{evals}",
    "NetworkBfBits": f"{bf16_bits}",
    "NetworkBfUnity": f"{2**bf16_bits}",
    "NetworkSnrTop": f"{snr_top:.0f}",
    "NetworkSnrQuantile": f"{100 * SNR_QUANTILE:g}",
}
for band in BANDS:
    key, problem, flow = band.capitalize(), problems[band], flows[band, "B"]
    assert problem.window_bins % flow.patch == 0
    numbers |= {
        f"NetworkBins{key}": f"{problem.window_bins}",
        f"NetworkTokens{key}": f"{problem.window_bins // flow.patch}",
        f"NetworkFreqPeriods{key}": f"{len(flow.freq_periods)}",
    }
saved = [count(flows["full", s]) - count(flows["narrow", s]) for s in SIZES]
numbers |= {
    "NetworkNarrowSavedMin": str(min(saved)),
    "NetworkNarrowSavedMax": str(max(saved)),
}
# encode and decode matmul FLOPs and the cache of B, from the network-anatomy report
rows = [json.loads(line) for line in (style.REPO / "reports/architecture/network-anatomy/data/anatomy.jsonl").read_text().splitlines()]
for band in BANDS:
    key, problem, flow = band.capitalize(), problems[band], flows[band, "B"]
    row = next(r for r in rows if r["config"] == f"{band}-B")
    assert (row["window_bins"], row["data_tokens"]) == (problem.window_bins, problem.window_bins // flow.patch), band
    assert (row["hidden_dim"], row["num_blocks"]) == (configs[band, "B"].network["hidden_dim"], configs[band, "B"].network["num_blocks"]), band
    encode, decode = (row["flops"]["matmul"][k] for k in ("encode", "decode"))
    numbers |= {
        f"NetworkEncodeGflop{key}": f"{encode / 1e9:.2g}",
        f"NetworkDecodeGflop{key}": f"{decode / 1e9:.2g}",
        f"NetworkEncodeOverDecode{key}": f"{encode / decode:.0f}",
        f"NetworkCacheMib{key}": f"{row['cache_bytes'] / 2**20:g}",
    }
style.save_numbers("network", numbers)
