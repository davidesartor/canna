"""The numbers in sections/narrow-band.tex: the frequency bands set by the response width, from the response-span report."""

import json
import math

import numpy as np
import yaml

import style

SPAN = style.REPO / "reports/prior/response-span/data"
BANDS = style.REPO / "src/canna/lisa/configs/band"
F0_BINS = 40
BAND_R = (32, 64, 128)
VISIBLE_SNR = 7
KEEP, SHARE = 1.0, 0.999


def pow2(q: float) -> int:
    """The smallest power of two holding q bins."""
    return 1 << max(1, math.ceil(math.log2(max(q, 2))))


def quantile(a: np.ndarray, w: np.ndarray, y: float) -> float:
    """The smallest value with at least a share y of the weight w at or below it."""
    order = np.argsort(a, kind="stable")
    cumulative = np.cumsum(w[order])
    at = np.searchsorted(cumulative, y * cumulative[-1] * (1 - 1e-12))
    return float(a[order][min(at, a.size - 1)])


def two_figures(x: float) -> str:
    """x to two significant figures, without an exponent."""
    return f"{x:.{max(0, 1 - math.floor(math.log10(abs(x))))}f}"


span = json.loads((SPAN / "response-span.json").read_text())
z = np.load(SPAN / "response-span.npz")
narrow = yaml.safe_load((BANDS / "narrow.yaml").read_text())["problem"]
assert span["narrow_f0_max"] == narrow["f0_range"][1]
assert span["narrow_response_points"] == narrow["response_points"]

# per amplitude draw: its source's f0, its weight among the sources above the noise, its SNR and widths
draws = span["amplitudes"]
f0 = np.repeat(z["f0"], draws)
w = np.repeat(z["f0_weight"] * z["seen"] / draws, draws)
snr = 10 ** z["log_a"].astype(np.float64).ravel() * np.repeat(z["snr_unit"], draws)
width = z["width"].reshape(w.size, -1)[:, span["keep"].index(KEEP)]

# R needed per log-f0 bin: every above-noise bin for 99.9% of the sources above the noise
edges = np.linspace(*np.log10(span["f0_range"]), F0_BINS + 1)
bins = np.clip(np.digitize(np.log10(f0), edges) - 1, 0, F0_BINS - 1)
need = [pow2(quantile(width[bins == b], w[bins == b], SHARE)) for b in range(F0_BINS)]

# band edges: the lowest f0 bin needing more than each R, from which every higher bin does too
first_over = [next(b for b in range(F0_BINS) if need[b] > r) for r in BAND_R]
assert all(min(need[b:]) > r for b, r in zip(first_over, BAND_R))
band_edges = [
    span["f0_range"][0],
    *(10 ** edges[b] for b in first_over),
    span["f0_range"][1],
]

# shares of the catalogue (every source) and of the visible sources (SNR >= 7) per band
catalogue = z["f0_weight"] / z["f0_weight"].sum()
visible = w * (snr >= VISIBLE_SNR)
visible = visible / visible.sum()
share = lambda weights, f, lo, hi: 100 * weights[(f >= lo) & (f < hi)].sum()

names = ("One", "Two", "Three", "Four")
numbers = {
    "NarrowBandKeepPercent": f"{100 * KEEP:g}",
    "NarrowBandSharePercent": f"{100 * SHARE:g}",
    "NarrowBandVisibleSnr": f"{VISIBLE_SNR}",
    "NarrowBandTopR": f"{need[-1]}",
}
for name, r, lo, hi in zip(names, (*BAND_R, None), band_edges, band_edges[1:]):
    numbers |= {
        f"NarrowBand{name}LowMhz": f"{lo * 1e3:.2g}",
        f"NarrowBand{name}HighMhz": f"{hi * 1e3:.2g}",
        f"NarrowBand{name}Catalogue": two_figures(share(catalogue, z["f0"], lo, hi)),
        f"NarrowBand{name}Visible": two_figures(share(visible, f0, lo, hi)),
    }
    if r is not None:
        numbers[f"NarrowBand{name}R"] = f"{r}"
numbers["NarrowBandNarrowVisible"] = two_figures(
    share(visible, f0, *narrow["f0_range"])
)

style.save_numbers("narrow-band", numbers)
