"""The numbers in sections/narrow-band.tex: the frequency bands set by the response width (reports/prior/response-span)
and how many sources a window of each band's width holds (reports/prior/source-count, as numbers_rates.py)."""

import importlib.util
import json
import math

import numpy as np
import yaml
from scipy.special import ndtr
from scipy.stats import binom

import style

from canna.lisa.physics import SKY_CATALOGUE_SOURCES
from canna.lisa.problem import WINDOW_RESPONSES

SPAN = style.REPO / "reports/prior/response-span/data"
COUNT = style.REPO / "reports/prior/source-count"
BANDS = style.REPO / "src/canna/lisa/configs/band"
F0_BINS = 40
BAND_R = (16, 32, 64, 128, 256)
BAND_NAMES = ("LowTail", "LowFreq", "Core", "HighFreq", "HighTail")
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
    if x == 0:
        return "0"
    digits = 1 - math.floor(math.log10(abs(x)))
    return f"{round(x, digits):.{max(0, digits)}f}"


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
first_over = [next(b for b in range(F0_BINS) if need[b] > r) for r in BAND_R[:-1]]
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

numbers = {
    "NarrowBandKeepPercent": f"{100 * KEEP:g}",
    "NarrowBandSharePercent": f"{100 * SHARE:g}",
    "NarrowBandVisibleSnr": f"{VISIBLE_SNR}",
    "NarrowBandTopR": f"{need[-1]}",
}
for name, r, lo, hi in zip(BAND_NAMES, BAND_R, band_edges, band_edges[1:]):
    numbers |= {
        f"NarrowBand{name}LowMhz": f"{lo * 1e3:.2g}",
        f"NarrowBand{name}HighMhz": f"{hi * 1e3:.2g}",
        f"NarrowBand{name}Catalogue": two_figures(share(catalogue, z["f0"], lo, hi)),
        f"NarrowBand{name}Visible": two_figures(share(visible, f0, lo, hi)),
        f"NarrowBand{name}R": f"{r}",
    }
# where even the top band's R falls short, and the visible share it loses bins of
over_top = 10 ** edges[next(b for b in range(F0_BINS) if need[b] > BAND_R[-1])]
numbers["NarrowBandOverTopMhz"] = f"{over_top * 1e3:.2g}"
numbers["NarrowBandOverTopVisible"] = two_figures(share(visible, f0, over_top, band_edges[-1]))
# sources and visible ones in a window of 4R bins across each band, by the binomial law of numbers_rates.py
spec = importlib.util.spec_from_file_location("source_count_page", COUNT / "scripts/build_page.py")
page = importlib.util.module_from_spec(spec)
spec.loader.exec_module(page)
d = json.loads((COUNT / "data/source-count.json").read_text())
c = np.load(COUNT / "data/source-count.npz")
count_edges = np.linspace(*np.log10(d["f0_range"]), page.F0_BINS + 1)
count_x = (count_edges[:-1] + count_edges[1:]) / 2
count_bins = np.clip(np.digitize(np.log10(c["f0"]), count_edges) - 1, 0, page.F0_BINS - 1)
count_weight = np.bincount(count_bins, c["f0_weight"], page.F0_BINS)
reach = ndtr((c["mean_log_a"] - np.log10(VISIBLE_SNR / c["snr_unit"])) / d["amp_std"])
reaching_visible = np.bincount(count_bins, c["f0_weight"] * reach, page.F0_BINS) / count_weight
SLOTS = (4, 8, 16)
windows = {}  # disjoint windows of 4R bins that tile each band
fitted = {slots: 0.0 for slots in SLOTS} | {"all": 0.0}  # visible sources a flow of S slots per window can fit, over the band
for name, r, lo, hi in zip(BAND_NAMES, BAND_R, band_edges, band_edges[1:]):
    centres = np.linspace(math.log10(lo), math.log10(hi), 200)
    width_bins = WINDOW_RESPONSES * r
    mean_all = SKY_CATALOGUE_SOURCES * page.window_share(
        centres, width_bins, count_x, np.ones(page.F0_BINS), d["t_obs"], d["f0_range"]
    )
    mean_visible = SKY_CATALOGUE_SOURCES * page.window_share(
        centres, width_bins, count_x, reaching_visible, d["t_obs"], d["f0_range"]
    )
    windows[name] = (hi - lo) * d["t_obs"] / width_bins
    # integral over f of the window density times E[min(K, S)], K ~ binomial visible sources in the window
    f = 10**centres
    for slots in SLOTS:
        fill = sum(binom.sf(j, SKY_CATALOGUE_SOURCES, mean_visible / SKY_CATALOGUE_SOURCES) for j in range(slots))
        fitted[slots] += np.trapezoid(fill * d["t_obs"] / width_bins, f)
    fitted["all"] += np.trapezoid(mean_visible * d["t_obs"] / width_bins, f)
    numbers |= {
        f"NarrowBand{name}Window": f"{width_bins}",
        f"NarrowBand{name}Windows": f"{windows[name]:.0f}",
        f"NarrowBand{name}AllLo": two_figures(mean_all.min()),
        f"NarrowBand{name}AllHi": two_figures(mean_all.max()),
        f"NarrowBand{name}VisibleLo": two_figures(mean_visible.min()) if mean_visible.min() >= 0.01 else "0",
        f"NarrowBand{name}VisibleHi": two_figures(mean_visible.max()),
    }
# windows tiling the bands between the tails, and the visible sources S slots per window fit (eq:narrow-band-fitted)
middle_windows = sum(windows[name] for name in BAND_NAMES[1:-1])
numbers["NarrowBandMiddleWindows"] = f"{middle_windows:.0f}"
for word, slots in zip(("Four", "Eight", "Sixteen"), SLOTS):
    numbers[f"NarrowBandFitted{word}"] = f"{fitted[slots]:.0f}"
numbers["NarrowBandVisibleTotal"] = f"{fitted['all']:.0f}"
print({name: round(n) for name, n in windows.items()}, {s: round(x) for s, x in fitted.items()})

numbers["NarrowBandNarrowVisible"] = two_figures(
    share(visible, f0, *narrow["f0_range"])
)

style.save_numbers("narrow-band", numbers)
