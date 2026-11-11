"""Numbers of the source-rate paragraph of appendix/priors.tex: how many sources, and visible ones, a window holds.

From reports/prior/source-count (data/ written by its count.sbatch, the helpers of its build_page.py): the catalogue's
N sources in the band, each in a window of w bins centred at f with the prior share p(f, w), so the window holds
K ~ Bin(N, p) of them, and those reaching SNR 7 alone in the window by the share with log A integrated out.
"""

import importlib.util
import json
import math

import numpy as np
from scipy.special import ndtr

import style

from canna.lisa import config
from canna.lisa.physics import SKY_CATALOGUE_SOURCES

REPORT = style.REPO / "reports/prior/source-count"
spec = importlib.util.spec_from_file_location("source_count_page", REPORT / "scripts/build_page.py")
page = importlib.util.module_from_spec(spec)
spec.loader.exec_module(page)

d = json.loads((REPORT / "data/source-count.json").read_text())
z = np.load(REPORT / "data/source-count.npz")
assert d["config"] == "full-B" and not d["src_dirty"], d
f_lo, f_hi = d["f0_range"]
edges = np.linspace(math.log10(f_lo), math.log10(f_hi), page.F0_BINS + 1)
count_x = (edges[:-1] + edges[1:]) / 2
bins = np.clip(np.digitize(np.log10(z["f0"]), edges) - 1, 0, page.F0_BINS - 1)
weight = np.bincount(bins, z["f0_weight"], page.F0_BINS)


def reaching(t: float) -> np.ndarray:
    """Per log-f0 bin, the prior share of sources there that reach SNR t alone in the window."""
    if t == 0:
        return np.ones(page.F0_BINS)
    reach = ndtr((z["mean_log_a"] - np.log10(t / z["snr_unit"])) / d["amp_std"])
    return np.bincount(bins, z["f0_weight"] * reach, page.F0_BINS) / weight


N = SKY_CATALOGUE_SOURCES
VISIBLE = page.VISIBLE_SNR
numbers = {
    "RatesCatalogue": f"{N:.0e}".replace("e+0", "e"),
    "RatesVisibleSnr": f"{VISIBLE:g}",
    "RatesVisibleShare": f"{100 * float(np.mean(z['f0_weight'] * ndtr((z['mean_log_a'] - np.log10(VISIBLE / z['snr_unit'])) / d['amp_std']))):.2f}",
}
for band, name in (("narrow", "Narrow"), ("full", "Full")):
    c = config.load(f"{band}-B")
    width = page.WINDOW_RESPONSES * c.problem["response_points"]
    lo, hi = c.problem["f0_range"]
    centres = np.linspace(math.log10(lo), math.log10(hi), 200)
    share_all = page.window_share(centres, width, count_x, reaching(0.0), d["t_obs"], d["f0_range"])
    share_vis = page.window_share(centres, width, count_x, reaching(VISIBLE), d["t_obs"], d["f0_range"])
    mean_all, mean_vis = N * share_all, N * share_vis
    busiest = int(np.argmax(mean_vis))
    numbers[f"Rates{name}Window"] = str(width)
    numbers[f"Rates{name}AllLo"] = f"{mean_all.min():.0f}"
    numbers[f"Rates{name}AllHi"] = f"{mean_all.max():.0f}"
    numbers[f"Rates{name}VisibleLo"] = f"{mean_vis[0]:.1f}" if mean_vis[0] < 10 else f"{mean_vis[0]:.0f}"
    numbers[f"Rates{name}VisibleTop"] = f"{mean_vis[-1]:.0f}"
    numbers[f"Rates{name}VisibleMax"] = f"{mean_vis[busiest]:.0f}"
    numbers[f"Rates{name}BusiestMhz"] = f"{10 ** centres[busiest] * 1e3:.2f}"
    print(band, width, f"all {mean_all.min():.3g}-{mean_all.max():.3g}", f"visible at lo {mean_vis[0]:.3g}, top {mean_vis[-1]:.3g},"
          f" max {mean_vis[busiest]:.3g} at {10 ** centres[busiest] * 1e3:.3g} mHz")

style.save_numbers("rates", numbers)
