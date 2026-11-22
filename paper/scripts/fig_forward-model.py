"""What cutting a response to R bins costs: the truncation card of reports/diagnostics/forward-model.

Redone 2026-11-22 (user) after that report's "What does truncating the response cost?" card: (a) each source's relative
error against its signed offset from the nearest bin centre, (b) the median and 16-84% band over a ladder of R, the
band's own R dashed. The data are the ones the report page embeds (1024 of 4096 prior sources per band, the ladder
quantiles over all 4096); they are cached in data/figures/forward-model.json so restyling never needs the page.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

import style

PAGE = style.REPO / "reports/diagnostics/forward-model/forward-model.html"
CACHE = style.PAPER / "data/figures/forward-model.json"
BANDS = {"narrow": style.ORANGE, "full": style.BLUE}


def report_data() -> dict:
    """The truncation data the report page embeds, cached beside the paper's figure data."""
    if not CACHE.exists():
        html = PAGE.read_text()
        page, _ = json.JSONDecoder().raw_decode(html[html.index('{"bands"') :])
        out = {band: page["bands"][band]["truncation"] for band in BANDS}
        out["reference_points"] = page["reference_points"]
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(out))
    return json.loads(CACHE.read_text())


style.setup()
data = report_data()
fig, (ax_offset, ax_ladder) = plt.subplots(1, 2, figsize=(0.7 * style.FULL, 2.2), layout="constrained")

# (a) each source's truncation error against its signed offset from the nearest bin centre
for name, color in BANDS.items():
    t = data[name]
    ax_offset.plot(
        t["bin_offset"], t["rel"], color=color, marker="o", ms=1.6, mew=0, alpha=0.45, ls="none", rasterized=True
    )
    ax_offset.plot([], [], color=color, marker="o", ms=3, ls="none", label=f"{name} band, $R = {t['response_points']}$")
ax_offset.set(
    yscale="log",
    xlim=(-0.5, 0.5),
    xlabel=r"$f_0 T_{\rm obs} - \mathrm{rint}(f_0 T_{\rm obs})$ [bins]",
    ylabel="relative error of the response",
)
ax_offset.legend(loc="lower right", fontsize=5.5)
ax_offset.set_title("Error against bin offset")
style.panel_label(ax_offset, "a")

# (b) median and 16-84% of the relative error over a ladder of R, the band's own R dashed
for name, color in BANDS.items():
    t = data[name]
    ladder = np.asarray(t["ladder_points"], float)
    lo, med, hi = (np.asarray(q) for q in t["ladder_q"])
    ax_ladder.fill_between(ladder, lo, hi, color=color, alpha=0.18, lw=0)
    ax_ladder.loglog(ladder, med, color=color, marker="o", ms=2.6, label=rf"{name}: $\propto R^{{{t['ladder_slope']:.2f}}}$")
    ax_ladder.axvline(t["response_points"], color=color, lw=1.0, ls=(0, (4, 2.5)), zorder=1.6, alpha=0.9)
ax_ladder.legend(loc="upper right", fontsize=5.5)
ax_ladder.set(
    xlabel=r"response points $R$",
    ylabel="relative error of the response",
    xlim=(ladder[0] / 1.3, ladder[-1] * 1.3),
)
ax_ladder.set_title("Error against $R$")
style.panel_label(ax_ladder, "b")

style.save(fig, "forward-model")
