"""Shared look for every paper figure: palette, ink, sizes and the model-to-colour map.

The categorical slots are the validated reference palette (dataviz skill), in its fixed
order, checked against the white page with validate_palette.js: every hard gate passes;
aqua, yellow and magenta are below 3:1 contrast, so they always come with a legend and,
where it fits, a direct label. A model keeps its colour in every figure.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"
DATA = PAPER / "data"
OUTPUTS = REPO / "outputs"

# ICML: 6.75 in text width, two 3.25 in columns
COLUMN = 3.25
FULL = 6.75

# categorical slots, fixed order (light mode, white page)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
# ink and chrome
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
# sequential blue ramp (light -> dark), for magnitude only
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# one colour per trained model, everywhere. The final model takes slot 1, and the three
# that meet in scatter plots (final, baseline, warped clock) hold the first three slots,
# the only ones that validate all-pairs
MODELS = {
    "XS-late-768-cool": dict(color=BLUE, label="768, warp, cooldown (final)", short="final"),
    "XS-1M": dict(color=ORANGE, label="512, uniform clock", short="uniform"),
    "XS-late": dict(color=AQUA, label="512, warp", short="warp"),
    "XS-late-768": dict(color=YELLOW, label="768, warp", short="warp 768"),
    "XS-late-cool": dict(color=MAGENTA, label="512, warp, cooldown", short="warp + cool"),
    "XS-500k": dict(color=GREEN, label="512, uniform clock, 0.5M steps", short="0.5M"),
    "B-late": dict(color=VIOLET, label="B: 512, warp, cooldown", short="B"),
}


def setup() -> None:
    """Thin marks, hairline solid grid, ink-coloured text, Type 42 fonts (no Type 3)."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["TeX Gyre Heros", "Nimbus Sans", "DejaVu Sans"],
            # STIX sans for math: close to Heros, and it has a real \mathcal
            "mathtext.fontset": "stixsans",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.6,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "axes.grid": True,
            "axes.grid.which": "major",
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.4,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "lines.markersize": 4.0,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax, letter: str, x: float = -0.02, y: float = 1.02) -> None:
    """(a), (b), ... at the top left of a panel, in ink, outside the plot area."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, fontweight="bold", color=INK)


def save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.pdf"
    fig.savefig(path)
    # a png twin, only for looking at the figure quickly; the paper uses the pdf
    fig.savefig(FIGURES / "preview" / f"{name}.png", dpi=200) if (FIGURES / "preview").is_dir() else None
    plt.close(fig)
    print(f"wrote {path}")
    return path
