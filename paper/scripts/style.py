"""Shared look for every paper figure: Inter type, a validated palette, light chrome, panel letters.

Categorical slots are the dataviz reference palette in its fixed CVD-safe order: BLUE is the
primary series, ORANGE the second; aqua, yellow and magenta are below 3:1 on white, so they
always come with a legend or a direct label. A model or band keeps its colour in every figure.
Markers get a white ring by default: an open marker (mfc="none") needs an explicit mec.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent
FIGURES = PAPER / "figures"
NUMBERS = PAPER / "numbers"
FONTS = PAPER / "fonts"  # vendored Inter (OFL, fonts/OFL.txt), registered by setup()

# ICML: 6.75 in text width, two 3.25 in columns
COLUMN = 3.25
FULL = 6.75

# categorical slots, fixed order (light mode, white page)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
# ink: primary text, secondary text (ticks, legends, notes), muted guides and context series
INK, INK2, MUTED = "#16161a", "#55555c", "#909098"
# chrome: hairline grid, spines and ticks
GRID, AXIS = "#e9e9ec", "#b8b8bf"
# neutral wash for shaded regions (guard bands, tolerance bands); the grid stays visible over it
FILL = "#f2f2f4"
# sequential blue ramp (light -> dark), for magnitude only
BLUE_RAMP = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]
# matching colormap for imshow/pcolormesh/contourf
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "paper_blue", ["#f4f8fe", *BLUE_RAMP]
)
# line kwargs for guides: y=1, asymptotes, medians, theory curves the data is compared to
GUIDE = dict(color=MUTED, lw=0.75, ls=(0, (4, 2.5)), zorder=1.6)


def setup() -> None:
    """Inter text and math, hairline chrome, left-aligned semibold titles, Type 42 fonts."""
    for path in sorted(FONTS.glob("*.ttf")):
        font_manager.fontManager.addfont(str(path))
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Nimbus Sans", "DejaVu Sans"],
            # math in Inter too; \mathcal from Computer Modern like the text, other symbols from STIX sans
            "mathtext.fontset": "custom",
            "mathtext.rm": "Inter",
            "mathtext.sf": "Inter",
            "mathtext.it": "Inter:italic",
            "mathtext.bf": "Inter:semibold",
            "mathtext.bfit": "Inter:italic",
            "mathtext.cal": "cmsy10",
            "mathtext.fallback": "stixsans",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 5,
            "axes.titlecolor": INK,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.3,
            "legend.labelcolor": INK2,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.labelpad": 2.5,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.5,
            "axes.facecolor": "white",
            "axes.prop_cycle": plt.cycler(
                color=[BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
            ),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.which": "major",
            # grid above fills and shaded bands (zorder 1.5), below lines and markers
            "axes.axisbelow": "line",
            "grid.color": GRID,
            "grid.linewidth": 0.45,
            "grid.linestyle": "-",
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "xtick.minor.size": 1.2,
            "ytick.minor.size": 1.2,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.pad": 2.0,
            "ytick.major.pad": 2.0,
            "lines.linewidth": 1.2,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "lines.dash_capstyle": "butt",
            "lines.markersize": 3.6,
            # a white ring keeps markers legible where they sit on a line or overlap
            "lines.markeredgecolor": "white",
            "lines.markeredgewidth": 0.5,
            "patch.linewidth": 0.5,
            "contour.linewidth": 0.6,
            "image.cmap": "paper_blue",
            "legend.frameon": False,
            "legend.handlelength": 1.5,
            "legend.handleheight": 0.6,
            "legend.handletextpad": 0.5,
            "legend.labelspacing": 0.3,
            "legend.columnspacing": 1.0,
            "legend.borderaxespad": 0.3,
            "legend.borderpad": 0.2,
            "figure.facecolor": "white",
            "figure.constrained_layout.w_pad": 0.03,
            "figure.constrained_layout.h_pad": 0.03,
            "figure.constrained_layout.wspace": 0.06,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    if SEQUENTIAL_CMAP.name not in matplotlib.colormaps:
        matplotlib.colormaps.register(SEQUENTIAL_CMAP)


def panel_label(
    ax, letter: str, x: float | None = None, y: float | None = None
) -> None:
    """Semibold panel letter at the panel's top left, on the title's baseline.

    By default it sits over the y tick labels and axis label; explicit x, y place it in axes coordinates.
    """
    style = dict(fontsize=8, fontweight="semibold", color=INK, annotation_clip=False)
    if x is not None or y is not None:
        ax.annotate(
            letter,
            (-0.02 if x is None else x, 1.02 if y is None else y),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            **style,
        )
        return
    left = lambda renderer: ax.yaxis.get_tightbbox(renderer) or ax.bbox
    ax.annotate(
        letter,
        (0, 1),
        xycoords=(left, "axes fraction"),
        xytext=(0, plt.rcParams["axes.titlepad"]),
        textcoords="offset points",
        ha="left",
        va="baseline",
        **style,
    )


def direct_label(
    ax, x: float, y: float, text: str, dx: float = 3, dy: float = 0, **kwargs
):
    """A series name at a data point (line end, peak), in secondary ink, offset (dx, dy) points."""
    style = dict(
        textcoords="offset points",
        ha="left" if dx >= 0 else "right",
        va="center",
        fontsize=6.3,
        color=INK2,
        annotation_clip=False,
    )
    return ax.annotate(text, (x, y), xytext=(dx, dy), **(style | kwargs))


def save_numbers(appendix: str, numbers: dict[str, str]) -> Path:
    """numbers/<appendix>.tex: one \\newcommand per number, which appendix/<appendix>.tex uses as \\Name{}."""
    NUMBERS.mkdir(parents=True, exist_ok=True)
    path = NUMBERS / f"{appendix}.tex"
    lines = [f"% generated by paper/scripts/numbers_{appendix}.py; do not edit by hand"]
    lines += [rf"\newcommand{{\{key}}}{{{value}}}" for key, value in numbers.items()]
    path.write_text("\n".join(lines) + "\n")
    print(path.read_text())
    return path


def save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")
    return path
