"""Shared Nature-style matplotlib helpers for HGCA gut-atlas downstream figures.

Implements the rules in ~/Projects/GCA/publication2026/plot_specs.md:
  - Arial/Helvetica, 5-7 pt at final size (default 6 pt)
  - white/transparent background, no gridlines, open L-shaped axes
  - thin 0.25-0.5 pt lines, colorblind-safe Wong palette
  - export cairo PDF (vector, editable) + 300 dpi PNG at final size
Sizes are specified in millimetres (single column 90 mm, double 180 mm, max height 170 mm).
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

MM = 1 / 25.4  # mm -> inches

# Wong colorblind-safe palette (plot_specs section 9)
WONG = dict(black="#000000", vermillion="#D55E00", blue="#0072B2",
            green="#009E73", sky="#56B4E9", yellow="#F0E442",
            purple="#CC79A7", lgrey="#E0E0E0", mgrey="#999999")
# lineage colours reused across the analysis
LINEAGE = dict(Epithelial="#009E73", Lymphoid="#0072B2", Myeloid="#D55E00",
               Stromal="#999999", Endothelial="#56B4E9", Glial="#CC79A7",
               Other="#E0E0E0")
# sequential (expression) and diverging (score) ramps
SEQ = LinearSegmentedColormap.from_list("gca_seq",
        ["#F2F2F2", "#FFE1BF", "#F1A340", "#A1430F"])
DIV = LinearSegmentedColormap.from_list("gca_div",
        ["#0072B2", "#E8E8E8", "#D55E00"])


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42,   # embed as editable text
        "svg.fonttype": "none",
        "font.size": 6, "axes.titlesize": 7, "axes.labelsize": 6,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5,
        "ytick.major.width": 0.5, "xtick.major.size": 2,
        "ytick.major.size": 2, "lines.linewidth": 0.5,
        "axes.grid": False, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


def open_axes(ax):
    """Open L-shaped axes (bottom + left only)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)


def fig_mm(width_mm, height_mm):
    """New figure sized in mm; caps height at the 170 mm page depth."""
    height_mm = min(height_mm, 170)
    return plt.subplots(figsize=(width_mm * MM, height_mm * MM))


def comma(x):
    return f"{int(x):,}"


def save(fig, stem, dpi=300):
    """Vector PDF (primary) + 300 dpi PNG preview, per plot_specs section 10."""
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
