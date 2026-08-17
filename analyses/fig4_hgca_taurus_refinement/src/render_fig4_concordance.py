"""
Fig 4 concordance panel — readable replacement for the dense
``final_analysis × predictions`` confusion matrix.

Instead of a 109 × 93 row-normalized heatmap (whose marginal log-count bars
bleed the same blue scale and bury the hierarchy signal), this figure shows:

  a. Cohort summary of hierarchy classes (overall + per lineage).
  b. Per-author ``final_analysis`` stacked bars (faceted by lineage): for each
     Taurus author label, the fraction of cells that landed in
     atlas-increased / same / reduced / minor–major reassignment /
     low-confidence. A thin grey n-cells bar sits to the right (neutral).
  c. Top reassignment flows: author → HGCA destination with severity +
     cell counts — the concrete audit of the manual crosswalk.

Uses the same Wong palette and hierarchy nomenclature as Fig 4a.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Reuse Fig 2 helpers for typography / export
FIG2_SRC = Path(__file__).resolve().parents[2] / "fig2_label_set" / "src"
sys.path.insert(0, str(FIG2_SRC))
from build_fig2_atlas_evidence import configure_plotting, save_figure  # noqa: E402

# Local panel constants
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_fig4_panel import (  # noqa: E402
    LINEAGE_COLORS,
    LINEAGE_STACK_ORDER,
    REASSIGNMENT_CLASSES,
    REFINEMENT_COLORS,
    REFINEMENT_STACK_ORDER,
    _pretty_class,
)


LOG = logging.getLogger("render_fig4_concordance")

CLASS_ORDER = list(REFINEMENT_STACK_ORDER)
MIN_CELLS_FOR_ROW = 50  # drop tiny author labels from the facet panels


def _load_cells(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "fig4a_refinement_by_cell.csv.gz"
    df = pd.read_csv(path)
    if "final_analysis" not in df.columns:
        raise SystemExit(
            f"{path} missing final_analysis — rebuild with build_fig4_metrics.py"
        )
    return df


def _plot_summary_stack(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    legend_ncol: int = 4,
    legend_bbox: Tuple[float, float] = (0.5, -0.42),
) -> None:
    scopes = ["overall", *LINEAGE_STACK_ORDER]
    y = np.arange(len(scopes))
    left = np.zeros(len(scopes))
    for cls in CLASS_ORDER:
        widths = []
        for scope in scopes:
            sub = df if scope == "overall" else df[df["assigned_lineage"] == scope]
            n = len(sub)
            widths.append(
                100.0 * (sub["refinement_class"] == cls).sum() / n if n else 0.0
            )
        ax.barh(
            y, widths, left=left, height=0.7,
            color=REFINEMENT_COLORS[cls], label=_pretty_class(cls),
            edgecolor="white", linewidth=0.3,
        )
        left = left + np.asarray(widths)
    ax.set_yticks(y)
    labels = []
    for scope in scopes:
        sub = df if scope == "overall" else df[df["assigned_lineage"] == scope]
        labels.append(f"{scope.capitalize()}\nn={len(sub):,}")
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.set_xlabel("% of cells", fontsize=6)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_title("Per-cell hierarchy class (reassignment graded by path distance)",
                 loc="left", fontsize=6.5, pad=3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=2, width=0.5)
    ax.legend(
        loc="lower center", bbox_to_anchor=legend_bbox, ncol=legend_ncol,
        fontsize=4.8, frameon=False, handlelength=0.9, handleheight=0.7,
        columnspacing=0.7, labelspacing=0.25,
    )


def _author_class_matrix(df: pd.DataFrame, lineage: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Rows = final_analysis labels; cols = hierarchy classes (fractions);
    also return n_cells per author label."""
    sub = df[df["assigned_lineage"] == lineage].copy()
    counts = sub.groupby(["final_analysis", "refinement_class"]).size().unstack(fill_value=0)
    for cls in CLASS_ORDER:
        if cls not in counts.columns:
            counts[cls] = 0
    counts = counts[CLASS_ORDER]
    n = counts.sum(axis=1)
    keep = n >= MIN_CELLS_FOR_ROW
    counts = counts.loc[keep]
    n = n.loc[keep]
    # order by total % reassignment (minor+moderate+major), descending
    frac = counts.div(n, axis=0)
    reassign_frac = frac[
        [c for c in REASSIGNMENT_CLASSES if c in frac.columns]
    ].sum(axis=1)
    order = reassign_frac.sort_values(ascending=False).index
    return frac.loc[order], n.loc[order]


def _plot_lineage_facet(
    ax_stack: plt.Axes,
    ax_n: plt.Axes,
    frac: pd.DataFrame,
    n_cells: pd.Series,
    lineage: str,
) -> None:
    if frac.empty:
        ax_stack.set_visible(False)
        ax_n.set_visible(False)
        return
    y = np.arange(len(frac))
    left = np.zeros(len(frac))
    for cls in CLASS_ORDER:
        w = frac[cls].to_numpy() * 100.0
        ax_stack.barh(
            y, w, left=left, height=0.85,
            color=REFINEMENT_COLORS[cls],
            edgecolor="white", linewidth=0.15,
        )
        left = left + w

    # Truncate long author labels
    labels = [
        (lab[:38] + "…") if len(lab) > 39 else lab
        for lab in frac.index.astype(str)
    ]
    ax_stack.set_yticks(y)
    ax_stack.set_yticklabels(labels, fontsize=4.5)
    ax_stack.set_xlim(0, 100)
    ax_stack.invert_yaxis()
    ax_stack.set_xlabel("% of author label", fontsize=5)
    ax_stack.set_title(
        f"{lineage.capitalize()}  (n_labels={len(frac)}, "
        f"n_cells={int(n_cells.sum()):,})",
        loc="left", fontsize=6, pad=2,
        color=LINEAGE_COLORS.get(lineage, "#333333"),
    )
    for spine in ("top", "right"):
        ax_stack.spines[spine].set_visible(False)
    ax_stack.tick_params(length=1.5, width=0.4)

    # Neutral grey n-cells bar (log10) — replaces the blue-bleeding sideways bar
    ax_n.barh(y, np.log10(n_cells.clip(lower=1).to_numpy()), height=0.85,
              color="#BDBDBD", edgecolor="white", linewidth=0.15)
    ax_n.set_yticks([])
    ax_n.invert_yaxis()
    ax_n.set_xlabel("log10 n", fontsize=5)
    ax_n.set_xlim(0, max(5.5, np.log10(n_cells.max()) * 1.05))
    for spine in ("top", "left", "right"):
        ax_n.spines[spine].set_visible(False)
    ax_n.tick_params(length=1.5, width=0.4, labelsize=4.5)
    ax_n.axvline(0, color="#808080", linewidth=0.4)


def _top_reassignment_flows(df: pd.DataFrame, top_n: int = 18) -> pd.DataFrame:
    """Largest author→HGCA flows among minor/moderate/major reassignment."""
    sub = df[df["refinement_class_taxonomy"].isin(REASSIGNMENT_CLASSES)].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "assigned_lineage", "final_analysis", "closest_GCA_celltype",
                "predicted_hgca_celltype_v1", "refinement_class_taxonomy",
                "lca_depth", "path_dist", "n_cells",
            ]
        )
    group_cols = [
        "assigned_lineage",
        "final_analysis",
        "closest_GCA_celltype",
        "predicted_hgca_celltype_v1",
        "refinement_class_taxonomy",
        "lca_depth",
    ]
    if "path_dist" in sub.columns:
        group_cols.append("path_dist")
    g = (
        sub.groupby(
            group_cols,
            observed=True,
        )
        .size()
        .reset_index(name="n_cells")
        .sort_values("n_cells", ascending=False)
        .head(top_n)
    )
    return g


def _plot_flow_table(ax: plt.Axes, flows: pd.DataFrame) -> None:
    ax.axis("off")
    ax.set_title(
        "Largest reassignment flows (author → HGCA) — graded by path distance",
        loc="left", fontsize=6.5, pad=4,
    )
    if flows.empty:
        ax.text(0.02, 0.5, "No reassignment cells.", fontsize=6, transform=ax.transAxes)
        return

    # Header
    headers = [
        "Lineage", "Severity", "Taurus author (final_analysis)",
        "via closest_GCA", "→ HGCA v1 prediction", "n cells",
    ]
    col_x = [0.00, 0.07, 0.18, 0.46, 0.64, 0.94]
    for x, h in zip(col_x, headers):
        ax.text(
            x, 0.98, h, fontsize=5, weight="bold",
            va="top", transform=ax.transAxes, color="#333333",
            ha="right" if h == "n cells" else "left",
        )

    n_rows = len(flows)
    for i, r in enumerate(flows.itertuples()):
        y = 0.95 - (i + 1) * (0.90 / (n_rows + 0.5))
        lin = str(r.assigned_lineage)
        ax.plot(
            [0.005, 0.050], [y, y],
            color=LINEAGE_COLORS.get(lin, "#999999"),
            linewidth=3.5, solid_capstyle="butt",
            transform=ax.transAxes, clip_on=False,
        )
        severity = _pretty_class(str(r.refinement_class_taxonomy))
        sev_color = REFINEMENT_COLORS.get(
            str(r.refinement_class_taxonomy), "#666666"
        )
        author = str(r.final_analysis)
        if len(author) > 38:
            author = author[:37] + "…"
        gca = str(r.closest_GCA_celltype)
        if len(gca) > 24:
            gca = gca[:23] + "…"
        pred = str(r.predicted_hgca_celltype_v1).replace("\n", " ")
        if len(pred) > 32:
            pred = pred[:31] + "…"
        ax.text(col_x[1], y, severity.replace(" reassignment", ""),
                fontsize=4.4, va="center", transform=ax.transAxes,
                color=sev_color, weight="bold")
        ax.text(col_x[2], y, author, fontsize=4.5, va="center",
                transform=ax.transAxes, color="#111111")
        ax.text(col_x[3], y, gca, fontsize=4.3, va="center",
                transform=ax.transAxes, color="#666666", style="italic")
        ax.text(col_x[4], y, pred, fontsize=4.5, va="center",
                transform=ax.transAxes, color="#111111")
        ax.text(col_x[5], y, f"{int(r.n_cells):,}", fontsize=4.5, va="center",
                ha="right", transform=ax.transAxes, color="#111111",
                family="monospace")


def render(data_dir: Path, out_dir: Path, *, half_width: bool = False) -> None:
    """Render Fig 4b. ``half_width`` → 90 mm single-column, same font sizes."""
    df = _load_cells(data_dir)
    configure_plotting()
    out_dir.mkdir(parents=True, exist_ok=True)
    flows = _top_reassignment_flows(df, top_n=16)

    if half_width:
        # Half of 180 mm; stack 2×2 lineage facets into a 4×1 column.
        fig = plt.figure(figsize=(90 / 25.4, 420 / 25.4))
        fig.subplots_adjust(top=0.970, bottom=0.018, left=0.06, right=0.97)
        outer = fig.add_gridspec(
            3, 1, height_ratios=[22, 280, 50], hspace=0.35,
        )
        ax_sum = fig.add_subplot(outer[0])
        _plot_summary_stack(
            ax_sum, df, legend_ncol=2, legend_bbox=(0.5, -0.55),
        )

        mid = outer[1].subgridspec(4, 1, hspace=0.28)
        for i, lineage in enumerate(LINEAGE_STACK_ORDER):
            cell = mid[i].subgridspec(1, 2, width_ratios=[5.5, 1.0], wspace=0.08)
            ax_s = fig.add_subplot(cell[0, 0])
            ax_n = fig.add_subplot(cell[0, 1], sharey=ax_s)
            frac, n_cells = _author_class_matrix(df, lineage)
            _plot_lineage_facet(ax_s, ax_n, frac, n_cells, lineage)

        ax_flow = fig.add_subplot(outer[2])
        _plot_flow_table(ax_flow, flows)

        fig.text(0.02, 0.992, "b", fontsize=9, weight="bold", ha="left", va="top")
        fig.text(
            0.08, 0.992,
            "Author → HGCA hierarchy classes\n(reassignment by path distance)",
            fontsize=6.5, weight="bold", ha="left", va="top",
        )
        stem = out_dir / "fig4b_author_hgca_concordance_halfwidth"
    else:
        # Tall ED-friendly canvas: summary + 2×2 lineage facets + flow table
        fig = plt.figure(figsize=(180 / 25.4, 220 / 25.4))
        fig.subplots_adjust(top=0.955, bottom=0.03, left=0.02, right=0.985)
        outer = fig.add_gridspec(
            3, 1, height_ratios=[28, 120, 55], hspace=0.55,
        )

        ax_sum = fig.add_subplot(outer[0])
        _plot_summary_stack(ax_sum, df)

        mid = outer[1].subgridspec(2, 2, hspace=0.35, wspace=0.45)
        for i, lineage in enumerate(LINEAGE_STACK_ORDER):
            r, c = divmod(i, 2)
            cell = mid[r, c].subgridspec(1, 2, width_ratios=[5.5, 1.0], wspace=0.08)
            ax_s = fig.add_subplot(cell[0, 0])
            ax_n = fig.add_subplot(cell[0, 1], sharey=ax_s)
            frac, n_cells = _author_class_matrix(df, lineage)
            _plot_lineage_facet(ax_s, ax_n, frac, n_cells, lineage)

        ax_flow = fig.add_subplot(outer[2])
        _plot_flow_table(ax_flow, flows)

        fig.text(0.010, 0.985, "b", fontsize=9, weight="bold", ha="left", va="top")
        fig.text(
            0.030, 0.985,
            "Author → HGCA hierarchy classes (reassignment by path distance)  "
            "(replaces the dense confusion matrix)",
            fontsize=6.5, weight="bold", ha="left", va="top",
        )
        stem = out_dir / "fig4b_author_hgca_concordance"

    l1 = pd.to_numeric(df.get("author_level1_match"), errors="coerce").mean()
    l2 = pd.to_numeric(df.get("author_level2_match"), errors="coerce").mean()
    l3 = pd.to_numeric(df.get("author_level3_match"), errors="coerce").mean()
    fig.text(
        0.02, 0.006 if half_width else 0.012,
        f"Author match levels (Fig 2): L1={100*l1:.1f}%  L2={100*l2:.1f}%  "
        f"L3={100*l3:.1f}% of evaluable cells.  "
        f"Grey bars = log10 cell count per author label.  "
        f"Rows with <{MIN_CELLS_FOR_ROW} cells omitted.",
        fontsize=4.5, color="#444444", ha="left", va="bottom",
    )

    save_figure(fig, stem)
    plt.close(fig)
    LOG.info("wrote %s.{svg,pdf,png}", stem)

    if not half_width:
        flow_path = data_dir / "fig4b_top_reassignment_flows.csv"
        flows.to_csv(flow_path, index=False)
        LOG.info("wrote %s", flow_path)
        flows.to_csv(data_dir / "fig4b_top_changed_branch_flows.csv", index=False)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",
                   default=Path(__file__).resolve().parents[1] / "data",
                   type=Path)
    p.add_argument("--out-dir",
                   default=Path(__file__).resolve().parents[1] / "out",
                   type=Path)
    p.add_argument(
        "--half-width",
        action="store_true",
        help="Also write a 90 mm single-column version (same font sizes).",
    )
    p.add_argument(
        "--half-width-only",
        action="store_true",
        help="Write only the 90 mm single-column version.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")
    args = parse_args(argv)
    if not args.half_width_only:
        render(args.data_dir, args.out_dir, half_width=False)
    if args.half_width or args.half_width_only:
        render(args.data_dir, args.out_dir, half_width=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
