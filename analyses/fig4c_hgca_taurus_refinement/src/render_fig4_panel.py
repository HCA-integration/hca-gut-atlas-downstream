"""
Render HGCA manuscript Fig 4a: HGCA label refinement, cohort support and
transfer certainty for the Taurus Crohn's disease dataset.

Layout (180 mm × 170 mm):

    +---------------------------+---------------------------------------+
    | (a1) label count bars     | (a2) per-cell refinement stacked bars |
    +---------------------------+---------------------------------------+
    | (b)  main heatmap: HGCA identities × sidecars                     |
    |      left column  = lymphoid + stromal (49 rows)                  |
    |      right column = epithelial + myeloid (44 rows)                |
    +-------------------------------------------------------------------+
    | (legend row: refinement classes / continuous scales / flag key)   |
    +-------------------------------------------------------------------+

Inputs are the CSVs produced by
``publication2026/fig4_hgca_taurus_refinement/src/build_fig4_metrics.py``.

Outputs (under ``publication2026/fig4_hgca_taurus_refinement/out/``):
* ``fig4a_hgca_taurus_refinement.svg``
* ``fig4a_hgca_taurus_refinement.pdf``
* ``fig4a_hgca_taurus_refinement.png``

All numbers are directly readable from the CSVs alongside; the panel does not
introduce any threshold beyond those set in ``build_fig4_metrics.py``.
Everything obeys ``publication2026/plot_specs.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Re-use configuration + save helpers from Fig 2 so all panels look identical.
PUB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PUB_ROOT / "fig2_label_set" / "src"))
from build_fig2_atlas_evidence import (  # type: ignore  # noqa: E402
    configure_plotting,
    save_figure,
    _scale01,
)

LOG = logging.getLogger("render_fig4_panel")

# -----------------------------------------------------------------------------
# Palette (Wong; matches publication2026/plot_specs.md § 8)
# -----------------------------------------------------------------------------
PALETTE = {
    "hgca":   "#0072B2",  # HCA blue — transferred HGCA v1
    "author": "#E69F00",  # Wong orange — author final_analysis
    "health": "#009E73",  # bluish green
    "sky":    "#56B4E9",  # sky blue
    "yellow": "#F0E442",
    "purple": "#CC79A7",  # reddish purple — moderate reassignment
    "novel":  "#000000",  # black — reserved for "novel from HGCA" identities
    "grey":   "#999999",
    "light":  "#E0E0E0",
    "warn":   "#D55E00",  # low-support / flag colour (same vermillion)
    "flag_bg": "#F5CBB2",
}
# Hierarchy classes: resolution terms align with Fig 2; former "changed
# branch" is split by LCA depth into minor / moderate / major reassignment.
REFINEMENT_COLORS: Dict[str, str] = {
    "atlas_increased_resolution": "#0072B2",
    "same_resolution_as_author":  "#009E73",
    "atlas_reduced_resolution":   "#F0E442",
    "minor_reassignment_from_author":    "#E8B5D0",  # pale — near-miss subtype
    "moderate_reassignment_from_author": "#CC79A7",  # Wong purple
    "major_reassignment_from_author":    "#882255",  # dark — broad identity flip
    "low_confidence":             "#999999",
    "uncertainty_unavailable":    "#BDBDBD",
    "no_author_crosswalk":        "#E0E0E0",
    "author_absent_from_taxonomy": "#E0E0E0",
    "predicted_absent_from_taxonomy": "#E0E0E0",
    "absent_from_taxonomy":       "#E0E0E0",
    "no_prediction":              "#E0E0E0",
}
REASSIGNMENT_CLASSES: Tuple[str, ...] = (
    "minor_reassignment_from_author",
    "moderate_reassignment_from_author",
    "major_reassignment_from_author",
)
LINEAGE_COLORS: Dict[str, str] = {
    "lymphoid":   "#56B4E9",
    "stroma":     "#CC79A7",
    "epithelial": "#009E73",
    "myeloid":    "#F0E442",
}
LINEAGE_ORDER_LEFT:  Tuple[str, ...] = ("lymphoid", "stroma")
LINEAGE_ORDER_RIGHT: Tuple[str, ...] = ("epithelial", "myeloid")
LINEAGE_STACK_ORDER: Tuple[str, ...] = ("lymphoid", "stroma", "epithelial", "myeloid")

REFINEMENT_STACK_ORDER: Tuple[str, ...] = (
    "atlas_increased_resolution",
    "same_resolution_as_author",
    "atlas_reduced_resolution",
    "minor_reassignment_from_author",
    "moderate_reassignment_from_author",
    "major_reassignment_from_author",
    "low_confidence",
)

# Sidecar layout for the main heatmap.
CONTINUOUS_METRICS: List[Tuple[str, str, Dict[str, bool]]] = [
    # (label,                 df column,                    kwargs to _scale01)
    ("log10 cells",           "n_cells",                    {"log1p": True}),
    ("% of lineage",          "pct_within_lineage",         {}),
    ("Donors",                "n_donors",                   {"log1p": True}),
    ("Samples",               "n_samples",                  {"log1p": True}),
    ("Donor prev. (≥10)",     "donor_prevalence_ge10",      {}),
    ("Sample prev. (≥10)",    "sample_prevalence_ge10",     {}),
    ("Confident (entropy<0.5)", "pct_confident",            {}),
    ("Median entropy (low = confident)",  "median_entropy", {"invert": True}),
]
STATE_METRICS: List[Tuple[str, str]] = [
    ("Healthy",       "present_healthy"),
    ("Pre-treatment", "present_pre_treatment"),
    ("Post-treatment","present_post_treatment"),
]
FLAG_METRICS: List[Tuple[str, str]] = [
    ("< 10 cells / <2 donors / <2 samples", "low_support_flag"),
    ("Single donor only",                   "single_donor_flag"),
    ("Single condition only",               "single_condition_flag"),
]


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def _load_all(data_dir: Path) -> Dict[str, pd.DataFrame]:
    LOG.info("loading tables from %s", data_dir)
    identity = pd.read_csv(data_dir / "fig4a_hgca_identity_evidence.csv")
    label_counts = pd.read_csv(data_dir / "fig4a_lineage_label_counts.csv")
    ref_summary = pd.read_csv(data_dir / "fig4a_refinement_summary.csv")
    with open(data_dir / "fig4a_headline_metrics.json") as fh:
        headline = json.load(fh)
    return {
        "identity": identity,
        "label_counts": label_counts,
        "ref_summary": ref_summary,
        "headline": headline,
    }


def _prepare_identity_frame(identity: pd.DataFrame) -> pd.DataFrame:
    """Order identities: lineage (manuscript order), then descending n_cells."""
    df = identity.copy()
    df["lineage"] = pd.Categorical(
        df["lineage"], categories=LINEAGE_STACK_ORDER, ordered=True
    )
    df = df.sort_values(["lineage", "n_cells"], ascending=[True, False]).reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# Top-strip panels
# -----------------------------------------------------------------------------

def _plot_label_counts(
    ax: plt.Axes,
    label_counts: pd.DataFrame,
    identity: pd.DataFrame,
) -> None:
    """Author ``final_analysis`` counts vs HGCA v1 totals (shared + novel).

    Audited columns come from ``fig4a_lineage_label_counts.csv``:
    ``n_author_labels`` (final_analysis), ``n_hgca_shared``, ``n_hgca_novel``,
    with ``n_hgca_shared + n_hgca_novel == n_hgca_v1_labels`` per scope.
    Shared/novel is relative to the author ``closest_GCA_celltype`` crosswalk.
    """
    del identity  # panel uses the audited label-count table only
    ax.set_title(
        "Label vocabulary by transfer lineage (mapping pickle strata)",
        loc="left", fontsize=6.5, pad=3,
    )
    sub = label_counts.query("author_resolution == 'final_analysis'").copy()
    scope_order = ["overall", *LINEAGE_STACK_ORDER]
    sub = sub.set_index("scope").reindex(scope_order).reset_index()
    if sub[["n_hgca_shared", "n_hgca_novel", "n_hgca_v1_labels"]].isna().any().any():
        raise ValueError(
            "fig4a_lineage_label_counts.csv missing n_hgca_shared/n_hgca_novel; "
            "rebuild with build_fig4_metrics.py"
        )
    if not (
        sub["n_hgca_shared"] + sub["n_hgca_novel"] == sub["n_hgca_v1_labels"]
    ).all():
        bad = sub.loc[
            sub["n_hgca_shared"] + sub["n_hgca_novel"] != sub["n_hgca_v1_labels"],
            ["scope", "n_hgca_shared", "n_hgca_novel", "n_hgca_v1_labels"],
        ]
        raise AssertionError(f"HGCA shared+novel != total:\n{bad}")

    shared_counts = sub["n_hgca_shared"].to_numpy(int)
    novel_counts = sub["n_hgca_novel"].to_numpy(int)
    total_hgca = sub["n_hgca_v1_labels"].to_numpy(int)

    x = np.arange(len(sub))
    w = 0.38
    ax.bar(
        x - w / 2,
        sub["n_author_labels"],
        width=w,
        color=PALETTE["author"],
        label="Author final_analysis",
    )
    ax.bar(
        x + w / 2,
        shared_counts,
        width=w,
        color=PALETTE["hgca"],
        label="HGCA v1 shared (in author closest-GCA crosswalk)",
    )
    ax.bar(
        x + w / 2,
        novel_counts,
        width=w,
        bottom=shared_counts,
        color=PALETTE["novel"],
        label="HGCA v1 novel (absent from author closest-GCA)",
    )

    for i, r in enumerate(sub.itertuples()):
        ax.text(
            i - w / 2,
            r.n_author_labels + 2,
            f"{int(r.n_author_labels)}",
            ha="center",
            va="bottom",
            fontsize=5,
            color=PALETTE["author"],
        )
        ax.text(
            i + w / 2,
            total_hgca[i] + 2,
            f"{int(total_hgca[i])}\n({int(shared_counts[i])}+{int(novel_counts[i])})",
            ha="center",
            va="bottom",
            fontsize=4.8,
            color="#333333",
            linespacing=0.95,
        )

    ax.set_xticks(x)
    xtick_labels = [
        f"{s.capitalize()}\nn={int(sub['n_hgca_cells'].iloc[i]):,}"
        for i, s in enumerate(sub["scope"])
    ]
    ax.set_xticklabels(xtick_labels, fontsize=5, ha="center")
    ax.set_ylabel("Unique labels", fontsize=6)
    ax.set_ylim(0, max(sub["n_author_labels"].max(), total_hgca.max()) * 1.38)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=2, width=0.5)
    ax.legend(
        loc="upper right",
        fontsize=4.5,
        frameon=False,
        handlelength=0.9,
        handleheight=0.8,
        labelspacing=0.25,
    )


def _plot_refinement_stack(
    ax: plt.Axes,
    ref_summary: pd.DataFrame,
    *,
    legend_ncol: int = 4,
    legend_bbox: Tuple[float, float] = (0.5, -0.55),
) -> None:
    """Stacked-bar of per-cell refinement classes as % of lineage cells."""
    ax.set_title(
        "Per-cell hierarchy class (reassignment graded by path distance)",
        loc="left", fontsize=7, pad=3,
    )
    scope_order = ["overall", *LINEAGE_STACK_ORDER]
    matrix = (
        ref_summary.pivot(index="scope", columns="refinement_class", values="pct_of_scope")
        .reindex(scope_order)
        .fillna(0.0)
    )
    denom = ref_summary.pivot(
        index="scope", columns="refinement_class", values="n_cells_in_scope"
    ).reindex(scope_order).iloc[:, 0].astype(int)  # any col; they're all equal

    x = np.arange(len(matrix))
    left = np.zeros(len(matrix))
    for cls in REFINEMENT_STACK_ORDER:
        vals = matrix[cls].values if cls in matrix.columns else np.zeros(len(matrix))
        ax.barh(x, vals, left=left, height=0.62,
                color=REFINEMENT_COLORS[cls], label=_pretty_class(cls))
        left = left + vals
    ax.set_yticks(x)
    labels = [
        f"{s.capitalize()}\n(n={int(denom[s]):,})" for s in matrix.index
    ]
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of cells", fontsize=6)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=2, width=0.5)
    ax.legend(
        loc="lower center", bbox_to_anchor=legend_bbox, ncol=legend_ncol,
        fontsize=4.8, frameon=False, handlelength=0.9, handleheight=0.8,
        columnspacing=0.7, labelspacing=0.25,
    )


def _pretty_class(cls: str) -> str:
    return {
        "atlas_increased_resolution": "Atlas increased resolution",
        "same_resolution_as_author":  "Same resolution as author",
        "atlas_reduced_resolution":   "Atlas reduced resolution",
        "minor_reassignment_from_author":    "Minor reassignment",
        "moderate_reassignment_from_author": "Moderate reassignment",
        "major_reassignment_from_author":    "Major reassignment",
        "changed_branch_from_author": "Major reassignment",  # legacy
        "low_confidence":             "Low confidence",
        "no_author_crosswalk":        "No author crosswalk",
    }.get(cls, cls.replace("_", " ").capitalize())


# -----------------------------------------------------------------------------
# Main heatmap
# -----------------------------------------------------------------------------

def _identity_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the three visual blocks:
        - continuous evidence matrix (identities × CONTINUOUS_METRICS)
        - state-presence matrix (identities × STATE_METRICS)  (0/1)
        - flag matrix (identities × FLAG_METRICS)              (0/1)
    Each column of the continuous block is _scale01'd within that column,
    matching Fig 2's `plot_celltype_sidecar` convention.
    """
    cont = pd.DataFrame(
        {
            label: _scale01(df[col], **kw)
            for label, col, kw in CONTINUOUS_METRICS
        }
    )
    state = pd.DataFrame(
        {label: df[col].astype(float) for label, col in STATE_METRICS}
    )
    flag = pd.DataFrame(
        {label: df[col].astype(float) for label, col in FLAG_METRICS}
    )
    return cont, state, flag


def _draw_heatmap_column(
    fig: plt.Figure,
    gs: mpl.gridspec.GridSpec,
    gs_row: int,
    df: pd.DataFrame,
    *,
    title: str,
) -> None:
    """Draw one lineage-block heatmap column.

    Column layout inside its sub-gridspec:
        [row labels] [refinement badge] [continuous evidence]
        [state presence] [flags] [reserved PanGI]
    """
    n_rows = len(df)
    # per-block width ratios (fig-relative widths inside this column).
    # Column order (left→right):
    #   labels · refinement badge · **novel from HGCA** · continuous evidence
    #   · state presence · robustness flags · reserved PanGI
    width_ratios = [
        32,                                 # row labels + lineage bar (wider
                                            # to fit "identity + (was X)")
        3,                                  # refinement badge
        3,                                  # novel-from-HGCA badge
        4 * len(CONTINUOUS_METRICS),        # continuous evidence
        3 * len(STATE_METRICS),             # state presence
        3 * len(FLAG_METRICS),              # flags
        3,                                  # reserved PanGI
    ]
    sub = gs[gs_row].subgridspec(
        1, len(width_ratios), width_ratios=width_ratios, wspace=0.10,
    )

    # --- Row labels + lineage color bar --------------------------------------
    # Two sub-columns of the label block: (i) HGCA identity right-aligned,
    # (ii) small italic grey "author-parent" note left-aligned + a lineage
    # colored tick. Full author-parent list is in the CSV.
    ax_labels = fig.add_subplot(sub[0, 0])
    ax_labels.set_xlim(0, 1)
    ax_labels.set_ylim(-0.5, n_rows - 0.5)
    ax_labels.invert_yaxis()
    ax_labels.set_xticks([]); ax_labels.set_yticks([])
    for spine in ax_labels.spines.values():
        spine.set_visible(False)

    # Split: identity in the *left ~44%* right-aligned; author-parent in the
    # *right ~53%* left-aligned italic grey; lineage tick at the extreme right.
    # The label axis now has more physical width (width_ratios[0]=32), so
    # italic parent labels can render without clipping into the badge column.
    identity_x = 0.44
    parent_x   = 0.46
    tick_x     = 0.985
    for i, r in enumerate(df.itertuples()):
        ax_labels.add_patch(mpatches.Rectangle(
            (tick_x, i - 0.5), 1 - tick_x, 1,
            color=LINEAGE_COLORS[r.lineage],
            linewidth=0, clip_on=False, alpha=0.9,
        ))
        identity = str(r.hgca_celltype_v1)
        ax_labels.text(identity_x, i, identity, ha="right", va="center",
                       fontsize=5.2, color="#111111")

        author = str(getattr(r, "majority_author_parent") or "").strip()
        if author and author != identity:
            author_short = (author[:11] + "…") if len(author) > 12 else author
            ax_labels.text(parent_x, i, f"(was {author_short})",
                           ha="left", va="center", fontsize=4.3,
                           color="#7A7A7A", style="italic")
    ax_labels.set_title(title, loc="left", fontsize=6.5, pad=3,
                        x=identity_x - 0.5)

    # --- Refinement badge (categorical single column) ------------------------
    ax_bad = fig.add_subplot(sub[0, 1], sharey=ax_labels)
    ax_bad.set_xlim(0, 1); ax_bad.set_xticks([]); ax_bad.set_yticks([])
    for i, cls in enumerate(df["majority_refinement_class"]):
        ax_bad.add_patch(mpatches.Rectangle(
            (0.05, i - 0.4), 0.9, 0.8,
            facecolor=REFINEMENT_COLORS.get(cls, PALETTE["light"]),
            edgecolor="white", linewidth=0.25,
        ))
    _style_bare_axes(ax_bad)

    # --- Novel-from-HGCA badge ---------------------------------------------
    ax_novel = fig.add_subplot(sub[0, 2], sharey=ax_labels)
    ax_novel.set_xlim(0, 1); ax_novel.set_xticks([]); ax_novel.set_yticks([])
    for i, is_novel in enumerate(df["novel_hgca_identity"]):
        color = PALETTE["novel"] if bool(is_novel) else "#F1F1F1"
        ax_novel.add_patch(mpatches.Rectangle(
            (0.05, i - 0.4), 0.9, 0.8,
            facecolor=color, edgecolor="#CCCCCC", linewidth=0.25,
        ))
    _style_bare_axes(ax_novel)

    # --- Continuous evidence heat block --------------------------------------
    cont, state, flag = _identity_matrix(df)
    ax_ev = fig.add_subplot(sub[0, 3], sharey=ax_labels)
    _draw_continuous_block(ax_ev, cont)

    # --- State presence block (dark = present) ------------------------------
    ax_st = fig.add_subplot(sub[0, 4], sharey=ax_labels)
    _draw_binary_block(ax_st, state, present_color=PALETTE["hgca"])

    # --- Flag block (vermillion = flagged, grey = fine) ---------------------
    ax_fl = fig.add_subplot(sub[0, 5], sharey=ax_labels)
    _draw_binary_block(ax_fl, flag, present_color=PALETTE["warn"], absent_color="#F1F1F1")

    # --- Reserved PanGI slot -------------------------------------------------
    ax_pg = fig.add_subplot(sub[0, 6], sharey=ax_labels)
    ax_pg.set_xlim(0, 1); ax_pg.set_ylim(-0.5, n_rows - 0.5)
    ax_pg.invert_yaxis()
    ax_pg.add_patch(mpatches.Rectangle(
        (0, -0.5), 1, n_rows,
        facecolor="#FAFAFA", edgecolor="#DDDDDD", linewidth=0.4,
        hatch="////", alpha=0.6,
    ))
    ax_pg.text(0.5, n_rows / 2, "PanGI\npending",
               ha="center", va="center", fontsize=4.8,
               color="#777777", rotation=90)
    _style_bare_axes(ax_pg)


def _draw_continuous_block(ax: plt.Axes, cont: pd.DataFrame) -> None:
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "gca_blue", [PALETTE["light"], PALETTE["hgca"]]
    )
    ax.imshow(cont.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=1,
              interpolation="none")
    ax.set_xticks(np.arange(cont.shape[1]))
    ax.set_xticklabels(cont.columns, rotation=90, fontsize=4.8)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=1.5, pad=1, width=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)


def _draw_binary_block(
    ax: plt.Axes,
    mat: pd.DataFrame,
    *,
    present_color: str,
    absent_color: str = "#F1F1F1",
) -> None:
    cmap = mpl.colors.ListedColormap([absent_color, present_color])
    ax.imshow(mat.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=1,
              interpolation="none")
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=90, fontsize=4.8)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=1.5, pad=1, width=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)


def _style_bare_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
    ax.set_xticks([])
    ax.set_yticks([])


# -----------------------------------------------------------------------------
# Legend row
# -----------------------------------------------------------------------------

def _plot_legend(ax: plt.Axes, *, narrow: bool = False) -> None:
    """Legend row. Full-width: side-by-side blocks. Narrow (90 mm): stacked."""
    ax.axis("off")

    refinement_handles = [
        mpatches.Patch(color=REFINEMENT_COLORS[c], label=_pretty_class(c))
        for c in REFINEMENT_STACK_ORDER
    ]
    lineage_handles = [
        mpatches.Patch(color=LINEAGE_COLORS[l], label=l.capitalize())
        for l in LINEAGE_STACK_ORDER
    ]
    state_key = [
        mpatches.Patch(color=PALETTE["hgca"], label="Present"),
        mpatches.Patch(facecolor="#F1F1F1", edgecolor="#CCCCCC",
                       linewidth=0.5, label="Absent"),
    ]
    flag_key = [
        mpatches.Patch(color=PALETTE["warn"], label="Flagged"),
        mpatches.Patch(facecolor="#F1F1F1", edgecolor="#CCCCCC",
                       linewidth=0.5, label="Passes"),
    ]
    novel_key = [
        mpatches.Patch(color=PALETTE["novel"],
                       label="Novel from HGCA (not in Taurus vocab.)"),
        mpatches.Patch(facecolor="#F1F1F1", edgecolor="#CCCCCC",
                       linewidth=0.5, label="Shared vocabulary"),
    ]

    if narrow:
        l1 = ax.legend(
            handles=refinement_handles, loc="upper left", bbox_to_anchor=(0.0, 1.0),
            fontsize=4.8, frameon=False, title="Hierarchy class",
            title_fontsize=5.5, handlelength=1.0, handleheight=0.8, ncol=2,
            labelspacing=0.25, columnspacing=0.8, borderpad=0,
        )
        l1._legend_box.align = "left"
        ax.add_artist(l1)
        l2 = ax.legend(
            handles=lineage_handles + state_key + flag_key + novel_key,
            loc="upper left", bbox_to_anchor=(0.0, 0.42),
            fontsize=4.8, frameon=False, title="Lineage / state / flag / novel",
            title_fontsize=5.5, handlelength=1.0, handleheight=0.8, ncol=2,
            labelspacing=0.25, columnspacing=0.8, borderpad=0,
        )
        l2._legend_box.align = "left"
        ax.add_artist(l2)
        grad_ax = ax.inset_axes([0.62, 0.08, 0.35, 0.10])
        grad_ax.imshow(
            np.linspace(0, 1, 256).reshape(1, -1), aspect="auto",
            cmap=mpl.colors.LinearSegmentedColormap.from_list(
                "gca_blue", [PALETTE["light"], PALETTE["hgca"]],
            ),
        )
        grad_ax.set_xticks([0, 255])
        grad_ax.set_xticklabels(["min", "max"], fontsize=4.5)
        grad_ax.set_yticks([])
        ax.text(
            0.0, -0.05,
            "Reassignment by path distance: minor ≤3, moderate=4, major ≥5 edges.\n"
            "Italic \"was X\" = majority author label. Entropy unavailable in this export.",
            ha="left", va="top", fontsize=4.6, color="#333333",
            transform=ax.transAxes,
        )
        return

    # --- full-width blocks -------------------------------------------------
    l1 = ax.legend(
        handles=refinement_handles, loc="upper left", bbox_to_anchor=(0.0, 1.0),
        fontsize=4.8, frameon=False, title="Hierarchy class (badge + top bars)",
        title_fontsize=5.5, handlelength=1.0, handleheight=0.8, ncol=1,
        labelspacing=0.28, borderpad=0,
    )
    l1._legend_box.align = "left"
    ax.add_artist(l1)

    l2 = ax.legend(
        handles=lineage_handles, loc="upper left", bbox_to_anchor=(0.26, 1.0),
        fontsize=5, frameon=False, title="Lineage tick",
        title_fontsize=5.5, handlelength=1.0, handleheight=0.8, ncol=1,
        labelspacing=0.35, borderpad=0,
    )
    l2._legend_box.align = "left"
    ax.add_artist(l2)

    l3 = ax.legend(
        handles=state_key, loc="upper left", bbox_to_anchor=(0.42, 1.0),
        fontsize=5, frameon=False, title="State presence",
        title_fontsize=5.5, handlelength=1.0, handleheight=0.8, ncol=1,
        labelspacing=0.35, borderpad=0,
    )
    l3._legend_box.align = "left"
    ax.add_artist(l3)

    l3b = ax.legend(
        handles=flag_key, loc="upper left", bbox_to_anchor=(0.52, 1.0),
        fontsize=5, frameon=False, title="Robustness flag",
        title_fontsize=5.5, handlelength=1.0, handleheight=0.8, ncol=1,
        labelspacing=0.35, borderpad=0,
    )
    l3b._legend_box.align = "left"
    ax.add_artist(l3b)

    l3c = ax.legend(
        handles=novel_key, loc="upper left", bbox_to_anchor=(0.60, 1.0),
        fontsize=5, frameon=False, title="Novel identity",
        title_fontsize=5.5, handlelength=1.0, handleheight=0.8, ncol=1,
        labelspacing=0.35, borderpad=0,
    )
    l3c._legend_box.align = "left"
    ax.add_artist(l3c)

    grad_ax = ax.inset_axes([0.74, 0.55, 0.22, 0.18])
    grad_ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), aspect="auto",
                   cmap=mpl.colors.LinearSegmentedColormap.from_list(
                       "gca_blue", [PALETTE["light"], PALETTE["hgca"]]))
    grad_ax.set_xticks([0, 255])
    grad_ax.set_xticklabels(["min", "max"], fontsize=4.5)
    grad_ax.set_yticks([])
    grad_ax.tick_params(axis="x", length=1.5, pad=1, width=0.4)
    for spine in grad_ax.spines.values():
        spine.set_linewidth(0.4)
    ax.text(0.74, 0.85, "Evidence (column-scaled 0–1)",
            ha="left", va="bottom", fontsize=5.5, transform=ax.transAxes)

    ax.text(
        0.74, 0.32,
        "Each row = one HGCA v1 identity called in Taurus. Italic grey after\n"
        "the identity name (\"was X\") is the majority Taurus author label those\n"
        "cells previously carried. Branch changes graded by tree path distance:\n"
        "minor (≤3 edges), moderate (4), major (≥5).\n"
        "Low-confidence overlay requires SCANVI entropy; the 2026-07-22 hard-\n"
        "label export did not include entropy (99.97% uncertainty unavailable).\n"
        "Author bars: Wong orange #E69F00; HGCA bars: blue #0072B2.",
        ha="left", va="top", fontsize=4.6, color="#333333",
        transform=ax.transAxes,
    )


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def render(data_dir: Path, out_dir: Path, *, half_width: bool = False) -> None:
    """Render Fig 4a.

    ``half_width=True`` writes a single-column (90 mm) version with the same
    point sizes; side-by-side blocks are stacked vertically and height grows.
    """
    tables = _load_all(data_dir)
    identity = _prepare_identity_frame(tables["identity"])

    left  = identity[identity["lineage"].isin(LINEAGE_ORDER_LEFT)].reset_index(drop=True)
    right = identity[identity["lineage"].isin(LINEAGE_ORDER_RIGHT)].reset_index(drop=True)

    configure_plotting()
    out_dir.mkdir(parents=True, exist_ok=True)

    if half_width:
        # 90 mm wide (half of 180 mm); taller so stacked panels stay readable.
        # Font sizes are unchanged (configure_plotting + explicit fontsize=).
        width_mm, height_mm = 90.0, 340.0
        fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
        fig.subplots_adjust(top=0.965, bottom=0.015, left=0.04, right=0.97)
        outer = fig.add_gridspec(
            5, 1,
            height_ratios=[28, 32, 110, 110, 28],
            hspace=0.55,
        )
        ax_counts = fig.add_subplot(outer[0])
        ax_stack = fig.add_subplot(outer[1])
        _plot_label_counts(ax_counts, tables["label_counts"], identity)
        _plot_refinement_stack(
            ax_stack, tables["ref_summary"],
            legend_ncol=2, legend_bbox=(0.5, -0.72),
        )

        left_gs = outer[2].subgridspec(1, 1)
        right_gs = outer[3].subgridspec(1, 1)
        _draw_heatmap_column(
            fig, left_gs, 0, left,
            title=f"Lymphoid + stromal ({len(left)} HGCA identities)",
        )
        _draw_heatmap_column(
            fig, right_gs, 0, right,
            title=f"Epithelial + myeloid ({len(right)} HGCA identities)",
        )

        ax_leg = fig.add_subplot(outer[4])
        _plot_legend(ax_leg, narrow=True)

        fig.text(0.02, 0.990, "a", fontsize=9, weight="bold", ha="left", va="top")
        fig.text(
            0.07, 0.990,
            "HGCA v1 label refinement in the Taurus Crohn's cohort\n"
            "(n=987,743 cells, 41 donors, 216 samples)",
            fontsize=6.5, weight="bold", ha="left", va="top",
        )
        stem = out_dir / "fig4a_hgca_taurus_refinement_halfwidth"
    else:
        fig = plt.figure(figsize=(180 / 25.4, 170 / 25.4))
        # top margin ~ 6 mm reserved for the header ("a  ...")
        fig.subplots_adjust(top=0.945, bottom=0.02, left=0.015, right=0.985)
        outer = fig.add_gridspec(
            3, 1,
            height_ratios=[38, 122, 18],
            hspace=0.55,
        )
        top = outer[0].subgridspec(1, 2, width_ratios=[0.55, 1.0], wspace=0.28)
        ax_counts = fig.add_subplot(top[0, 0])
        ax_stack  = fig.add_subplot(top[0, 1])
        _plot_label_counts(ax_counts, tables["label_counts"], identity)
        _plot_refinement_stack(ax_stack, tables["ref_summary"])

        main = outer[1].subgridspec(1, 2, width_ratios=[len(left), len(right)],
                                    wspace=0.15)
        _draw_heatmap_column(fig, main, 0, left,
                             title=f"Lymphoid + stromal ({len(left)} HGCA identities)")
        _draw_heatmap_column(fig, main, 1, right,
                             title=f"Epithelial + myeloid ({len(right)} HGCA identities)")

        ax_leg = fig.add_subplot(outer[2])
        _plot_legend(ax_leg)

        # Panel label + short header on one line (Nature panel-label convention).
        # Placed in the 6 mm top margin so it never crashes into subplot titles.
        fig.text(0.010, 0.985, "a", fontsize=9, weight="bold", ha="left", va="top")
        fig.text(0.030, 0.985,
                 "HGCA v1 label refinement in the Taurus Crohn's cohort  "
                 "(n=987,743 cells, 41 donors, 216 samples)",
                 fontsize=6.5, weight="bold", ha="left", va="top")
        stem = out_dir / "fig4a_hgca_taurus_refinement"

    save_figure(fig, stem)
    plt.close(fig)
    LOG.info("wrote %s.{svg,pdf,png}", stem)


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
