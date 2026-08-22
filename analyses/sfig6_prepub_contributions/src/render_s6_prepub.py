#!/usr/bin/env python3
"""Render Supplementary Figure 6 — prepublication contributions strengthen the atlas.

Source analysis:
  hca-gut-atlas-downstream/vignettes/Prepub_vs_postPub.ipynb
  Prior PDF exports: ~/Projects/GCA/github_vignette_output/Prepub_vs_postPub/

Panels (plot_specs.md: Wong palette, Helvetica/Arial 5–7 pt, 90/180 mm):
  a  Provenance cell counts by lineage (stacked bars)
  b  Provenance by lineage × segment (small multiples)
  c  Lymphoid coverage by segment (stacked bars)
  d  Myeloid coverage by segment (stacked bars)
  e  Myeloid published-only vs published+contributed by segment (grouped bars)
  f  Ileum-vs-colon cell-level Wilcoxon power curve (kept for comparison)
  g  Ileum-vs-colon pseudobulk DESeq2 power curve (~ dataset_id + seg)

Usage:
  ~/miniforge3/envs/scanpy/bin/python src/render_s6_prepub.py
  ~/miniforge3/envs/scanpy/bin/python src/render_s6_prepub.py --skip-power
  ~/miniforge3/envs/patpy/bin/python src/compute_deseq2_power.py
  ~/miniforge3/envs/scanpy/bin/python src/render_s6_prepub.py --skip-bars --skip-power --plot-deseq2
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG = logging.getLogger("render_s6")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
OUT = ROOT / "out"

CAP_DIR = Path(os.environ["HGCA_CAP_DIR"]) if os.environ.get("HGCA_CAP_DIR") else None
_OBJECTS = Path(os.environ["HGCA_OBJECTS"]) if os.environ.get("HGCA_OBJECTS") else None
V1_MYELOID = (_OBJECTS / "myeloid.h5ad") if _OBJECTS is not None else None

PATHS = (
    {
        "epithelial": CAP_DIR / "epithelial.h5ad",
        "lymphoid": CAP_DIR / "lymphoid.h5ad",
        "myeloid": CAP_DIR / "myeloid.h5ad",
        "stroma": CAP_DIR / "stroma.h5ad",
    }
    if CAP_DIR is not None
    else {}
)

# Provenance IDs used for bar panels (for_cap objects) and power curve (v1 object).
PREPUB_IDS = {
    "ArendsHelmsley",
    "DominguezUnpub2",
    "DominguezUnpub",
    "BasuGCARNA",
    "BasuHelmsley",
    "HamiltonHelmsley",
    "KarakashevaHelmlsey",
}

TISSUE_COL = "tissue_ontology_term"
CT_COL = "Prelim annotation"  # bar-panel objects (for_cap)
DATASET_COL = "dataset_id"

SEG_ORDER = ["duodenum", "jejunum", "ileum", "colon"]
LINEAGES = ["epithelial", "lymphoid", "myeloid", "stroma"]
PROV_ORDER = ["Published studies", "Consortium contributed"]

PAL = {
    "black": "#000000",
    "vermillion": "#D55E00",
    "hca_blue": "#0072B2",
    "bluish_green": "#009E73",
    "sky_blue": "#56B4E9",
    "light_grey": "#E0E0E0",
    "mid_grey": "#999999",
}
PAL_PROVENANCE = {
    "Published studies": PAL["vermillion"],
    "Consortium contributed": PAL["hca_blue"],
    "Published only": PAL["vermillion"],
    "Published plus contributed": PAL["hca_blue"],
}

UBERON_TO_SEGMENT = {
    "UBERON:0002114": "duodenum",
    "UBERON:0002115": "jejunum",
    "UBERON:0002116": "ileum",
    "UBERON:0001155": "colon",
}

# Panel f: ileum vs colon power curve on HGCA v1 labels
POWER_CELLTYPES = [
    ("cDC1", "cDC1"),
    ("cDC2", "cDC2"),
    ("Homeostatic Macrophages", "Hom. mac"),
]
POWER_SEG_A = "ileum"
POWER_SEG_B = "colon"
POWER_FDR = 0.05
POWER_N_SEEDS = 5
POWER_N_GRID = [50, 75, 100, 150, 200, 300, 400, 600, 800, 1000, 1200, 1500, 2000]


def configure_plotting(font_size_pt: float = 6) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": font_size_pt,
            "axes.titlesize": min(font_size_pt + 1, 7),
            "axes.labelsize": font_size_pt,
            "xtick.labelsize": font_size_pt,
            "ytick.labelsize": font_size_pt,
            "legend.fontsize": font_size_pt,
            "legend.title_fontsize": font_size_pt,
            "text.color": PAL["black"],
            "axes.labelcolor": PAL["black"],
            "axes.edgecolor": PAL["black"],
            "axes.titlecolor": PAL["black"],
            "xtick.color": PAL["black"],
            "ytick.color": PAL["black"],
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def mm_to_in(mm: float) -> float:
    return mm / 25.4


def figure_nature(
    width_mm: float = 90,
    height_mm: float = 60,
    nrows: int = 1,
    ncols: int = 1,
    **kwargs,
):
    configure_plotting()
    return plt.subplots(
        nrows,
        ncols,
        figsize=(mm_to_in(width_mm), mm_to_in(height_mm)),
        **kwargs,
    )


def save_figure(fig: mpl.figure.Figure, stem: str, *, dpi: int = 300) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / stem
    for suffix in (".pdf", ".svg"):
        fig.savefig(base.with_suffix(suffix), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    LOG.info("saved %s.{pdf,svg,png}", stem)


def segment_from_tissue(x) -> str:
    s = str(x)
    if s in UBERON_TO_SEGMENT:
        return UBERON_TO_SEGMENT[s]
    sl = s.lower()
    if "duod" in sl:
        return "duodenum"
    if "jejun" in sl:
        return "jejunum"
    if "ile" in sl:
        return "ileum"
    if "colon" in sl or "large intestine" in sl:
        return "colon"
    return "other"


def load_obs(path: Path) -> pd.DataFrame:
    """Load obs only (backed) for provenance bar panels."""
    import scanpy as sc

    LOG.info("loading obs from %s", path.name)
    ad = sc.read_h5ad(path, backed="r")
    obs = ad.obs.copy()
    ad.file.close()
    needed = [TISSUE_COL, DATASET_COL]
    missing = [c for c in needed if c not in obs.columns]
    if missing:
        raise KeyError(f"{path.name} missing columns: {missing}")
    out = obs[needed].copy()
    out["segment_simple"] = out[TISSUE_COL].map(segment_from_tissue)
    out["data_provenance"] = np.where(
        out[DATASET_COL].astype(str).isin(PREPUB_IDS),
        "Consortium contributed",
        "Published studies",
    )
    out = out[out["segment_simple"].isin(SEG_ORDER)].copy()
    return out


# -----------------------------------------------------------------------------
# Panel a–e: provenance bars
# -----------------------------------------------------------------------------
def compute_provenance_tables(obs_by_lineage: dict[str, pd.DataFrame]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)

    # By lineage
    counts = pd.DataFrame(
        {
            prov: {
                lin: int((obs_by_lineage[lin]["data_provenance"] == prov).sum())
                for lin in LINEAGES
            }
            for prov in PROV_ORDER
        },
        index=LINEAGES,
    )
    counts["total"] = counts.sum(axis=1)
    counts["pct_contributed"] = (
        100 * counts["Consortium contributed"] / counts["total"].clip(lower=1)
    )
    counts.to_csv(DATA / "provenance_by_lineage.csv")
    LOG.info("\n%s", counts)

    # By lineage × segment
    rows = []
    for lin in LINEAGES:
        tab = (
            pd.crosstab(
                obs_by_lineage[lin]["segment_simple"],
                obs_by_lineage[lin]["data_provenance"],
            )
            .reindex(SEG_ORDER)
            .fillna(0)
            .reindex(columns=PROV_ORDER, fill_value=0)
        )
        for seg in SEG_ORDER:
            for prov in PROV_ORDER:
                rows.append(
                    {
                        "lineage": lin,
                        "segment": seg,
                        "provenance": prov,
                        "n_cells": int(tab.loc[seg, prov]),
                    }
                )
    pd.DataFrame(rows).to_csv(DATA / "provenance_by_lineage_segment.csv", index=False)


def plot_provenance_by_lineage() -> None:
    counts = pd.read_csv(DATA / "provenance_by_lineage.csv", index_col=0)
    fig, ax = figure_nature(width_mm=90, height_mm=65)
    x = np.arange(len(LINEAGES))
    bottom = np.zeros(len(LINEAGES))
    for prov in PROV_ORDER:
        vals = counts.loc[LINEAGES, prov].to_numpy(dtype=float)
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=PAL_PROVENANCE[prov],
            edgecolor=PAL["black"],
            linewidth=0.25,
            label=prov,
        )
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(LINEAGES, rotation=45, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Cell count")
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _: f"{int(v):,}")
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), title="")
    fig.tight_layout()
    save_figure(fig, "s6_a_provenance_by_lineage")
    plt.close(fig)


def plot_provenance_by_lineage_segment() -> None:
    long = pd.read_csv(DATA / "provenance_by_lineage_segment.csv")
    fig, axes = figure_nature(
        width_mm=180, height_mm=60, ncols=len(LINEAGES), sharey=True
    )
    x = np.arange(len(SEG_ORDER))
    for ax, lin in zip(axes, LINEAGES):
        sub = long[long["lineage"] == lin]
        tab = (
            sub.pivot(index="segment", columns="provenance", values="n_cells")
            .reindex(SEG_ORDER)
            .reindex(columns=PROV_ORDER, fill_value=0)
            .fillna(0)
        )
        bottom = np.zeros(len(SEG_ORDER))
        for prov in PROV_ORDER:
            vals = tab[prov].to_numpy(dtype=float)
            ax.bar(
                x,
                vals,
                bottom=bottom,
                color=PAL_PROVENANCE[prov],
                edgecolor=PAL["black"],
                linewidth=0.25,
                label=prov,
            )
            bottom += vals
        ax.set_title(lin)
        ax.set_xticks(x)
        ax.set_xticklabels(SEG_ORDER, rotation=45, ha="right")
        ax.set_xlabel("")
    axes[0].set_ylabel("Cell count")
    axes[0].yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _: f"{int(v):,}")
    )
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), title="")
    fig.tight_layout()
    save_figure(fig, "s6_b_provenance_by_lineage_and_segment")
    plt.close(fig)


def plot_gap_coverage(lineage: str, stem: str, title: str) -> None:
    long = pd.read_csv(DATA / "provenance_by_lineage_segment.csv")
    sub = long[long["lineage"] == lineage]
    tab = (
        sub.pivot(index="segment", columns="provenance", values="n_cells")
        .reindex(SEG_ORDER)
        .reindex(columns=PROV_ORDER, fill_value=0)
        .fillna(0)
    )
    fig, ax = figure_nature(width_mm=90, height_mm=60)
    x = np.arange(len(SEG_ORDER))
    bottom = np.zeros(len(SEG_ORDER))
    for prov in PROV_ORDER:
        vals = tab[prov].to_numpy(dtype=float)
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=PAL_PROVENANCE[prov],
            edgecolor=PAL["black"],
            linewidth=0.25,
            label=prov,
        )
        bottom += vals
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(SEG_ORDER, rotation=45, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Cell count")
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _: f"{int(v):,}")
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), title="")
    fig.tight_layout()
    save_figure(fig, stem)
    plt.close(fig)


def plot_myeloid_published_vs_all(obs_myeloid: pd.DataFrame) -> None:
    is_pub = obs_myeloid["data_provenance"] == "Published studies"
    tab_pub = (
        obs_myeloid.loc[is_pub, "segment_simple"]
        .value_counts()
        .reindex(SEG_ORDER)
        .fillna(0)
    )
    tab_all = obs_myeloid["segment_simple"].value_counts().reindex(SEG_ORDER).fillna(0)
    df = pd.DataFrame(
        {
            "Published only": tab_pub,
            "Published plus contributed": tab_all,
        }
    )
    df.to_csv(DATA / "myeloid_published_vs_all_by_segment.csv")

    fig, ax = figure_nature(width_mm=90, height_mm=60)
    x = np.arange(len(SEG_ORDER))
    bar_w = 0.4
    ax.bar(
        x - bar_w / 2,
        df["Published only"].to_numpy(),
        width=bar_w,
        color=PAL_PROVENANCE["Published only"],
        edgecolor=PAL["black"],
        linewidth=0.25,
        label="Published only",
    )
    ax.bar(
        x + bar_w / 2,
        df["Published plus contributed"].to_numpy(),
        width=bar_w,
        color=PAL_PROVENANCE["Published plus contributed"],
        edgecolor=PAL["black"],
        linewidth=0.25,
        label="Published plus contributed",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(SEG_ORDER, rotation=45, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Cells")
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _: f"{int(v):,}")
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), title="")
    fig.tight_layout()
    save_figure(fig, "s6_e_myeloid_published_vs_contributed_by_segment")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Panel f: classic DE power curve (ileum vs colon, HGCA v1 labels)
# -----------------------------------------------------------------------------
def _clean_ct(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"[\r\n]+", " ", regex=True).str.strip()


def load_myeloid_v1_for_power():
    """Load HGCA v1 myeloid object for ileum–colon power curve."""
    import scanpy as sc

    if V1_MYELOID is None or not V1_MYELOID.exists():
        raise FileNotFoundError("Set HGCA_OBJECTS to a directory that contains myeloid.h5ad")
    LOG.info("loading HGCA v1 myeloid for power curve: %s", V1_MYELOID)
    ad = sc.read_h5ad(V1_MYELOID)
    ad.obs["segment"] = ad.obs["tissue_level_1"].astype(str)
    ad.obs["ct"] = _clean_ct(ad.obs["hgca_celltype_v1"])
    ad.obs["provenance"] = np.where(
        ad.obs[DATASET_COL].astype(str).isin(PREPUB_IDS),
        "Consortium contributed",
        "Published studies",
    )
    ad = ad[ad.obs["segment"].isin([POWER_SEG_A, POWER_SEG_B])].copy()
    if "log1p" in ad.uns:
        del ad.uns["log1p"]
    return ad


def _n_de_balanced(sub, n_per: int, seed: int, fdr: float = POWER_FDR) -> float:
    import scanpy as sc

    rng = np.random.default_rng(seed)
    a = np.where(sub.obs["segment"].to_numpy() == POWER_SEG_A)[0]
    b = np.where(sub.obs["segment"].to_numpy() == POWER_SEG_B)[0]
    if len(a) < n_per or len(b) < n_per:
        return np.nan
    idx = np.concatenate(
        [rng.choice(a, n_per, replace=False), rng.choice(b, n_per, replace=False)]
    )
    s = sub[idx].copy()
    if "log1p" in s.uns:
        del s.uns["log1p"]
    sc.pp.normalize_total(s, target_sum=1e4)
    sc.pp.log1p(s)
    sc.tl.rank_genes_groups(
        s,
        groupby="segment",
        groups=[POWER_SEG_A],
        reference=POWER_SEG_B,
        method="wilcoxon",
        use_raw=False,
    )
    df = sc.get.rank_genes_groups_df(s, group=POWER_SEG_A)
    return float((df["pvals_adj"].to_numpy() < fdr).sum())


def compute_power_curve(ad) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Downsample-balanced Wilcoxon DE counts vs n for selected HGCA v1 types."""
    curve_rows: list[dict] = []
    mark_rows: list[dict] = []

    for ct_full, ct_short in POWER_CELLTYPES:
        sub = ad[ad.obs["ct"] == ct_full].copy()
        if sub.n_obs == 0:
            LOG.warning("missing cell type %s — skip", ct_full)
            continue
        pub = sub[sub.obs["provenance"] == "Published studies"]
        n_pub = int(
            min(
                (pub.obs["segment"] == POWER_SEG_A).sum(),
                (pub.obs["segment"] == POWER_SEG_B).sum(),
            )
        )
        n_all = int(
            min(
                (sub.obs["segment"] == POWER_SEG_A).sum(),
                (sub.obs["segment"] == POWER_SEG_B).sum(),
            )
        )
        LOG.info("%s: balanced n published=%d, all=%d", ct_short, n_pub, n_all)

        n_grid = sorted(
            {n for n in POWER_N_GRID if 50 <= n <= n_all} | {n_pub, n_all}
        )
        for n in n_grid:
            vals = [
                _n_de_balanced(sub, int(n), seed=s) for s in range(POWER_N_SEEDS)
            ]
            vals = [v for v in vals if np.isfinite(v)]
            if not vals:
                continue
            curve_rows.append(
                {
                    "celltype": ct_full,
                    "celltype_short": ct_short,
                    "n_per_segment": int(n),
                    "n_de_mean": float(np.mean(vals)),
                    "n_de_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "n_reps": len(vals),
                }
            )
            LOG.info("  n=%d → DE=%.0f ± %.0f", n, np.mean(vals), np.std(vals))

        # Markers at published-only and published+contributed operating points
        for label, n_mark in [
            ("Published only", n_pub),
            ("Published + contributed", n_all),
        ]:
            if n_mark < 50:
                continue
            vals = [
                _n_de_balanced(sub, int(n_mark), seed=s) for s in range(POWER_N_SEEDS)
            ]
            vals = [v for v in vals if np.isfinite(v)]
            mark_rows.append(
                {
                    "celltype": ct_full,
                    "celltype_short": ct_short,
                    "point": label,
                    "n_per_segment": int(n_mark),
                    "n_de_mean": float(np.mean(vals)) if vals else np.nan,
                }
            )

    curve = pd.DataFrame(curve_rows)
    marks = pd.DataFrame(mark_rows)
    DATA.mkdir(parents=True, exist_ok=True)
    curve.to_csv(DATA / "de_power_curve_ileum_vs_colon.csv", index=False)
    marks.to_csv(DATA / "de_power_curve_markers.csv", index=False)
    return curve, marks


def load_cached_power() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(DATA / "de_power_curve_ileum_vs_colon.csv"),
        pd.read_csv(DATA / "de_power_curve_markers.csv"),
    )


def _plot_power_curve_core(
    curve: pd.DataFrame,
    marks: pd.DataFrame,
    *,
    line_colors: dict[str, str],
    xlabel: str,
    ylabel: str,
    title: str,
    stem: str,
) -> None:
    from matplotlib.lines import Line2D

    fig, ax = figure_nature(width_mm=90, height_mm=65)
    for ct_short, sub in curve.groupby("celltype_short", sort=False):
        sub = sub.sort_values("n_per_segment")
        color = line_colors.get(ct_short, PAL["mid_grey"])
        ax.plot(
            sub["n_per_segment"],
            sub["n_de_mean"],
            color=color,
            linewidth=0.9,
            label=ct_short,
            zorder=2,
        )
        if "n_de_sd" in sub and sub["n_de_sd"].notna().any():
            ax.fill_between(
                sub["n_per_segment"],
                sub["n_de_mean"] - sub["n_de_sd"],
                sub["n_de_mean"] + sub["n_de_sd"],
                color=color,
                alpha=0.12,
                linewidth=0,
                zorder=1,
            )

    for _, r in marks.iterrows():
        if not np.isfinite(r["n_de_mean"]):
            continue
        color = line_colors.get(r["celltype_short"], PAL["mid_grey"])
        face = "white" if r["point"] == "Published only" else color
        ax.scatter(
            [r["n_per_segment"]],
            [r["n_de_mean"]],
            s=18,
            facecolors=face,
            edgecolors=color,
            linewidths=0.7,
            zorder=3,
        )

    handles, labels = ax.get_legend_handles_labels()
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor=PAL["black"],
                markersize=4.5,
                markeredgewidth=0.7,
                label="Published only",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=PAL["black"],
                markeredgecolor=PAL["black"],
                markersize=4.5,
                label="Published + contributed",
            ),
        ]
    )
    ax.legend(
        handles,
        labels + ["Published only", "Published + contributed"],
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
        title="",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _: f"{int(v):,}")
    )
    ax.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _: f"{int(v):,}")
    )
    fig.tight_layout()
    save_figure(fig, stem)
    plt.close(fig)


def plot_power_curve(curve: pd.DataFrame, marks: pd.DataFrame) -> None:
    """Cell-level Wilcoxon power curve (panel f)."""
    _plot_power_curve_core(
        curve,
        marks,
        line_colors={
            "cDC1": PAL["vermillion"],
            "cDC2": PAL["hca_blue"],
            "Hom. mac": PAL["bluish_green"],
        },
        xlabel="Cells per segment (balanced)",
        ylabel=f"DE genes (FDR < {POWER_FDR:g})",
        title="Ileum vs colon power (Wilcoxon)",
        stem="s6_f_de_power_ileum_vs_colon",
    )


# Panel-g variants (must match compute_deseq2_power.VARIANTS keys)
DESEQ2_VARIANTS = {
    "epithelial_omega": {
        "stem": "s6_g_de_power_deseq2_epithelial_omega",
        "title": "Ileum vs colon (DESeq2, epithelial)",
        "colors": {
            "Goblet": PAL["vermillion"],
            "TA": PAL["hca_blue"],
        },
    },
    "lymphoid_t": {
        "stem": "s6_g_de_power_deseq2_lymphoid_t",
        "title": "Ileum vs colon (DESeq2, T cells)",
        "colors": {
            "CD8 IEL": PAL["vermillion"],
            "CD8 TEM": PAL["hca_blue"],
            "CD4 Tfh": PAL["bluish_green"],
        },
    },
    "lymphoid_b": {
        "stem": "s6_g_de_power_deseq2_lymphoid_b",
        "title": "Ileum vs colon (DESeq2, B / plasma)",
        "colors": {
            "Memory B": PAL["vermillion"],
            "Plasma IgA": PAL["hca_blue"],
            "GC B LZ": PAL["bluish_green"],
        },
    },
    "myeloid_dc_mast": {
        "stem": "s6_g_de_power_deseq2_myeloid_dc_mast",
        "title": "Ileum vs colon (DESeq2, myeloid)",
        "colors": {
            "cDC2": PAL["hca_blue"],
            "Mast": PAL["vermillion"],
            "cMono": PAL["bluish_green"],
        },
    },
}


def plot_deseq2_power_curve(variant: str | None = None) -> None:
    """Pseudobulk DESeq2 discovery-count curves (legacy panel-g variants)."""
    names = [variant] if variant else list(DESEQ2_VARIANTS)
    for name in names:
        spec = DESEQ2_VARIANTS[name]
        curve_path = DATA / f"de_power_deseq2_{name}.csv"
        mark_path = DATA / f"de_power_deseq2_{name}_markers.csv"
        if not curve_path.exists():
            LOG.warning("missing %s — skip", curve_path.name)
            continue
        curve = pd.read_csv(curve_path)
        marks = pd.read_csv(mark_path) if mark_path.exists() else pd.DataFrame()
        # Map columns to the shared plotter (n_de_mean)
        _plot_power_curve_core(
            curve,
            marks,
            line_colors=spec["colors"],
            xlabel="Samples per segment (balanced)",
            ylabel="DE genes (FDR < 0.05)",
            title=spec["title"],
            stem=spec["stem"],
        )


def plot_deseq2_recovery_power() -> None:
    """Recovery-power slope chart (preferred panel g) + optional balanced curve."""
    from matplotlib.lines import Line2D

    curve_path = DATA / "de_power_deseq2_recovery.csv"
    mark_path = DATA / "de_power_deseq2_recovery_markers.csv"
    if not mark_path.exists():
        LOG.warning("missing %s — run compute_deseq2_recovery_power.py", mark_path)
        return

    marks = pd.read_csv(mark_path)
    curve = pd.read_csv(curve_path) if curve_path.exists() else pd.DataFrame()
    colors = {
        "Goblet": PAL["vermillion"],
        "TA": PAL["hca_blue"],
        "CD8 IEL": PAL["bluish_green"],
        "Memory B": "#E69F00",  # Wong orange
        "Plasma IgA": PAL["sky_blue"],
    }

    # Preferred view: slope / dumbbell at ACTUAL sample-set operating points
    if not marks.empty:
        wide = marks.pivot(index="celltype_short", columns="point", values="power_mean")
        need = ["Published only", "Published + contributed"]
        if all(c in wide.columns for c in need):
            order = [c for c in colors if c in wide.index]
            order += [c for c in wide.index if c not in order]
            wide = wide.reindex(order)

            fig, ax = figure_nature(width_mm=90, height_mm=55)
            y = np.arange(len(wide))
            for i, ct in enumerate(wide.index):
                x0 = wide.loc[ct, "Published only"]
                x1 = wide.loc[ct, "Published + contributed"]
                color = colors.get(ct, PAL["mid_grey"])
                ax.plot([x0, x1], [i, i], color=color, linewidth=0.8, zorder=1)
                ax.scatter(
                    [x0], [i], s=22, facecolors="white", edgecolors=color,
                    linewidths=0.8, zorder=2,
                )
                ax.scatter(
                    [x1], [i], s=22, facecolors=color, edgecolors=color,
                    linewidths=0.8, zorder=2,
                )
                # Annotate gain
                ax.text(
                    min(x1 + 0.02, 1.02),
                    i,
                    f"+{(x1 - x0):.0%}",
                    va="center",
                    ha="left",
                    fontsize=5.5,
                    color=PAL["mid_grey"],
                )
            ax.set_yticks(y)
            ax.set_yticklabels(wide.index)
            ax.set_xlabel("Power (fraction of full-atlas DE genes recovered)")
            ax.set_xlim(-0.02, 1.18)
            ax.xaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0%}")
            )
            ax.set_title("Ileum vs colon recovery power (DESeq2)")
            ax.legend(
                handles=[
                    Line2D(
                        [0], [0], marker="o", color="none", markerfacecolor="white",
                        markeredgecolor=PAL["black"], markersize=4.5,
                        markeredgewidth=0.7, label="Published only",
                    ),
                    Line2D(
                        [0], [0], marker="o", color="none", markerfacecolor=PAL["black"],
                        markeredgecolor=PAL["black"], markersize=4.5,
                        label="Published + contributed",
                    ),
                ],
                loc="lower right",
                frameon=False,
            )
            fig.tight_layout()
            save_figure(fig, "s6_g_de_power_deseq2_recovery_slope")
            plt.close(fig)

            # Also write as the main panel-g stem for Illustrator drop-in
            fig, ax = figure_nature(width_mm=90, height_mm=55)
            for i, ct in enumerate(wide.index):
                x0 = wide.loc[ct, "Published only"]
                x1 = wide.loc[ct, "Published + contributed"]
                color = colors.get(ct, PAL["mid_grey"])
                ax.plot([x0, x1], [i, i], color=color, linewidth=0.8, zorder=1)
                ax.scatter(
                    [x0], [i], s=22, facecolors="white", edgecolors=color,
                    linewidths=0.8, zorder=2,
                )
                ax.scatter(
                    [x1], [i], s=22, facecolors=color, edgecolors=color,
                    linewidths=0.8, zorder=2,
                )
            ax.set_yticks(y)
            ax.set_yticklabels(wide.index)
            ax.set_xlabel("Power (fraction of full-atlas DE genes recovered)")
            ax.set_xlim(0, 1.05)
            ax.xaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0%}")
            )
            ax.set_title("Ileum vs colon recovery power (DESeq2)")
            ax.legend(
                handles=[
                    Line2D(
                        [0], [0], marker="o", color="none", markerfacecolor="white",
                        markeredgecolor=PAL["black"], markersize=4.5,
                        markeredgewidth=0.7, label="Published only",
                    ),
                    Line2D(
                        [0], [0], marker="o", color="none", markerfacecolor=PAL["black"],
                        markeredgecolor=PAL["black"], markersize=4.5,
                        label="Published + contributed",
                    ),
                ],
                loc="lower right",
                frameon=False,
            )
            fig.tight_layout()
            save_figure(fig, "s6_g_de_power_deseq2_recovery")
            plt.close(fig)

    # Optional balanced downsampling curve (no operating-point markers — those
    # use actual sample sets and would sit off this curve).
    if not curve.empty and "power_mean" in curve.columns and len(curve):
        fig, ax = figure_nature(width_mm=90, height_mm=65)
        for ct_short, sub in curve.groupby("celltype_short", sort=False):
            sub = sub.sort_values("n_per_segment")
            color = colors.get(ct_short, PAL["mid_grey"])
            ax.plot(
                sub["n_per_segment"],
                sub["power_mean"],
                color=color,
                linewidth=0.9,
                label=ct_short,
                zorder=2,
            )
            if "power_sd" in sub and sub["power_sd"].notna().any():
                ax.fill_between(
                    sub["n_per_segment"],
                    (sub["power_mean"] - sub["power_sd"]).clip(lower=0),
                    (sub["power_mean"] + sub["power_sd"]).clip(upper=1),
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                    zorder=1,
                )
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
        ax.set_xlabel("Samples per segment (balanced)")
        ax.set_ylabel("Power (DE genes recovered)")
        ax.set_title("Ileum vs colon recovery vs sample size")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0%}")
        )
        fig.tight_layout()
        save_figure(fig, "s6_g_de_power_deseq2_recovery_curve")
        plt.close(fig)


def plot_deseq2_analytical_power() -> None:
    """Wald analytical power: ~dataset_id+seg vs ~seg; curve + slope."""
    from matplotlib.lines import Line2D

    curve_path = DATA / "de_power_deseq2_analytical.csv"
    mark_path = DATA / "de_power_deseq2_analytical_markers.csv"
    if not mark_path.exists():
        LOG.warning("missing %s — run compute_deseq2_analytical_power.py", mark_path)
        return

    marks = pd.read_csv(mark_path)
    curve = pd.read_csv(curve_path) if curve_path.exists() else pd.DataFrame()
    colors = {
        "cDC2": PAL["hca_blue"],
        "Hom. mac": PAL["bluish_green"],
        "CD8 IEL": PAL["vermillion"],
        "Memory B": "#E69F00",
        "Goblet": PAL["sky_blue"],
    }
    design_titles = {
        "dataset_seg": r"$\sim$ dataset_id + seg",
        "seg_only": r"$\sim$ seg (samples as replicates)",
    }

    for design, dsub in marks.groupby("design", sort=False):
        title = design_titles.get(design, design)
        stem = f"s6_g_de_power_deseq2_analytical_{design}"

        wide = dsub.pivot(index="celltype_short", columns="point", values="power_mean")
        need = ["Published only", "Published + contributed"]
        if all(c in wide.columns for c in need):
            order = [c for c in colors if c in wide.index]
            order += [c for c in wide.index if c not in order]
            wide = wide.reindex(order)
            fig, ax = figure_nature(width_mm=90, height_mm=55)
            y = np.arange(len(wide))
            for i, ct in enumerate(wide.index):
                x0 = wide.loc[ct, "Published only"]
                x1 = wide.loc[ct, "Published + contributed"]
                color = colors.get(ct, PAL["mid_grey"])
                ax.plot([x0, x1], [i, i], color=color, linewidth=0.8, zorder=1)
                ax.scatter(
                    [x0], [i], s=22, facecolors="white", edgecolors=color,
                    linewidths=0.8, zorder=2,
                )
                ax.scatter(
                    [x1], [i], s=22, facecolors=color, edgecolors=color,
                    linewidths=0.8, zorder=2,
                )
                ax.text(
                    min(x1 + 0.02, 1.02),
                    i,
                    f"+{(x1 - x0):.0%}",
                    va="center",
                    ha="left",
                    fontsize=5.5,
                    color=PAL["mid_grey"],
                )
            ax.set_yticks(y)
            ax.set_yticklabels(wide.index)
            ax.set_xlabel("Mean Wald power (target DE genes)")
            ax.set_xlim(-0.02, 1.18)
            ax.xaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0%}")
            )
            ax.set_title(f"Ileum vs colon power ({title})")
            ax.legend(
                handles=[
                    Line2D(
                        [0], [0], marker="o", color="none", markerfacecolor="white",
                        markeredgecolor=PAL["black"], markersize=4.5,
                        markeredgewidth=0.7, label="Published only",
                    ),
                    Line2D(
                        [0], [0], marker="o", color="none", markerfacecolor=PAL["black"],
                        markeredgecolor=PAL["black"], markersize=4.5,
                        label="Published + contributed",
                    ),
                ],
                loc="lower right",
                frameon=False,
            )
            fig.tight_layout()
            save_figure(fig, f"{stem}_slope")
            plt.close(fig)

        # Smooth analytical power curve vs balanced n (± 95% CI of mean)
        if not curve.empty:
            csub = curve[curve["design"] == design]
            if len(csub):
                fig, ax = figure_nature(width_mm=90, height_mm=65)
                for ct_short, sub in csub.groupby("celltype_short", sort=False):
                    sub = sub.sort_values("n_per_segment")
                    color = colors.get(ct_short, PAL["mid_grey"])
                    # Visible ribbon = mean ± s.d. across genes (gene-to-gene
                    # heterogeneity). SE-of-mean 95% CI is in the CSV but too
                    # narrow to see with large gene sets.
                    if {"power_sd_lo", "power_sd_hi"}.issubset(sub.columns):
                        ax.fill_between(
                            sub["n_per_segment"],
                            sub["power_sd_lo"],
                            sub["power_sd_hi"],
                            color=color,
                            alpha=0.15,
                            linewidth=0,
                            zorder=1,
                        )
                    elif "power_sd" in sub.columns:
                        ax.fill_between(
                            sub["n_per_segment"],
                            (sub["power_mean"] - sub["power_sd"]).clip(lower=0),
                            (sub["power_mean"] + sub["power_sd"]).clip(upper=1),
                            color=color,
                            alpha=0.15,
                            linewidth=0,
                            zorder=1,
                        )
                    ax.plot(
                        sub["n_per_segment"],
                        sub["power_mean"],
                        color=color,
                        linewidth=0.9,
                        label=ct_short,
                        zorder=2,
                    )
                # Markers on curve at balanced n
                for _, r in dsub.iterrows():
                    color = colors.get(r["celltype_short"], PAL["mid_grey"])
                    face = "white" if r["point"] == "Published only" else color
                    match = csub[
                        (csub["celltype_short"] == r["celltype_short"])
                        & (csub["n_per_segment"] == r["n_per_segment"])
                    ]
                    yv = (
                        float(match["power_mean"].iloc[0])
                        if len(match)
                        else float(r["power_mean"])
                    )
                    ax.scatter(
                        [r["n_per_segment"]],
                        [yv],
                        s=18,
                        facecolors=face,
                        edgecolors=color,
                        linewidths=0.7,
                        zorder=3,
                    )
                handles, labels = ax.get_legend_handles_labels()
                handles.extend(
                    [
                        Line2D(
                            [0], [0], marker="o", color="none", markerfacecolor="white",
                            markeredgecolor=PAL["black"], markersize=4.5,
                            markeredgewidth=0.7, label="Published only",
                        ),
                        Line2D(
                            [0], [0], marker="o", color="none", markerfacecolor=PAL["black"],
                            markeredgecolor=PAL["black"], markersize=4.5,
                            label="Published + contributed",
                        ),
                    ]
                )
                ax.legend(
                    handles,
                    labels + ["Published only", "Published + contributed"],
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1),
                    frameon=False,
                )
                ax.set_xlabel("Samples per segment (balanced)")
                ax.set_ylabel("Mean Wald power (± s.d.)")
                ax.set_title(f"Ileum vs colon power ({title})")
                ax.set_ylim(0, 1.05)
                ax.yaxis.set_major_formatter(
                    mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0%}")
                )
                fig.tight_layout()
                save_figure(fig, stem)
                plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-power", action="store_true", help="Skip Wilcoxon power panel")
    p.add_argument(
        "--reuse-power",
        action="store_true",
        help="Reuse cached Wilcoxon power-curve CSVs in data/",
    )
    p.add_argument("--skip-bars", action="store_true", help="Skip provenance bar panels")
    p.add_argument(
        "--plot-deseq2",
        action="store_true",
        help="Render legacy DESeq2 discovery-count variants from cached CSVs",
    )
    p.add_argument(
        "--deseq2-variant",
        action="append",
        choices=sorted(DESEQ2_VARIANTS),
        help="Restrict --plot-deseq2 to one variant (repeatable)",
    )
    p.add_argument(
        "--plot-deseq2-recovery",
        action="store_true",
        help="Render recovery-power curve + slope chart from cached CSVs",
    )
    p.add_argument(
        "--plot-deseq2-analytical",
        action="store_true",
        help="Render analytical Wald power curves/slopes from cached CSVs",
    )
    # Back-compat aliases
    p.add_argument("--skip-de", dest="skip_power", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--reuse-de", dest="reuse_power", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(ROOT / "logs" / "render_s6.log"),
        ],
    )
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    # Quiet fontTools SVG subset noise
    logging.getLogger("fontTools").setLevel(logging.WARNING)

    if not args.skip_bars:
        if not PATHS:
            raise SystemExit("Set HGCA_CAP_DIR to the directory with the four lineage h5ads.")
        for name, path in PATHS.items():
            if not path.exists():
                LOG.error("missing input: %s", path)
                return 1
        obs_by_lineage = {lin: load_obs(PATHS[lin]) for lin in LINEAGES}
        compute_provenance_tables(obs_by_lineage)
        plot_provenance_by_lineage()
        plot_provenance_by_lineage_segment()
        plot_gap_coverage(
            "lymphoid", "s6_c_lymphoid_coverage_by_segment", "Lymphoid coverage"
        )
        plot_gap_coverage(
            "myeloid", "s6_d_myeloid_coverage_by_segment", "Myeloid coverage"
        )
        plot_myeloid_published_vs_all(obs_by_lineage["myeloid"])
    else:
        LOG.info("skipped bar panels (--skip-bars)")

    if not args.skip_power:
        cache = DATA / "de_power_curve_ileum_vs_colon.csv"
        if args.reuse_power and cache.exists():
            LOG.info("reusing cached Wilcoxon power curve")
            curve, marks = load_cached_power()
        else:
            ad = load_myeloid_v1_for_power()
            curve, marks = compute_power_curve(ad)
            del ad
        plot_power_curve(curve, marks)
    else:
        LOG.info("skipped Wilcoxon power panel (--skip-power)")

    if args.plot_deseq2:
        if args.deseq2_variant:
            for v in args.deseq2_variant:
                plot_deseq2_power_curve(v)
        else:
            plot_deseq2_power_curve()

    if args.plot_deseq2_recovery or (
        DATA / "de_power_deseq2_recovery.csv"
    ).exists():
        plot_deseq2_recovery_power()

    if args.plot_deseq2_analytical or (
        DATA / "de_power_deseq2_analytical_markers.csv"
    ).exists():
        plot_deseq2_analytical_power()

    LOG.info("done — outputs in %s", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
