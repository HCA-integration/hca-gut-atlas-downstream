#!/usr/bin/env python3
"""Build Figure 2 atlas-building evidence panels from HGCA cell metadata.

The script produces publication-ready vector/raster triples plus source tables:

1. Author-label crosswalk vs. HGCA v0 per-class precision/recall/F1.
2. An ARBOL-ordered cell-type sidecar summarizing atlas breadth and reannotation.
3. A dataset-by-cell-type supplemental heatmap with dataset attribute sidecars.
4. PanGI/HGCA label-count tables and a PanGI-to-HGCA crosswalk template for the
   separate ARBOL overlay renderer.

Terminology is intentionally conservative: exact disagreement between
``closest_GCA_celltype`` and ``hgca_celltype_v0`` is called "reannotation" or
"discordance", not "misannotation". The former label is a manually harmonized
author-label crosswalk and often sits at a coarser ontology depth than v0.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
FIGURE_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
DEFAULT_METADATA = REPO_ROOT / "data" / "demo" / "hgca_all_lineages_v1_demo.h5ad"
DEFAULT_TAXONOMY = REPO_ROOT / "data" / "demo" / "GCA_taxonomy_2026_CAP.csv"
DEFAULT_BENCHMARK_RESULTS = Path(os.environ.get("HGCA_BENCHMARK_RESULTS", ""))
DEFAULT_CAP_LABELS = Path(
    os.environ.get("HGCA_CAP_LABELS")
    or str(REPO_ROOT / "data" / "cap" / "cap_labels_901.csv")
)
DEFAULT_CAP_FEEDBACK = Path(
    os.environ.get("HGCA_CAP_FEEDBACK")
    or str(REPO_ROOT / "data" / "cap" / "cap_feedback_901.csv")
)
DEFAULT_CAP_BRIDGE = Path(
    os.environ.get("HGCA_CAP_BRIDGE")
    or str(REPO_ROOT / "data" / "cap" / "taxonomy_marker_source_report.csv")
)
LODO_LINEAGE_DIRS = {
    "myeloid": "myeloid_hgca_v0_v1_pangi",
    "lymphoid": "lymphoid_hgca_v0_v1_pangi",
    "epithelial": "epithelial_hgca_v0_v1_pangi",
    "stromal": "stroma_hgca_v0_v1_pangi",
}

PALETTE = {
    "hgca": "#0072B2",
    "author": "#D55E00",
    "agreement": "#009E73",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "absent": "#E0E0E0",
    "midgrey": "#999999",
    "black": "#000000",
}

METADATA_COLUMNS = [
    "author_cell_type",
    "closest_GCA_celltype",
    "hgca_celltype_v0",
    "hgca_celltype_v1",
    "hgca_celltype_level1",
    "dataset_id",
    "sample_id",
    "donor_id",
    "tissue_level_1",
    "tissue_level_2",
    "radial_tissue_term",
    "sample_collection_method",
    "disease",
    "sampled_site_condition",
]
TAXONOMY_LEVEL_COLUMNS = [f"hgca_celltype_level{i}" for i in range(1, 6)]


def configure_plotting() -> None:
    """Apply the project Nature figure specification at final export size."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6,
            "axes.titlesize": 7,
            "axes.labelsize": 6,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "legend.fontsize": 5,
            "legend.title_fontsize": 6,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: mpl.figure.Figure, base: Path, *, dpi: int = 300) -> None:
    """Save Illustrator-editable PDF/SVG and a 300-dpi PNG preview."""
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".svg"):
        fig.savefig(base.with_suffix(suffix), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")


def clean_label(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(r"[\r\n]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def load_metadata(path: Path) -> pd.DataFrame:
    print(f"Loading atlas metadata: {path}")
    if path.suffix.lower() == ".h5ad":
        import anndata as ad

        atlas = ad.read_h5ad(path, backed="r")
        missing = sorted(set(METADATA_COLUMNS) - set(atlas.obs.columns))
        if missing:
            atlas.file.close()
            raise ValueError(f"Atlas metadata is missing columns: {missing}")
        df = atlas.obs.loc[:, METADATA_COLUMNS].copy()
        atlas.file.close()
    else:
        df = pd.read_csv(path, usecols=METADATA_COLUMNS, low_memory=False)
    for col in METADATA_COLUMNS:
        df[col] = clean_label(df[col])
    print(
        f"Loaded {len(df):,} cells, {df['dataset_id'].nunique():,} datasets, "
        f"{df['sample_id'].nunique():,} samples"
    )
    return df


def taxonomy_order(path: Path) -> tuple[list[str], pd.DataFrame]:
    tax = pd.read_csv(path, low_memory=False)
    required = ["hgca_celltype_v1", "hgca_celltype_v0", "hgca_celltype_level1"]
    missing = [c for c in required if c not in tax]
    if missing:
        raise ValueError(f"Taxonomy is missing columns: {missing}")
    for col in required:
        tax[col] = clean_label(tax[col])
    order = tax["hgca_celltype_v1"].dropna().drop_duplicates().tolist()
    return order, tax


def author_v0_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Treat the author-label crosswalk as truth and v0 as the compared label set."""
    valid = df.dropna(subset=["closest_GCA_celltype", "hgca_celltype_v0"]).copy()
    valid["exact_match"] = valid["closest_GCA_celltype"].eq(valid["hgca_celltype_v0"])

    labels = valid["closest_GCA_celltype"].drop_duplicates().tolist()
    rows: list[dict[str, float | int | str]] = []
    for label in labels:
        truth = valid["closest_GCA_celltype"].eq(label)
        pred = valid["hgca_celltype_v0"].eq(label)
        tp = int((truth & pred).sum())
        fp = int((~truth & pred).sum())
        fn = int((truth & ~pred).sum())
        support = int(truth.sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "closest_GCA_celltype": label,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "reannotated_cells": fn,
                "reannotated_fraction": fn / support if support else np.nan,
            }
        )

    per_class = pd.DataFrame(rows).sort_values(
        ["f1", "support"], ascending=[True, False]
    )
    overall = pd.DataFrame(
        [
            {
                "n_cells": len(valid),
                "n_exact_match": int(valid["exact_match"].sum()),
                "n_reannotated": int((~valid["exact_match"]).sum()),
                "exact_match_fraction": float(valid["exact_match"].mean()),
                "reannotated_fraction": float((~valid["exact_match"]).mean()),
                "n_author_crosswalk_types": valid["closest_GCA_celltype"].nunique(),
                "n_hgca_v0_types": valid["hgca_celltype_v0"].nunique(),
            }
        ]
    )
    return per_class, overall


def plot_author_v0_metrics(per_class: pd.DataFrame, overall: pd.DataFrame, out: Path) -> None:
    plot_df = per_class.sort_values("f1", ascending=True).reset_index(drop=True)
    height_mm = min(170, max(90, 4.2 * len(plot_df)))
    fig, ax = plt.subplots(figsize=(180 / 25.4, height_mm / 25.4))
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["f1"], color=PALETTE["hgca"], height=0.68, linewidth=0)
    ax.scatter(
        plot_df["recall"],
        y,
        color=PALETTE["author"],
        s=8,
        zorder=3,
        label="Recall (author-crosswalk cells retained exactly)",
    )
    ax.set_yticks(y, plot_df["closest_GCA_celltype"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Exact-label score")
    ax.set_ylabel("Closest GCA cell type for author label")
    pct = 100 * float(overall.loc[0, "reannotated_fraction"])
    n = int(overall.loc[0, "n_cells"])
    ax.set_title(
        f"Author-label crosswalk versus HGCA v0 "
        f"({pct:.1f}% changed at exact label level; n = {n:,} cells)",
        loc="left",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, out / "fig2_author_crosswalk_v0_f1")
    plt.close(fig)


def _disease_groups(series: pd.Series) -> pd.Series:
    s = series.fillna("Unknown").str.lower()
    return pd.Series(
        np.select(
            [
                s.str.contains("normal|healthy|control"),
                s.str.contains("crohn|ulcerative|colitis|ibd|disease"),
            ],
            ["Health/control", "Disease/case"],
            default="Other/unknown",
        ),
        index=series.index,
        dtype="string",
    )


def _longest_common_prefix(paths: list[tuple[str, ...]]) -> tuple[str, ...]:
    if not paths:
        return ()
    prefix = list(paths[0])
    for path in paths[1:]:
        shared = 0
        for left, right in zip(prefix, path):
            if left != right:
                break
            shared += 1
        prefix = prefix[:shared]
        if not prefix:
            break
    return tuple(prefix)


def _taxonomy_paths(taxonomy: pd.DataFrame) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Return author-term and HGCA-v1 paths through taxonomy levels 1–5."""
    tax = taxonomy[
        ["hgca_celltype_v0", "hgca_celltype_v1", *TAXONOMY_LEVEL_COLUMNS]
    ].copy()
    for col in tax:
        tax[col] = clean_label(tax[col])

    row_paths = [
        tuple(str(value) for value in row if pd.notna(value) and str(value) != "")
        for row in tax[TAXONOMY_LEVEL_COLUMNS].itertuples(index=False, name=None)
    ]
    tax["_path"] = row_paths

    v1_paths = (
        tax.dropna(subset=["hgca_celltype_v1"])
        .drop_duplicates("hgca_celltype_v1")
        .set_index("hgca_celltype_v1")["_path"]
        .to_dict()
    )
    author_paths: dict[str, tuple[str, ...]] = {}
    for label, group in tax.dropna(subset=["hgca_celltype_v0"]).groupby(
        "hgca_celltype_v0", observed=True
    ):
        author_paths[str(label)] = _longest_common_prefix(group["_path"].tolist())
    for label, path in v1_paths.items():
        author_paths.setdefault(str(label), path)
    return author_paths, {str(label): path for label, path in v1_paths.items()}


def _add_hierarchical_author_metrics(
    cells: pd.DataFrame, taxonomy: pd.DataFrame
) -> pd.DataFrame:
    """Classify author-to-atlas changes using ancestry, not exact strings."""
    author_paths, atlas_paths = _taxonomy_paths(taxonomy)
    pairs = cells[
        ["closest_GCA_celltype", "hgca_celltype_v1"]
    ].drop_duplicates().copy()
    records: list[dict[str, object]] = []
    for author_label, atlas_label in pairs.itertuples(index=False, name=None):
        author_path = author_paths.get(str(author_label), ())
        atlas_path = atlas_paths.get(str(atlas_label), ())
        mapped = bool(author_path and atlas_path)
        common_depth = (
            len(_longest_common_prefix([author_path, atlas_path])) if mapped else 0
        )
        author_is_prefix = (
            mapped
            and len(author_path) < len(atlas_path)
            and common_depth == len(author_path)
        )
        atlas_is_prefix = (
            mapped
            and len(atlas_path) < len(author_path)
            and common_depth == len(atlas_path)
        )
        same_node = mapped and author_path == atlas_path
        record: dict[str, object] = {
            "closest_GCA_celltype": author_label,
            "hgca_celltype_v1": atlas_label,
            "author_taxonomy_mapped": float(mapped),
            "atlas_increased_resolution": float(author_is_prefix),
            "atlas_same_resolution": float(same_node),
            "atlas_reduced_resolution": float(atlas_is_prefix),
            "atlas_changed_branch": float(
                mapped and not author_is_prefix and not atlas_is_prefix and not same_node
            ),
        }
        for level in range(1, 6):
            evaluable = mapped and len(author_path) >= level and len(atlas_path) >= level
            record[f"author_level{level}_match"] = (
                float(author_path[level - 1] == atlas_path[level - 1])
                if evaluable
                else np.nan
            )
        records.append(record)
    metrics = pd.DataFrame.from_records(records)
    return cells.merge(
        metrics,
        on=["closest_GCA_celltype", "hgca_celltype_v1"],
        how="left",
        validate="many_to_one",
    )


def _normalize_celltype_label(value: object) -> str:
    return " ".join(str(value).replace("\n", " ").split())


def load_lodo_v1_per_class_f1(benchmark_results: Path) -> pd.DataFrame:
    """Load newest HGCA v1 LODO support-weighted per-class F1 summaries.

    Uses ``results/comparisons/e3_full_lodo_*/{lineage}_hgca_v0_v1_pangi/
    hgca_v1_per_class_f1_summary.csv``. Values are already on a 0–1 scale.
    """
    comparisons = benchmark_results / "comparisons"
    if not comparisons.is_dir():
        raise FileNotFoundError(f"Missing benchmark comparisons dir: {comparisons}")
    candidate_runs = sorted(
        [
            path
            for path in comparisons.iterdir()
            if path.is_dir()
            and path.name.startswith("e3_full_lodo_")
            and "partial" not in path.name.lower()
        ],
        key=lambda path: path.stat().st_mtime,
    )
    run_dir = None
    for candidate in reversed(candidate_runs):
        if all(
            (candidate / folder / "hgca_v1_per_class_f1_summary.csv").is_file()
            for folder in LODO_LINEAGE_DIRS.values()
        ):
            run_dir = candidate
            break
    if run_dir is None:
        raise FileNotFoundError(
            "No complete e3_full_lodo_* comparison with "
            "hgca_v1_per_class_f1_summary.csv for all lineages under "
            f"{comparisons}"
        )
    frames: list[pd.DataFrame] = []
    for lineage, folder in LODO_LINEAGE_DIRS.items():
        path = run_dir / folder / "hgca_v1_per_class_f1_summary.csv"
        one = pd.read_csv(path)
        required = {"cell_type", "mean_f1_weighted", "total_support"}
        missing = required - set(one.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        one = one.rename(
            columns={
                "cell_type": "hgca_celltype_v1",
                "mean_f1_weighted": "lodo_f1",
                "total_support": "lodo_f1_support",
                "std_f1_across_folds": "lodo_f1_std",
                "n_folds_present": "lodo_f1_n_folds",
            }
        )
        one["hgca_celltype_v1"] = one["hgca_celltype_v1"].map(
            _normalize_celltype_label
        )
        one["lineage"] = lineage
        one["lodo_run"] = run_dir.name
        frames.append(
            one[
                [
                    "hgca_celltype_v1",
                    "lodo_f1",
                    "lodo_f1_support",
                    "lodo_f1_std",
                    "lodo_f1_n_folds",
                    "lineage",
                    "lodo_run",
                ]
            ]
        )
    out = pd.concat(frames, ignore_index=True)
    # Prefer the leaf-level row when an internal node shares a cleaned label.
    out = out.sort_values(
        ["hgca_celltype_v1", "lodo_f1_support"], ascending=[True, False]
    ).drop_duplicates("hgca_celltype_v1", keep="first")
    return out.reset_index(drop=True)


def build_celltype_tables(
    df: pd.DataFrame, ordered_types: list[str], taxonomy: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = df.dropna(subset=["hgca_celltype_v1"]).copy()
    d["author_v0_exact"] = d["closest_GCA_celltype"].eq(d["hgca_celltype_v0"])
    d = _add_hierarchical_author_metrics(d, taxonomy)
    d["disease_group"] = _disease_groups(d["disease"])
    total = len(d)

    summary = (
        d.groupby("hgca_celltype_v1", observed=True)
        .agg(
            n_cells=("hgca_celltype_v1", "size"),
            n_datasets=("dataset_id", "nunique"),
            n_samples=("sample_id", "nunique"),
            n_donors=("donor_id", "nunique"),
            n_tissues=("tissue_level_1", "nunique"),
            n_author_labels=("author_cell_type", "nunique"),
            author_v0_exact_fraction=("author_v0_exact", "mean"),
            author_level1_match_fraction=("author_level1_match", "mean"),
            author_level2_match_fraction=("author_level2_match", "mean"),
            author_level3_match_fraction=("author_level3_match", "mean"),
            author_level4_match_fraction=("author_level4_match", "mean"),
            author_level5_match_fraction=("author_level5_match", "mean"),
            atlas_increased_resolution_fraction=("atlas_increased_resolution", "mean"),
            atlas_same_resolution_fraction=("atlas_same_resolution", "mean"),
            atlas_reduced_resolution_fraction=("atlas_reduced_resolution", "mean"),
            atlas_changed_branch_fraction=("atlas_changed_branch", "mean"),
            author_taxonomy_mapped_fraction=("author_taxonomy_mapped", "mean"),
        )
        .reset_index()
    )
    summary["cell_fraction"] = summary["n_cells"] / total
    summary["rare_lt_0_1pct"] = summary["cell_fraction"] < 0.001
    summary["one_dataset_only"] = summary["n_datasets"] == 1

    disease_presence = (
        d.groupby(["hgca_celltype_v1", "disease_group"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    for col in ["Health/control", "Disease/case", "Other/unknown"]:
        if col not in disease_presence:
            disease_presence[col] = 0
    disease_presence = disease_presence.gt(0).astype(int).reset_index()
    summary = summary.merge(disease_presence, on="hgca_celltype_v1", how="left")

    tax_order = {label: i for i, label in enumerate(ordered_types)}
    lineage_map = (
        taxonomy[["hgca_celltype_v1", "hgca_celltype_level1"]]
        .dropna(subset=["hgca_celltype_v1"])
        .drop_duplicates("hgca_celltype_v1")
    )
    summary = summary.merge(lineage_map, on="hgca_celltype_v1", how="left")
    summary["taxonomy_order"] = summary["hgca_celltype_v1"].map(tax_order)
    summary = summary.sort_values(
        ["taxonomy_order", "hgca_celltype_level1", "hgca_celltype_v1"],
        na_position="last",
    )

    tissue = (
        d.groupby(["tissue_level_1", "hgca_celltype_v1"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    tissue["present"] = (tissue["n_cells"] > 0).astype(int)

    dataset = (
        d.groupby(["dataset_id", "hgca_celltype_v1"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    return summary, tissue, dataset


def build_compositional_enrichment(
    df: pd.DataFrame, ordered_types: list[str]
) -> pd.DataFrame:
    """Compute per-sample CLR composition and category-wise row z-scores."""
    d = df.dropna(
        subset=["dataset_id", "sample_id", "hgca_celltype_v1"]
    ).copy()
    d["_sample_key"] = d["dataset_id"].astype(str) + "\x1f" + d["sample_id"].astype(str)

    annotation_specs = [
        (
            "Tissue",
            "tissue_level_1",
            ["duodenum", "jejunum", "ileum", "colon", "mesentery", "accessory"],
        ),
        (
            "Collection method",
            "sample_collection_method",
            ["biopsy", "surgical resection"],
        ),
        (
            "Radial layer",
            "radial_tissue_term",
            ["EPI", "LP", "EPI_LP", "EPI_LP_MUSC", "WM"],
        ),
    ]
    celltypes = [
        label for label in ordered_types
        if label in set(d["hgca_celltype_v1"].dropna())
    ]

    frames: list[pd.DataFrame] = []
    for annotation_group, column, preferred_order in annotation_specs:
        one_annotation = d.dropna(subset=[column]).copy()
        # A few source sample IDs span multiple anatomical sites. Treat each
        # sample/category combination as a sample stratum instead of silently
        # assigning the whole sample to one category.
        one_annotation["_sample_stratum"] = (
            one_annotation["_sample_key"]
            + "\x1f"
            + one_annotation[column].astype(str)
        )
        levels = one_annotation[column].unique().tolist()
        level_order = [level for level in preferred_order if level in levels]
        level_order.extend(sorted(set(levels) - set(level_order)))
        counts = (
            one_annotation.groupby(
                ["_sample_stratum", "hgca_celltype_v1"], observed=True
            )
            .size()
            .unstack(fill_value=0)
            .reindex(columns=celltypes, fill_value=0)
        )
        # Pseudocount 1 matches Methods / patpy CLR.
        logged = np.log(counts.astype(float) + 1.0)
        clr = logged.sub(logged.mean(axis=1), axis=0)
        stratum_metadata = one_annotation.drop_duplicates(
            "_sample_stratum"
        ).set_index("_sample_stratum")
        grouped = (
            clr.assign(_annotation_level=stratum_metadata.loc[clr.index, column])
            .groupby("_annotation_level", observed=True)
            .mean()
            .reindex(level_order)
        )
        centered = grouped.sub(grouped.mean(axis=0), axis=1)
        scale = grouped.std(axis=0, ddof=0).replace(0, np.nan)
        row_z = centered.div(scale, axis=1).fillna(0)
        n_samples = one_annotation.drop_duplicates(
            ["_sample_key", column]
        )[column].value_counts()

        mean_long = grouped.rename_axis("annotation_level").stack().rename("mean_clr")
        z_long = row_z.rename_axis("annotation_level").stack().rename("row_z")
        one = pd.concat([mean_long, z_long], axis=1).reset_index()
        one = one.rename(columns={"hgca_celltype_v1": "hgca_celltype_v1"})
        one["annotation_group"] = annotation_group
        one["annotation_column"] = column
        one["level_order"] = one["annotation_level"].map(
            {level: index for index, level in enumerate(level_order)}
        )
        one["n_samples"] = one["annotation_level"].map(n_samples).astype(int)
        frames.append(one)

    return pd.concat(frames, ignore_index=True)[
        [
            "annotation_group",
            "annotation_column",
            "annotation_level",
            "level_order",
            "hgca_celltype_v1",
            "n_samples",
            "mean_clr",
            "row_z",
        ]
    ]


CAP_LABEL_ALIASES = {
    "Endothelial": "vascular endothelial",
    "Arteriolar Endothelial": "vascular endothelial Arteriolar",
    "Capillary Endothelial": "vascular endothelial Capillary",
    "Post Arteriole Capillary Endothelial (PAC)": (
        "vascular endothelial Post arteriole capillary (PAC)"
    ),
    "Pre Venule Capillary Endothelial (PVC)": (
        "vascular endothelial Pre venule capillary (PVC)"
    ),
    "Venular Endothelial": "vascular endothelial Venular",
    "Lymphatic Endothelial": "vascular endothelial Lymphatic",
    "Lacteal Endothelial": "vascular endothelial Lacteals",
}


def build_cap_celltype_summary(
    taxonomy: pd.DataFrame,
    atlas_metadata: pd.DataFrame | None = None,
    labels_path: Path = DEFAULT_CAP_LABELS,
    feedback_path: Path = DEFAULT_CAP_FEEDBACK,
    bridge_path: Path = DEFAULT_CAP_BRIDGE,
) -> pd.DataFrame:
    """Map canonical CAP project-901 vote summaries onto HGCA v1 taxonomy rows."""
    labels = pd.read_csv(labels_path)
    feedback = pd.read_csv(feedback_path)
    bridge = pd.read_csv(bridge_path, dtype="string")
    for frame in (labels, feedback):
        frame["lineage_key"] = clean_label(frame["lineage"]).str.lower()
        name_column = "label_name"
        frame["label_key"] = clean_label(frame[name_column]).map(_norm_token)

    split_merge = (
        feedback.assign(
            is_split_merge=feedback["explanation_type"].isin(["split", "merge"])
        )
        .groupby(["lineage_key", "label_key"], observed=True)["is_split_merge"]
        .sum()
        .rename("n_split_merge")
        .reset_index()
    )
    labels = labels.merge(
        split_merge,
        on=["lineage_key", "label_key"],
        how="left",
        validate="many_to_one",
    )
    labels["n_split_merge"] = labels["n_split_merge"].fillna(0).astype(int)
    labels = (
        labels.groupby(["lineage_key", "label_key"], observed=True)
        .agg(
            label_name=("label_name", "first"),
            score_agree=("score_agree", "sum"),
            score_disagree=("score_disagree", "sum"),
            score_idk=("score_idk", "sum"),
            n_feedback=("n_feedback", "sum"),
            n_split_merge=("n_split_merge", "max"),
        )
        .reset_index()
    )
    lookup = labels.set_index(["lineage_key", "label_key"])
    globally_unique = labels.groupby("label_key", observed=True).filter(
        lambda group: len(group) == 1
    )
    global_lookup = globally_unique.set_index("label_key")
    bridge_lookup = (
        bridge.dropna(subset=["hgca_celltype_v1", "cap_label_matched"])
        .drop_duplicates("hgca_celltype_v1", keep="first")
        .set_index("hgca_celltype_v1")["cap_label_matched"]
        .to_dict()
    )
    empirical_v0: dict[str, str] = {}
    if atlas_metadata is not None:
        v0_counts = (
            atlas_metadata.dropna(
                subset=["hgca_celltype_v1", "hgca_celltype_v0"]
            )
            .groupby(
                ["hgca_celltype_v1", "hgca_celltype_v0"], observed=True
            )
            .size()
            .rename("n_cells")
            .reset_index()
            .sort_values(
                ["hgca_celltype_v1", "n_cells"],
                ascending=[True, False],
            )
            .drop_duplicates("hgca_celltype_v1")
        )
        empirical_v0 = dict(
            zip(v0_counts["hgca_celltype_v1"], v0_counts["hgca_celltype_v0"])
        )

    tax_columns = [
        "hgca_celltype_level1",
        *TAXONOMY_LEVEL_COLUMNS[1:],
        "hgca_celltype_v0",
        "hgca_celltype_v1",
    ]
    tax = taxonomy[tax_columns].dropna(
        subset=["hgca_celltype_v1"]
    ).drop_duplicates("hgca_celltype_v1").copy()
    for column in tax_columns:
        tax[column] = clean_label(tax[column])

    tax_rows = list(tax.itertuples(index=False))
    ancestor_v0: dict[str, str] = {}
    for target in tax_rows:
        target_path = tuple(
            value
            for value in (
                getattr(target, column) for column in TAXONOMY_LEVEL_COLUMNS
            )
            if pd.notna(value) and value != ""
        )
        candidates: list[tuple[int, str]] = []
        for candidate in tax_rows:
            if pd.isna(candidate.hgca_celltype_v0):
                continue
            candidate_path = tuple(
                value
                for value in (
                    getattr(candidate, column)
                    for column in TAXONOMY_LEVEL_COLUMNS
                )
                if pd.notna(value) and value != ""
            )
            if (
                len(candidate_path) <= len(target_path)
                and target_path[: len(candidate_path)] == candidate_path
            ):
                candidates.append(
                    (len(candidate_path), str(candidate.hgca_celltype_v0))
                )
        if candidates:
            ancestor_v0[str(target.hgca_celltype_v1)] = max(candidates)[1]

    records: list[dict[str, object]] = []
    for row in tax.itertuples(index=False):
        lineage_key = str(row.hgca_celltype_level1).strip().lower()
        candidates = [
            empirical_v0.get(str(row.hgca_celltype_v1), ""),
            row.hgca_celltype_v0,
            ancestor_v0.get(str(row.hgca_celltype_v1), ""),
            bridge_lookup.get(str(row.hgca_celltype_v1), ""),
            CAP_LABEL_ALIASES.get(str(row.hgca_celltype_v1), ""),
            row.hgca_celltype_v1,
        ]
        match = None
        match_source = ""
        for source, candidate in zip(
            (
                "atlas_v0",
                "taxonomy_v0",
                "ancestor_v0",
                "canonical_bridge",
                "alias",
                "v1_fallback",
            ),
            candidates,
        ):
            key = (lineage_key, _norm_token(candidate))
            if key in lookup.index:
                match = lookup.loc[key]
                match_source = source
                break
            if key[1] in global_lookup.index:
                match = global_lookup.loc[key[1]]
                match_source = f"{source}_cross_lineage"
                break

        record: dict[str, object] = {
            "hgca_celltype_v1": row.hgca_celltype_v1,
            "cap_reviewed": int(match is not None),
            "cap_label_name": "" if match is None else match["label_name"],
            "cap_match_source": match_source,
            "cap_v0_bridge": (
                empirical_v0.get(str(row.hgca_celltype_v1))
                or ancestor_v0.get(str(row.hgca_celltype_v1), "")
            ),
            "cap_vote_count": 0 if match is None else int(match["n_feedback"]),
            "cap_agreement_fraction": np.nan,
            "cap_split_merge_fraction": np.nan,
            "cap_uncertain_fraction": np.nan,
        }
        if match is not None:
            agree = float(match["score_agree"])
            disagree = float(match["score_disagree"])
            uncertain = float(match["score_idk"])
            decisive = agree + disagree
            total = decisive + uncertain
            record["cap_agreement_fraction"] = (
                agree / decisive if decisive > 0 else np.nan
            )
            record["cap_split_merge_fraction"] = (
                float(match["n_split_merge"]) / decisive
                if decisive > 0
                else np.nan
            )
            record["cap_uncertain_fraction"] = (
                uncertain / total if total > 0 else np.nan
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _scale01(series: pd.Series, *, log1p: bool = False, invert: bool = False) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    if log1p:
        x = np.log1p(x)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    y = (x - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=x.index)
    return 1 - y if invert else y


def plot_celltype_sidecar(
    summary: pd.DataFrame, tissue_long: pd.DataFrame, out: Path
) -> None:
    types = summary["hgca_celltype_v1"].tolist()
    type_index = {x: i for i, x in enumerate(types)}
    metrics = pd.DataFrame(
        {
            "Cells": _scale01(summary["n_cells"], log1p=True),
            "Datasets": _scale01(summary["n_datasets"]),
            "Samples": _scale01(summary["n_samples"], log1p=True),
            "Donors": _scale01(summary["n_donors"], log1p=True),
            "Tissues": _scale01(summary["n_tissues"]),
            "Author-label concordance": summary["author_v0_exact_fraction"].fillna(0),
            "Rare (<0.1% of cells)": summary["rare_lt_0_1pct"].astype(float),
            "One dataset only": summary["one_dataset_only"].astype(float),
            "Health/control": summary["Health/control"].fillna(0).astype(float),
            "Disease/case": summary["Disease/case"].fillna(0).astype(float),
        }
    ).T.astype(float)

    tissues = sorted(tissue_long["tissue_level_1"].dropna().unique())
    tissue_matrix = (
        tissue_long.pivot_table(
            index="tissue_level_1",
            columns="hgca_celltype_v1",
            values="present",
            aggfunc="max",
            fill_value=0,
        )
        .reindex(index=tissues, columns=types, fill_value=0)
        .astype(float)
    )

    width_mm = 180
    height_mm = min(170, max(100, 3.2 * (len(metrics) + len(tissues)) + 24))
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[len(metrics), max(1, len(tissues))],
        hspace=0.08,
    )
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "gca_blue", [PALETTE["absent"], PALETTE["hgca"]]
    )
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(metrics.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax1.set_yticks(np.arange(len(metrics)), metrics.index)
    ax1.set_xticks([])
    ax1.set_title("Cell-type atlas breadth and author-label reannotation", loc="left")

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.imshow(tissue_matrix.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax2.set_yticks(np.arange(len(tissues)), tissues)
    ax2.set_xticks(np.arange(len(types)), types, rotation=90)
    ax2.set_ylabel("Tissue presence")
    ax2.tick_params(axis="x", pad=1)
    for ax in (ax1, ax2):
        ax.set_xticks(np.arange(-0.5, len(types), 1), minor=True)
        ax.grid(which="minor", axis="x", color="white", linewidth=0.25)
        ax.tick_params(which="minor", bottom=False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.25)

    fig.subplots_adjust(left=0.18, right=0.995, top=0.96, bottom=0.31)
    save_figure(fig, out / "fig2_arbol_celltype_sidecar")
    plt.close(fig)


def plot_dataset_heatmap(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    dataset_long: pd.DataFrame,
    out: Path,
) -> pd.DataFrame:
    types = summary["hgca_celltype_v1"].tolist()
    matrix = dataset_long.pivot_table(
        index="dataset_id",
        columns="hgca_celltype_v1",
        values="n_cells",
        aggfunc="sum",
        fill_value=0,
    ).reindex(columns=types, fill_value=0)

    valid = df.dropna(subset=["dataset_id"]).copy()
    valid["author_v0_exact"] = valid["closest_GCA_celltype"].eq(valid["hgca_celltype_v0"])
    attrs = (
        valid.groupby("dataset_id", observed=True)
        .agg(
            n_cells=("dataset_id", "size"),
            n_samples=("sample_id", "nunique"),
            n_donors=("donor_id", "nunique"),
            n_tissues=("tissue_level_2", "nunique"),
            n_celltypes=("hgca_celltype_v1", "nunique"),
            author_v0_exact_fraction=("author_v0_exact", "mean"),
        )
        .reindex(matrix.index)
    )
    matrix = matrix.loc[attrs.sort_values("n_cells", ascending=False).index]
    attrs = attrs.loc[matrix.index]

    attr_plot = pd.DataFrame(
        {
            "Cells": _scale01(attrs["n_cells"], log1p=True),
            "Samples": _scale01(attrs["n_samples"], log1p=True),
            "Donors": _scale01(attrs["n_donors"], log1p=True),
            "Tissues": _scale01(attrs["n_tissues"]),
            "Cell types": _scale01(attrs["n_celltypes"]),
            "Author-label concordance": attrs["author_v0_exact_fraction"],
        },
        index=attrs.index,
    ).astype(float)

    fig = plt.figure(figsize=(180 / 25.4, 170 / 25.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[6, len(types)], wspace=0.03)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "gca_blue", [PALETTE["absent"], PALETTE["sky"], PALETTE["hgca"]]
    )
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(attr_plot.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax0.set_yticks(np.arange(len(attrs)), attrs.index)
    ax0.set_xticks(np.arange(attr_plot.shape[1]), attr_plot.columns, rotation=90)

    ax1 = fig.add_subplot(gs[1])
    log_counts = np.log10(matrix.to_numpy(dtype=float) + 1)
    ax1.imshow(log_counts, aspect="auto", cmap=cmap, vmin=0)
    ax1.set_yticks(np.arange(len(attrs)), [""] * len(attrs))
    ax1.set_xticks(np.arange(len(types)), types, rotation=90)
    fig.suptitle(
        "Dataset attributes and HGCA cell-type origin (log10 cells + 1)",
        x=0.13,
        y=0.985,
        ha="left",
        fontsize=7,
    )

    for ax in (ax0, ax1):
        for spine in ax.spines.values():
            spine.set_linewidth(0.25)
    fig.subplots_adjust(left=0.13, right=0.995, top=0.95, bottom=0.34)
    save_figure(fig, out / "fig2_supp_dataset_by_celltype")
    plt.close(fig)

    attrs.reset_index().to_csv(out.parent / "data" / "dataset_summary.csv", index=False)
    return matrix


def _prediction_counts(
    files: Iterable[Path], *, require_unique_cell_ids: bool = False
) -> pd.DataFrame:
    frames = []
    for path in sorted(files):
        try:
            usecols = ["cell_id", "true_label"] if require_unique_cell_ids else ["true_label"]
            one = pd.read_csv(path, usecols=usecols)
        except (ValueError, pd.errors.EmptyDataError):
            continue
        frames.append(one)
    if not frames:
        return pd.DataFrame(columns=["label", "n_cells"])
    combined = pd.concat(frames, ignore_index=True)
    if require_unique_cell_ids:
        duplicated = combined["cell_id"].astype(str).duplicated(keep=False)
        if duplicated.any():
            examples = combined.loc[duplicated, "cell_id"].astype(str).unique()[:5]
            raise ValueError(
                "PanGI held-out predictions do not count each atlas cell exactly once; "
                f"found {int(duplicated.sum()):,} duplicate rows, including {examples.tolist()}"
            )
        print(
            f"Verified PanGI healthy-atlas counts from {len(combined):,} unique cell IDs"
        )
    labels = combined["true_label"].dropna().astype(str)
    return labels.value_counts().rename_axis("label").rename("n_cells").reset_index()


def _norm_token(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(label).lower())


# Primary biological inferences for colleague review. Values are:
# (HGCA v1 primary, HGCA v1 alternatives, relationship, confidence, rationale).
# The primary is always an exact term in GCA_taxonomy_2026_CAP.csv; where HGCA
# lacks the PanGI identity, it is the nearest defensible parent and the
# relationship/rationale state that limitation explicitly.
PANGI_INFERRED_MAP: dict[str, tuple[str, str, str, str, str]] = {
    "Enterocyte": ("Enterocytes", "", "equivalent", "high", "Direct singular/plural match."),
    "Keratinocyte_stratified": ("Epithelial", "", "no direct counterpart; mapped to parent", "medium", "HGCA lacks stratified squamous keratinocyte subtypes."),
    "B_plasma_IgA1": ("Plasma IGA", "", "PanGI narrower", "high", "HGCA combines IgA1 and IgA2 plasma cells."),
    "Mesoderm_2": ("Stromal", "", "no direct counterpart; mapped to parent", "medium", "Undifferentiated mesoderm state is absent from HGCA."),
    "Crypt_fibroblast_PI16": ("Submucosal Fibroblasts (S3)", "Subepithelial Fibroblasts (S2)", "partial overlap", "medium", "PI16 crypt-support fibroblasts most closely match S3, with possible S2 overlap."),
    "Lamina_propria_fibroblast_ADAMDEC1": ("Lamina propria Fibroblasts (S1)", "", "equivalent", "high", "Anatomy and ADAMDEC1 identity support S1."),
    "B_naive": ("Naive B", "", "equivalent", "high", "Direct match."),
    "Colonocyte": ("Colonocytes", "", "equivalent", "high", "Direct match."),
    "B_memory": ("Memory B", "", "equivalent", "high", "Direct match."),
    "Mesoderm_1": ("Stromal", "", "no direct counterpart; mapped to parent", "medium", "Undifferentiated mesoderm state is absent from HGCA."),
    "EC_venous": ("Venular Endothelial", "", "equivalent", "high", "Closest vessel-class match."),
    "Trm_CD4": ("CD4 Memory", "", "PanGI narrower", "medium", "HGCA lacks a dedicated CD4 TRM leaf."),
    "Tnaive/cm_CD4": ("CD4 Naive", "CD4 Memory", "PanGI composite", "medium", "PanGI combines naive and central-memory CD4 states."),
    "Trm_CD8": ("CD8 TRM", "", "equivalent", "high", "Direct match."),
    "Villus_fibroblast_F3": ("Subepithelial Fibroblasts (S2)", "", "PanGI narrower", "high", "Villus/subepithelial location and F3 support S2."),
    "TA": ("Transiently Amplifying Cells (TA)", "", "equivalent", "high", "Direct match."),
    "Cycling_fibroblast": ("Fibroblasts", "", "PanGI narrower", "high", "HGCA lacks a cycling fibroblast state."),
    "Goblet": ("Goblet Cells", "", "equivalent", "high", "Direct match."),
    "Mature_colonocyte": ("Crypt Top Colonocytes", "", "partial overlap", "medium", "Crypt-top colonocytes are the mature/late HGCA colonocyte state."),
    "Trm/em_CD8": ("CD8 TRM", "CD8 Effector Memory", "PanGI composite", "medium", "PanGI combines resident- and effector-memory CD8 states."),
    "Epithelial_cycling_G2M": ("Proliferative Epithelial", "", "PanGI narrower", "high", "HGCA captures proliferation but not G2/M phase."),
    "Myofibroblast": ("Myofibroblasts", "", "equivalent", "high", "Direct match."),
    "Epithelial_cycling_S": ("Proliferative Epithelial", "", "PanGI narrower", "high", "HGCA captures proliferation but not S phase."),
    "Trm_Th17": ("CD4 Th17", "", "PanGI narrower", "high", "Resident-memory Th17 is a state subset."),
    "Keratinocyte_outer": ("Epithelial", "", "no direct counterpart; mapped to parent", "medium", "HGCA lacks outer stratified keratinocytes."),
    "SMC_PPLP2": ("Smooth Muscle Cells (SMC)", "", "PanGI narrower", "high", "PPLP2 SMC subtype is absent from HGCA."),
    "gdT": ("Gamma Delta T Cells", "", "equivalent", "high", "Direct match."),
    "Basal": ("Epithelial", "", "no direct counterpart; mapped to parent", "medium", "Likely squamous basal epithelium; no HGCA basal-cell leaf."),
    "Mucous_gland_neck": ("Neck Cells", "", "equivalent", "high", "Direct gastric mucous neck-cell match."),
    "BEST4_enterocyte_colonocyte": ("BEST4 Enterocytes", "BEST4 Colonocytes", "PanGI composite", "medium", "Mixed PanGI label spans small-intestinal and colonic BEST4 cells."),
    "Cycling": ("Proliferative Epithelial", "", "PanGI narrower", "high", "This label occurs in the PanGI epithelial subset."),
    "EC_capillary": ("Capillary Endothelial", "", "equivalent", "high", "Direct vessel-class match."),
    "Epithelial_stem": ("Intestinal Stem Cells (ISC)", "", "equivalent", "high", "Direct intestinal epithelial stem-cell match."),
    "Treg_IL10": ("CD4 Tr1", "CD4 Treg", "partial overlap", "medium", "IL-10 regulatory cells overlap Tr1 but can include FOXP3-positive Tregs."),
    "Surface_foveolar": ("Foveolar Cells", "", "equivalent", "high", "Direct gastric match."),
    "Tfh_naive": ("CD4 Tfh", "", "partial overlap", "medium", "Closest Tfh lineage; naive-Tfh is not an HGCA state."),
    "Vascular_smooth_muscle": ("Vascular SMCs (vSMC)", "", "equivalent", "high", "Direct match."),
    "Macrophage": ("Macrophages", "", "equivalent", "high", "Direct generic match."),
    "B_plasma_IgG": ("Plasma IGG", "", "equivalent", "high", "Direct isotype match."),
    "Pericyte": ("Pericytes", "", "equivalent", "high", "Direct match."),
    "Tfh": ("CD4 Tfh", "", "equivalent", "high", "Direct match."),
    "DC_cDC2": ("cDC2", "", "equivalent", "high", "Direct match."),
    "SMC_CAPN3": ("Smooth Muscle Cells (SMC)", "", "PanGI narrower", "high", "CAPN3 SMC subtype is absent from HGCA."),
    "Mast": ("Mast Cells", "", "equivalent", "high", "Direct match."),
    "Treg": ("CD4 Treg", "", "equivalent", "high", "Direct match."),
    "Tnaive/cm_CD8": ("CD8 Naive", "CD8 Memory", "PanGI composite", "medium", "PanGI combines naive and central-memory CD8 states."),
    "Glial/Enteric_neural_crest": ("Glia", "Progenitor Glia", "partial overlap", "medium", "Glial component matches, but undifferentiated neural-crest cells may be included."),
    "B_plasma_IgM": ("Plasma IGM", "", "equivalent", "high", "Direct isotype match."),
    "B_plasma_IgA2": ("Plasma IGA", "", "PanGI narrower", "high", "HGCA combines IgA1 and IgA2 plasma cells."),
    "EC_arterial_1": ("Arteriolar Endothelial", "", "partial overlap", "medium", "Closest arterial HGCA node; PanGI subtype 1 is not retained."),
    "ILC3": ("ILC3", "", "equivalent", "high", "Direct match."),
    "NK_CD56bright": ("NK Cells", "", "PanGI narrower", "high", "HGCA lacks CD56-bright NK resolution."),
    "Oral_mucosa_fibroblast": ("Fibroblasts", "", "PanGI narrower", "high", "HGCA lacks oral-mucosa-specific fibroblasts."),
    "Proximal_progenitor_ILE": ("Enterocyte Progenitors", "", "PanGI narrower", "medium", "Closest ileal absorptive progenitor mapping."),
    "Serous": ("Secretory Epithelial", "", "no direct counterpart; mapped to parent", "medium", "HGCA lacks serous gland cells."),
    "EC_lymphatic": ("Lymphatic Endothelial", "", "equivalent", "high", "Direct lineage match."),
    "Enteric_neural_crest_cycling": ("Progenitor Glia", "Neurons", "partial overlap", "low", "Closest HGCA precursor-like enteric neural-crest compartment."),
    "Monocyte": ("Monocytes", "", "equivalent", "high", "Direct generic match."),
    "Macrophage_LYVE1": ("Perivascular Resident Macrophages", "Tissue Resident Macrophages", "partial overlap", "medium", "LYVE1 supports resident/perivascular identity but is not location-specific alone."),
    "Proximal_progenitor_DUO/JEJ": ("Enterocyte Progenitors", "", "PanGI narrower", "medium", "Closest duodenal/jejunal absorptive progenitor mapping."),
    "B_GC_I": ("GC B Dark Zone (GC B DZ)", "", "equivalent", "high", "GC-I most closely corresponds to dark-zone GC B cells."),
    "EC_arterial_2": ("Arteriolar Endothelial", "", "partial overlap", "medium", "Closest arterial HGCA node; PanGI subtype 2 is not retained."),
    "B_GC_II": ("GC B Light Zone (GC B LZ)", "", "equivalent", "medium", "GC-II most closely corresponds to light-zone GC B cells."),
    "gdT_naive": ("Gamma Delta T Cells", "", "PanGI narrower", "high", "HGCA lacks a naive gamma-delta state."),
    "Branch_B_excitatory_motor_neuron": ("Excitatory Neurons", "", "PanGI narrower", "high", "HGCA lacks motor-neuron and branch-B resolution."),
    "Mesothelium": ("Mesothelial", "", "equivalent", "high", "Direct match."),
    "Distal_progenitor": ("Colonocyte Progenitors", "", "PanGI narrower", "medium", "Closest distal/colonic absorptive progenitor mapping."),
    "Immature_pericyte": ("Immature Pericytes", "", "equivalent", "high", "Direct match."),
    "Neuroblast": ("Neurons", "", "no direct counterpart; mapped to parent", "medium", "HGCA lacks a neuronal precursor leaf."),
    "Fibroblast_reticular": ("Fibroblastic Reticular Cells (FRC)", "", "equivalent", "high", "Direct match."),
    "NK_CD16": ("NK Cells", "", "PanGI narrower", "high", "HGCA lacks CD16-positive NK resolution."),
    "Basal_cycling": ("Proliferative Epithelial", "Epithelial", "partial overlap", "medium", "Proliferative state matches, but basal lineage is absent."),
    "Enteroendocrine": ("Enteroendocrine Cells (EEC)", "", "equivalent", "high", "Direct generic match."),
    "MAIT": ("MAIT Cells", "", "equivalent", "high", "Direct match."),
    "Glial_3": ("Glia", "", "partial overlap", "low", "Numeric PanGI glial subtype lacks marker context in the cached results."),
    "Chief": ("Chief Cells", "", "equivalent", "high", "Direct gastric match."),
    "Goblet_progenitor": ("Secretory Progenitors", "", "partial overlap", "medium", "HGCA secretory progenitors include but are not restricted to goblet commitment."),
    "Tuft": ("Tuft Cells", "", "equivalent", "high", "Direct match."),
    "Macrophage_TREM2": ("Macrophages", "", "no direct counterpart; mapped to parent", "medium", "HGCA lacks a TREM2-defined macrophage leaf."),
    "Glial_2": ("Glia", "", "partial overlap", "low", "Numeric PanGI glial subtype lacks marker context in the cached results."),
    "Goblet_cycling": ("Secretory Progenitors", "", "partial overlap", "medium", "Closest proliferative secretory-lineage state."),
    "T/NK_cycling": ("Cycling T Cells", "NK Cells", "PanGI composite", "medium", "HGCA captures cycling T but not cycling NK or a combined population."),
    "Seromucous": ("Secretory Epithelial", "", "no direct counterpart; mapped to parent", "medium", "HGCA lacks seromucous gland cells."),
    "Duct": ("Epithelial", "", "no direct counterpart; mapped to parent", "low", "HGCA lacks glandular duct-cell identity."),
    "Rectum_fibroblast": ("Fibroblasts", "", "PanGI narrower", "high", "Location alone does not justify an S1/S2/S3 subtype."),
    "Pareital": ("Parietal Cells", "", "equivalent", "high", "Typographical variant of parietal."),
    "Branch_B_primary_afferent_neuron": ("Excitatory Neurons", "", "PanGI narrower", "medium", "Primary afferents are generally excitatory; branch identity is lost."),
    "EC_cycling": ("Endothelial", "", "PanGI narrower", "high", "HGCA lacks a cycling endothelial state."),
    "Glial_1": ("Glia", "", "partial overlap", "low", "Numeric PanGI glial subtype lacks marker context in the cached results."),
    "Branch_A_primary_afferent_neuron": ("Excitatory Neurons", "", "PanGI narrower", "medium", "Primary afferents are generally excitatory; branch identity is lost."),
    "DC_cDC1": ("cDC1", "", "equivalent", "high", "Direct match."),
    "ICC": ("Interstitial Cells of Cajal (ICC)", "", "equivalent", "high", "Direct match."),
    "Macrophage_MMP9": ("M1 Macrophages", "Follicle Associated Resident Macrophages", "partial overlap", "low", "MMP9 supports inflammatory macrophages but does not uniquely define an HGCA subtype."),
    "Enterochromaffin": ("EEC Enterochromaffin (EC)", "", "equivalent", "high", "Direct match."),
    "Branch_A_inhibtory_motor_neuron": ("Inhibitory Neurons", "", "PanGI narrower", "high", "HGCA lacks motor-neuron and branch-A resolution."),
    "Angiogenic_pericyte": ("Angiogenic Pericytes", "", "equivalent", "high", "Direct match."),
    "Mucous": ("Secretory Epithelial", "", "no direct counterpart; mapped to parent", "medium", "Likely glandular mucous cells rather than specifically intestinal goblet cells."),
    "Gland_basal": ("Epithelial", "", "no direct counterpart; mapped to parent", "medium", "HGCA lacks gland-basal epithelial cells."),
    "Branch_A_interneuron": ("Neurons", "Excitatory Neurons; Inhibitory Neurons", "partial overlap", "medium", "Excitatory versus inhibitory identity is unspecified."),
    "Oesophagus_fibroblast": ("Fibroblasts", "", "PanGI narrower", "high", "HGCA lacks an esophagus-specific fibroblast label."),
    "B_preB": ("B Cells", "", "no direct counterpart; mapped to parent", "high", "HGCA lacks a pre-B developmental leaf."),
    "Enteroendocrine_MX": ("EEC Mo", "EEC Gastric", "partial overlap", "medium", "Captures motilin-positive M identity but may include ghrelin/X identity."),
    "DC_pDC": ("pDC", "", "equivalent", "high", "Direct match."),
    "Microfold": ("Microfold Cells (M Cells)", "", "equivalent", "high", "Direct match."),
    "Paneth": ("Paneth Cells", "", "equivalent", "high", "Direct match."),
    "NTS": ("EEC N", "", "equivalent", "high", "NTS defines the HGCA N-cell category."),
    "Enteroendocrine_X": ("EEC Gastric", "", "partial overlap", "medium", "X/A-like ghrelin cells are gastric EECs; HGCA lacks an X-cell leaf."),
    "B_plasmablast": ("Plasma Cells", "", "partial overlap", "medium", "HGCA lacks a distinct plasmablast leaf."),
    "SMC_CAPN3_cycling": ("Smooth Muscle Cells (SMC)", "", "PanGI narrower", "high", "CAPN3 and cycling states are absent from HGCA."),
    "B_proB": ("B Cells", "", "no direct counterpart; mapped to parent", "high", "HGCA lacks a pro-B developmental leaf."),
    "Enteroendocrine_G": ("EEC G", "", "equivalent", "high", "Direct gastrin-positive G-cell match."),
    "DC_migratory": ("migDC", "", "equivalent", "high", "Direct match."),
    "Immune_recruiting_pericyte": ("Secretory Pericytes", "", "partial overlap", "medium", "Closest functional HGCA pericyte state."),
    "Mesenchymal_LTO": ("Mesenchymal Lymphoid Tissue Organizer Cells (mLTo Cells)", "", "equivalent", "high", "Direct match."),
    "Erythrocytes": ("Red Blood Cells (RBC)", "", "equivalent", "high", "Direct match."),
    "Melanocyte": ("Stromal", "", "no direct counterpart; mapped to parent", "high", "HGCA has no melanocyte leaf."),
    "Distal_progenitor_PRAC1": ("Colonocyte Progenitors", "", "PanGI narrower", "medium", "PRAC1-positive distal progenitor subtype is not separately represented."),
    "CLDN10": ("Secretory Epithelial", "", "no direct counterpart; mapped to parent", "low", "Marker-only PanGI label lacks a unique HGCA counterpart."),
    "Enteroendocrine_progenitor": ("EEC Progenitors", "", "equivalent", "high", "Direct match."),
    "Keratinocyte_inflammatory": ("Epithelial", "", "no direct counterpart; mapped to parent", "medium", "HGCA lacks inflammatory keratinocyte states."),
    "Gland_duct": ("Epithelial", "", "no direct counterpart; mapped to parent", "medium", "HGCA lacks glandular duct-cell identity."),
    "Gland_fetal": ("Epithelial", "", "no direct counterpart; mapped to parent", "medium", "HGCA lacks fetal glandular epithelial subtypes."),
    "DCS_MUC17": ("Secretory Epithelial", "", "no direct counterpart; mapped to parent", "medium", "HGCA lacks a dedicated MUC17-positive deep-crypt secretory leaf."),
    "Myoepithelial": ("Epithelial", "", "no direct counterpart; mapped to parent", "medium", "HGCA lacks myoepithelial cells."),
    "Mono/neutrophil_MPO": ("Neutrophils", "Classical Monocytes", "PanGI composite", "medium", "MPO favors neutrophils, but the PanGI label explicitly includes monocytes."),
    "DC_langerhans": ("Dendritic Cells (DC)", "", "no direct counterpart; mapped to parent", "high", "HGCA lacks a Langerhans-cell leaf."),
    "Follicular_DC": ("Follicular Dendritic Cells (fDC)", "", "equivalent", "high", "Direct stromal-cell match."),
    "Myoblast/myocyte": ("Stromal", "Smooth Muscle Cells (SMC)", "no direct counterpart; mapped to parent", "medium", "HGCA lacks generic myoblast/skeletal-myocyte labels."),
    "Keratinocyte_fetal": ("Epithelial", "", "no direct counterpart; mapped to parent", "medium", "HGCA lacks fetal keratinocyte subtypes."),
    "DCS_MUC17_cycling": ("Secretory Progenitors", "Secretory Epithelial", "partial overlap", "low", "Closest cycling secretory compartment; DCS identity is absent."),
    "Macrophage_CD5L": ("Tissue Resident Macrophages", "", "partial overlap", "medium", "CD5L supports resident macrophage identity but is not unique."),
    "Eosinophil/basophil": ("Eosinophils", "Basophils", "PanGI composite", "high", "HGCA separates the two granulocyte types."),
    "Ionocytes": ("Epithelial", "", "no direct counterpart; mapped to parent", "medium", "HGCA lacks an epithelial ionocyte leaf."),
    "Gland_mucous": ("Secretory Epithelial", "", "no direct counterpart; mapped to parent", "medium", "HGCA lacks gland-specific mucous cells."),
    "Gastric_fetal_epithelial": ("Epithelial", "", "no direct counterpart; mapped to parent", "high", "HGCA lacks fetal gastric epithelial subtypes."),
    "SF_like": ("Foveolar Cells", "", "partial overlap", "medium", "Interpreted as surface-foveolar-like from PanGI epithelial context."),
    "Megakaryocyte/platelet": ("Megakaryocytes", "", "PanGI composite", "high", "HGCA represents megakaryocytes but has no platelet leaf."),
}


def build_pangi_inputs(
    results_root: Path,
    data_dir: Path,
    taxonomy: pd.DataFrame,
    atlas_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hgca = (
        atlas_metadata["hgca_celltype_v1"]
        .dropna()
        .astype(str)
        .value_counts()
        .rename_axis("label")
        .rename("n_cells")
        .reset_index()
    )
    pangi_files = list(results_root.glob(
        "pangi_*/slurm_benchmarks/e3_pangi_20260416T185203Z_pangi_*"
        "/fold_*/predictions/predictions_scanvi_level_3_annot.csv"
    ))
    pangi = _prediction_counts(pangi_files, require_unique_cell_ids=True)
    hgca.to_csv(data_dir / "hgca_v1_reference_label_counts.csv", index=False)
    pangi.to_csv(data_dir / "pangi_level3_reference_label_counts.csv", index=False)

    crosswalk = pangi.rename(
        columns={"label": "pangi_level3_label", "n_cells": "pangi_n_cells"}
    ).copy()

    observed = set(crosswalk["pangi_level3_label"])
    inferred = set(PANGI_INFERRED_MAP)
    missing_inference = sorted(observed - inferred)
    stale_inference = sorted(inferred - observed)
    if missing_inference or stale_inference:
        raise ValueError(
            "PanGI inference dictionary is out of sync. "
            f"Missing={missing_inference}; stale={stale_inference}"
        )

    inferred_df = pd.DataFrame.from_dict(
        PANGI_INFERRED_MAP,
        orient="index",
        columns=[
            "hgca_v1_label",
            "alternative_hgca_v1_labels",
            "relationship_to_hgca_v1",
            "confidence",
            "mapping_notes",
        ],
    ).rename_axis("pangi_level3_label").reset_index()
    crosswalk = crosswalk.merge(
        inferred_df, on="pangi_level3_label", how="left", validate="one_to_one"
    )

    tax_lookup = (
        taxonomy[
            ["hgca_celltype_v1", "hgca_celltype_v0", "hgca_celltype_level1"]
        ]
        .dropna(subset=["hgca_celltype_v1"])
        .drop_duplicates("hgca_celltype_v1")
        .rename(
            columns={
                "hgca_celltype_v1": "hgca_v1_label",
                "hgca_celltype_v0": "hgca_v0_label",
                "hgca_celltype_level1": "lineage",
            }
        )
    )
    crosswalk = crosswalk.merge(
        tax_lookup, on="hgca_v1_label", how="left", validate="many_to_one"
    )
    bad_targets = crosswalk.loc[crosswalk["lineage"].isna(), "hgca_v1_label"].unique()
    if len(bad_targets):
        raise ValueError(f"Inferred HGCA v1 labels absent from taxonomy: {bad_targets}")

    crosswalk["suggested_hgca_v1_label"] = crosswalk["hgca_v1_label"]
    crosswalk["hgca_v0_label"] = crosswalk["hgca_v0_label"].fillna("")
    crosswalk["v0_alignment_status"] = np.where(
        crosswalk["hgca_v0_label"].ne(""),
        "direct v0 term from taxonomy",
        "no direct v0 term (new or reorganized in v1)",
    )
    crosswalk["mapping_status"] = "inferred_needs_review"
    crosswalk["review_status"] = "not_reviewed"
    crosswalk["parent_path"] = ""
    crosswalk["include"] = True

    crosswalk_path = data_dir / "pangi_to_hgca_v1_crosswalk.csv"
    if crosswalk_path.is_file():
        existing = pd.read_csv(crosswalk_path, dtype="string").drop_duplicates(
            "pangi_level3_label", keep="last"
        )
        existing = existing.set_index("pangi_level3_label")
        editable = [
            "hgca_v1_label",
            "alternative_hgca_v1_labels",
            "relationship_to_hgca_v1",
            "confidence",
            "mapping_status",
            "review_status",
            "parent_path",
            "include",
            "mapping_notes",
        ]
        for col in editable:
            if col in existing:
                prior_values = crosswalk["pangi_level3_label"].map(existing[col])
                crosswalk[col] = prior_values.where(
                    prior_values.notna(), crosswalk[col]
                )

    # Re-derive taxonomy fields after preserving curator-selected v1 labels.
    tax_by_v1 = tax_lookup.set_index("hgca_v1_label")
    crosswalk["hgca_v0_label"] = crosswalk["hgca_v1_label"].map(
        tax_by_v1["hgca_v0_label"]
    )
    crosswalk["lineage"] = crosswalk["hgca_v1_label"].map(tax_by_v1["lineage"])
    bad_targets = crosswalk.loc[
        crosswalk["lineage"].isna(), "hgca_v1_label"
    ].dropna().unique()
    if len(bad_targets):
        raise ValueError(f"Curated HGCA v1 labels absent from taxonomy: {bad_targets}")
    crosswalk["hgca_v0_label"] = crosswalk["hgca_v0_label"].fillna("")
    crosswalk["v0_alignment_status"] = np.where(
        crosswalk["hgca_v0_label"].ne(""),
        "direct v0 term from taxonomy",
        "no direct v0 term (new or reorganized in v1)",
    )

    column_order = [
        "pangi_level3_label",
        "pangi_n_cells",
        "hgca_v1_label",
        "hgca_v0_label",
        "alternative_hgca_v1_labels",
        "relationship_to_hgca_v1",
        "confidence",
        "lineage",
        "v0_alignment_status",
        "mapping_status",
        "review_status",
        "include",
        "parent_path",
        "mapping_notes",
        "suggested_hgca_v1_label",
    ]
    crosswalk = crosswalk[column_order]
    crosswalk.to_csv(crosswalk_path, index=False)
    print(
        f"PanGI/HGCA inputs: {len(pangi)} PanGI labels, {len(hgca)} HGCA labels; "
        f"{len(crosswalk)} biological mappings awaiting review"
    )
    return hgca, crosswalk


def write_summary(
    overall: pd.DataFrame,
    celltypes: pd.DataFrame,
    pangi_crosswalk: pd.DataFrame,
    data_dir: Path,
) -> None:
    row = overall.iloc[0]
    lines = [
        "# Figure 2 atlas-evidence build summary",
        "",
        f"- Cells compared: {int(row['n_cells']):,}",
        f"- Exact author-crosswalk/HGCA v0 matches: {int(row['n_exact_match']):,}",
        (
            f"- Reannotated at exact label level: {int(row['n_reannotated']):,} "
            f"({100 * row['reannotated_fraction']:.2f}%)"
        ),
        (
            "- Interpretation: this is label-set discordance, not a validated "
            "misannotation rate; the author crosswalk is often coarser than HGCA v0."
        ),
        f"- HGCA v1 cell types represented: {len(celltypes):,}",
        f"- Rare HGCA v1 types (<0.1% cells): {int(celltypes['rare_lt_0_1pct'].sum()):,}",
        f"- HGCA v1 types present in one dataset only: {int(celltypes['one_dataset_only'].sum()):,}",
        (
            f"- PanGI labels requiring curator crosswalk review: "
            f"{int(pangi_crosswalk['review_status'].eq('not_reviewed').sum()):,}"
        ),
        "",
    ]
    (data_dir / "build_summary.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(os.environ.get("HGCA_H5AD", str(DEFAULT_METADATA))),
    )
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument(
        "--benchmark-results", type=Path, default=DEFAULT_BENCHMARK_RESULTS
    )
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use the bundled demo slice and skip LODO/CAP/PanGI caches.",
    )
    parser.add_argument(
        "--allow-missing-lodo",
        action="store_true",
        help="Write metadata tables even if LODO F1 summaries are absent.",
    )
    parser.add_argument(
        "--skip-lodo",
        action="store_true",
        help="Do not read LODO F1 caches (demo / laptop check).",
    )
    parser.add_argument(
        "--skip-pangi",
        action="store_true",
        help="Skip PanGI prediction caches (demo / laptop check).",
    )
    parser.add_argument(
        "--skip-cap",
        action="store_true",
        help="Skip CAP vote tables (demo / laptop check).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    demo_h5ad = REPO_ROOT / "data" / "demo" / "hgca_all_lineages_v1_demo.h5ad"
    demo_tax = REPO_ROOT / "data" / "demo" / "GCA_taxonomy_2026_CAP.csv"
    expected_fig2 = REPO_ROOT / "data" / "demo" / "expected" / "fig2"
    if args.demo:
        if "--metadata" not in sys.argv:
            args.metadata = demo_h5ad
        if "--taxonomy" not in sys.argv:
            args.taxonomy = demo_tax
        args.allow_missing_lodo = True
        args.skip_lodo = True
        args.skip_pangi = True
        args.skip_cap = True
        if args.figure_dir == FIGURE_DIR:
            args.figure_dir = expected_fig2
            print(f"Demo input: writing to {args.figure_dir}")
        print("DEMO MODE: results are for software checking, not manuscript figures.")
    elif "demo" in Path(args.metadata).name and args.figure_dir == FIGURE_DIR:
        args.figure_dir = expected_fig2
        print(f"Demo input: writing to {args.figure_dir}")
        print("DEMO MODE: results are for software checking, not manuscript figures.")

    figure_dir = args.figure_dir.resolve()
    data_dir = figure_dir / "data"
    out_dir = figure_dir / "out"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    df = load_metadata(args.metadata.resolve())
    order, taxonomy = taxonomy_order(args.taxonomy.resolve())

    per_class, overall = author_v0_metrics(df)
    per_class.to_csv(data_dir / "author_crosswalk_v0_per_class_metrics.csv", index=False)
    overall.to_csv(data_dir / "author_crosswalk_v0_overall.csv", index=False)
    plot_author_v0_metrics(per_class, overall, out_dir)

    celltypes, tissue, dataset = build_celltype_tables(df, order, taxonomy)
    empty_lodo = pd.DataFrame(
        columns=[
            "hgca_celltype_v1",
            "lodo_f1",
            "lodo_f1_support",
            "lodo_f1_std",
            "lodo_f1_n_folds",
            "lineage",
            "lodo_run",
        ]
    )
    if args.skip_lodo:
        print("Skipping LODO F1 caches (--skip-lodo / --demo).")
        lodo_f1 = empty_lodo
    else:
        try:
            lodo_f1 = load_lodo_v1_per_class_f1(args.benchmark_results.resolve())
        except FileNotFoundError:
            if not args.allow_missing_lodo:
                raise
            print("LODO F1 summaries not found; writing metadata tables without F1.")
            lodo_f1 = empty_lodo
    celltypes = celltypes.merge(
        lodo_f1.drop(columns=["lineage"], errors="ignore"),
        on="hgca_celltype_v1",
        how="left",
    )
    matched = int(celltypes["lodo_f1"].notna().sum()) if "lodo_f1" in celltypes else 0
    if matched == 0 and not args.allow_missing_lodo:
        raise RuntimeError("No HGCA v1 cell types matched LODO F1 summaries")
    lodo_f1.to_csv(data_dir / "lodo_v1_per_class_f1_summary.csv", index=False)
    celltypes.to_csv(data_dir / "celltype_atlas_summary.csv", index=False)
    tissue.to_csv(data_dir / "celltype_tissue_presence_long.csv", index=False)
    dataset.to_csv(data_dir / "dataset_celltype_counts_long.csv", index=False)
    composition = build_compositional_enrichment(df, order)
    composition.to_csv(
        data_dir / "celltype_compositional_enrichment_long.csv", index=False
    )
    if args.skip_cap:
        print("Skipping CAP vote tables (--skip-cap / --demo).")
    else:
        cap_summary = build_cap_celltype_summary(taxonomy, atlas_metadata=df)
        cap_summary.to_csv(data_dir / "cap_celltype_summary.csv", index=False)
    dataset_matrix = plot_dataset_heatmap(df, celltypes, dataset, out_dir)
    dataset_matrix.to_csv(data_dir / "dataset_celltype_counts_matrix.csv")

    if args.skip_pangi:
        hgca = (
            df["hgca_celltype_v1"]
            .dropna()
            .astype(str)
            .value_counts()
            .rename_axis("label")
            .rename("n_cells")
            .reset_index()
        )
        hgca.to_csv(data_dir / "hgca_v1_reference_label_counts.csv", index=False)
        pangi_crosswalk = pd.DataFrame({"review_status": pd.Series(dtype="string")})
        print("Skipping PanGI caches (--skip-pangi / --demo).")
    else:
        _, pangi_crosswalk = build_pangi_inputs(
            args.benchmark_results.resolve(), data_dir, taxonomy, df
        )
    write_summary(overall, celltypes, pangi_crosswalk, data_dir)
    print(f"Wrote Figure 2 data under {data_dir}")
    print(f"Wrote Figure 2 plots under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
