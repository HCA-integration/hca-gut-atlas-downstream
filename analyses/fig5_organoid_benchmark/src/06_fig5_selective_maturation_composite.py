#!/usr/bin/env python3
"""Figure 5 composite: selective maturation and regional identity."""
from __future__ import annotations

import textwrap

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from shapely import vectorized as shapely_vectorized
from shapely.geometry import Point, Polygon
from shapely.ops import polylabel, unary_union
from skimage import measure as sk_measure
from statsmodels.stats.multitest import multipletests

DISH_CENTER = (0.5, 0.5)
DISH_RADIUS = 0.49

import common as C


MM = 1 / 25.4
SEGMENT_COLORS = {
    "Duodenum": "#E9C61D",
    "Jejunum": "#E96475",
    "Ileum": "#208230",
    "Small Intestine": "#3D8B5A",
    "Colon": "#3A68AE",
}
SMALL_INTESTINE_TISSUE_LABELS = {
    "duodenum",
    "jejunum",
    "ileum",
    "small intestine",
}
COLON_TISSUE_LABELS = {
    "colon",
    "ascending colon",
    "transverse colon",
    "descending colon",
    "sigmoid colon",
    "rectum",
    "cecum",
    "caecum",
}
SOURCE_LABELS = {
    "ASC": "Adult stem cell-derived (ASC)",
    "FSC": "Fetal stem cell-derived (FSC)",
    "PSC": "Pluripotent stem cell-derived (PSC)",
}
BRANCH_COLORS = {
    "Progenitor": "#E69F00",
    "Absorptive": "#0072B2",
    "Secretory": "#CC79A7",
}
BRANCH_ORDER = ["Progenitor", "Absorptive", "Secretory"]
PROGENITOR_IDENTITIES = {
    "Intestinal Stem Cells (ISC)",
    "Transiently Amplifying Cells (TA)",
    "Secretory Progenitors",
    "Enterocyte Progenitors",
    "Colonocyte Progenitors",
    "EEC Progenitors",
    "Tuft Progenitors",
}
TIME_COLORS = {
    "≤14 d": "#56B4E9",
    "15–55 d": "#0072B2",
    "≥56 d": "#CC79A7",
}


def configure_style() -> None:
    sns.set_theme(style="ticks", context="paper")
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.labelsize": 6,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "legend.fontsize": 5,
            "axes.titlesize": 6.5,
        }
    )


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )


def short_identity_label(label: str) -> str:
    replacements = {
        "Intestinal Stem Cells (ISC)": "ISC",
        "Transiently Amplifying Cells (TA)": "TA",
        "Secretory Progenitors": "Secretory progenitors",
        "Enterocyte Progenitors": "Enterocyte progenitors",
        "Colonocyte Progenitors": "Colonocyte progenitors",
        "Lower Villus Enterocytes": "Lower-villus enterocytes",
        "Mid Villus Enterocytes": "Mid-villus enterocytes",
        "Villus Tip Enterocytes": "Villus-tip enterocytes",
        "Lower Crypt Colonocytes": "Lower-crypt colonocytes",
        "Mid Crypt Colonocytes": "Mid-crypt colonocytes",
        "Crypt Top Colonocytes": "Crypt-top colonocytes",
        "BEST4 Enterocytes": "BEST4 enterocytes",
        "BEST4 Colonocytes": "BEST4 colonocytes",
        "Mature Goblet Cells": "Mature goblet cells",
        "Goblet Cells": "Goblet cells",
        "Tuft Cells": "Tuft cells",
        "Tuft Progenitors": "Tuft progenitors",
    }
    return replacements.get(
        label, textwrap.shorten(label, width=26, placeholder="…")
    )


def branch_category(label: str, hierarchy: pd.DataFrame) -> str:
    if label in PROGENITOR_IDENTITIES:
        return "Progenitor"
    if label not in hierarchy.index:
        return "Other"
    level2 = str(hierarchy.loc[label, "hgca_celltype_level2"])
    if level2 in {"Secretory Epithelial", "Enteroendocrine Cells (EEC)"}:
        return "Secretory"
    if level2 in {
        "Absorptive Epithelial",
        "Follicle associated enterocyte (FAE)",
    }:
        return "Absorptive"
    return "Other"


def cross_compartment_mask(
    labels: pd.Series, regions: pd.Series
) -> pd.Series:
    return (
        labels.astype(str).str.contains("Colonocyte", case=False, na=False)
        & regions.isin(["Duodenum", "Jejunum", "Ileum"])
    ) | (
        labels.astype(str).str.contains("Enterocyte", case=False, na=False)
        & regions.eq("Colon")
    )


def parse_numeric_culture_day(values: pd.Series) -> pd.Series:
    extracted = values.astype(str).str.extract(
        r"(?i)^\s*(\d+(?:\.\d+)?)\s*day\s*$"
    )[0]
    return pd.to_numeric(extracted, errors="coerce")


def taxonomy_order(hierarchy: pd.DataFrame) -> list[str]:
    frame = hierarchy.reset_index()
    cols = [
        column
        for column in [
            "hgca_celltype_level1",
            "hgca_celltype_level2",
            "hgca_celltype_level3",
            "hgca_celltype_level4",
            "hgca_celltype_v1",
        ]
        if column in frame.columns
    ]
    return frame.sort_values(cols)["hgca_celltype_v1"].tolist()


def build_panel_c_branch_distance() -> tuple[pd.DataFrame, pd.DataFrame]:
    hierarchy = pd.read_csv(
        C.DATA / "hgca_epithelial_hierarchy.csv"
    ).set_index("hgca_celltype_v1")
    metadata = pd.read_csv(C.DATA / "sample_metadata.csv")
    cells = pd.read_csv(
        C.DATA / "per_cell_mapping_qc_flags.csv.gz",
        usecols=[
            "sample_id",
            "hgca_pred_celltype_sysvi_knn",
            "hgca_pred_celltype_sysvi_knn_thresh",
            "d_nn1",
            "confident",
            "strict_mapping_pass",
        ],
    )
    cells = cells.merge(
        metadata[
            [
                "sample_id",
                "publication_display",
                "source_standardized",
                "region_broad",
            ]
        ],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    cells = cells[
        cells["confident"]
        & cells["hgca_pred_celltype_sysvi_knn_thresh"].ne("Unknown")
        & cells["d_nn1"].notna()
    ].copy()
    cells["branch"] = cells["hgca_pred_celltype_sysvi_knn"].map(
        lambda value: branch_category(value, hierarchy)
    )
    cells = cells[
        cells["branch"].isin(BRANCH_ORDER)
        & ~cross_compartment_mask(
            cells["hgca_pred_celltype_sysvi_knn"],
            cells["region_broad"],
        )
    ].copy()
    sample = (
        cells.groupby(
            [
                "sample_id",
                "publication_display",
                "source_standardized",
                "region_broad",
                "branch",
            ],
            observed=True,
        )
        .agg(
            n_cells=("d_nn1", "size"),
            median_nn_distance=("d_nn1", "median"),
        )
        .reset_index()
    )
    sample = sample[sample["n_cells"] >= 20].copy()
    sample["branch"] = pd.Categorical(
        sample["branch"], categories=BRANCH_ORDER, ordered=True
    )

    wide = sample.pivot(
        index="sample_id", columns="branch", values="median_nn_distance"
    )
    contrasts = []
    pairs = [
        ("Absorptive", "Progenitor"),
        ("Secretory", "Progenitor"),
        ("Secretory", "Absorptive"),
    ]
    p_values = []
    for category, baseline in pairs:
        if category not in wide.columns or baseline not in wide.columns:
            p_values.append(np.nan)
            continue
        difference = (wide[category] - wide[baseline]).dropna()
        if len(difference) < 5:
            p_values.append(np.nan)
            continue
        # Wilcoxon signed-rank on paired sample medians.
        from scipy.stats import wilcoxon

        statistic, p_value = wilcoxon(
            difference.to_numpy(float), alternative="two-sided"
        )
        p_values.append(float(p_value))
        contrasts.append(
            {
                "contrast": f"{category} minus {baseline}",
                "category": category,
                "baseline": baseline,
                "n_paired_samples": int(len(difference)),
                "median_paired_difference": float(difference.median()),
                "mean_paired_difference": float(difference.mean()),
                "wilcoxon_statistic": float(statistic),
                "p_value": float(p_value),
            }
        )
    contrast_frame = pd.DataFrame(contrasts)
    if not contrast_frame.empty:
        mask = contrast_frame["p_value"].notna()
        rejected, q_values, *_ = multipletests(
            contrast_frame.loc[mask, "p_value"],
            method="fdr_bh",
        )
        contrast_frame.loc[mask, "q_value"] = q_values
        contrast_frame.loc[mask, "significant_fdr_0.05"] = rejected
    return sample, contrast_frame


def build_panel_d_identity_time_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """PSC sample × identity distance and fraction versus culture day."""
    hierarchy = pd.read_csv(
        C.DATA / "hgca_epithelial_hierarchy.csv"
    ).set_index("hgca_celltype_v1")
    metadata = pd.read_csv(C.DATA / "sample_metadata.csv")
    metadata["day"] = parse_numeric_culture_day(metadata["time"])
    psc_samples = metadata[
        metadata["source_standardized"].eq("PSC") & metadata["day"].notna()
    ].copy()
    if len(psc_samples) != 41:
        raise RuntimeError(
            f"Expected 41 numeric-time PSC samples for panel d, found {len(psc_samples)}"
        )
    cells = pd.read_csv(
        C.DATA / "per_cell_mapping_qc_flags.csv.gz",
        usecols=[
            "sample_id",
            "hgca_pred_celltype_sysvi_knn",
            "hgca_pred_celltype_sysvi_knn_thresh",
            "d_nn1",
            "confident",
        ],
    )
    cells = cells.merge(
        psc_samples[
            [
                "sample_id",
                "publication_display",
                "region_broad",
                "day",
            ]
        ],
        on="sample_id",
        how="inner",
        validate="many_to_one",
    )
    cells = cells[
        cells["confident"]
        & cells["hgca_pred_celltype_sysvi_knn_thresh"].ne("Unknown")
        & cells["d_nn1"].notna()
        & ~cross_compartment_mask(
            cells["hgca_pred_celltype_sysvi_knn"],
            cells["region_broad"],
        )
    ].copy()
    cells["branch"] = cells["hgca_pred_celltype_sysvi_knn"].map(
        lambda value: branch_category(value, hierarchy)
    )
    cells = cells[cells["branch"].isin(BRANCH_ORDER)].copy()
    sample_totals = (
        cells.groupby("sample_id", observed=True)
        .size()
        .rename("n_confident_epithelial")
    )
    identity_sample = (
        cells.groupby(
            [
                "sample_id",
                "publication_display",
                "day",
                "hgca_pred_celltype_sysvi_knn",
                "branch",
            ],
            observed=True,
        )
        .agg(
            n_cells=("d_nn1", "size"),
            median_nn_distance=("d_nn1", "median"),
        )
        .reset_index()
        .rename(columns={"hgca_pred_celltype_sysvi_knn": "hgca_celltype_v1"})
    )
    identity_sample = identity_sample.join(
        sample_totals, on="sample_id", how="left"
    )
    identity_sample["identity_fraction"] = (
        identity_sample["n_cells"]
        / identity_sample["n_confident_epithelial"]
    )
    identity_sample["log1p_day"] = np.log1p(identity_sample["day"])
    identity_sample["display"] = identity_sample["hgca_celltype_v1"].map(
        short_identity_label
    )
    # Keep identities with a stable per-sample median in enough samples.
    supported = (
        identity_sample[identity_sample["n_cells"] >= 20]
        .groupby("hgca_celltype_v1", observed=True)["sample_id"]
        .nunique()
    )
    supported_identities = supported[supported >= 8].index
    identity_sample = identity_sample[
        identity_sample["hgca_celltype_v1"].isin(supported_identities)
        & identity_sample["n_cells"].ge(20)
    ].copy()
    if identity_sample.empty:
        raise RuntimeError("No supported identities for panel d")

    # Lineage trends use one observation per sample × lineage to avoid
    # pseudo-replicating multiple identities within a sample.
    lineage_rows = []
    for (sample_id, branch), frame in identity_sample.groupby(
        ["sample_id", "branch"], observed=True
    ):
        lineage_rows.append(
            {
                "sample_id": sample_id,
                "publication_display": frame["publication_display"].iloc[0],
                "day": float(frame["day"].iloc[0]),
                "log1p_day": float(frame["log1p_day"].iloc[0]),
                "branch": branch,
                "median_nn_distance": float(frame["median_nn_distance"].mean()),
                "identity_fraction": float(frame["identity_fraction"].sum()),
                "n_identities": int(frame["hgca_celltype_v1"].nunique()),
                "n_cells": int(frame["n_cells"].sum()),
            }
        )
    lineage_sample = pd.DataFrame(lineage_rows)
    model_rows = []
    for outcome in ["median_nn_distance", "identity_fraction"]:
        for branch in BRANCH_ORDER:
            frame = lineage_sample[lineage_sample["branch"].eq(branch)].copy()
            if (
                len(frame) < 8
                or frame["log1p_day"].nunique() < 2
                or frame["publication_display"].nunique() < 2
            ):
                continue
            fit = smf.ols(
                f"{outcome} ~ log1p_day + publication_display",
                data=frame,
            ).fit(cov_type="HC3")
            confidence = fit.conf_int().loc["log1p_day"]
            model_rows.append(
                {
                    "outcome": outcome,
                    "branch": branch,
                    "n_samples": int(len(frame)),
                    "n_publications": int(
                        frame["publication_display"].nunique()
                    ),
                    "coefficient": float(fit.params["log1p_day"]),
                    "standard_error": float(fit.bse["log1p_day"]),
                    "ci_low": float(confidence.iloc[0]),
                    "ci_high": float(confidence.iloc[1]),
                    "p_value": float(fit.pvalues["log1p_day"]),
                    "formula": (
                        f"{outcome} ~ log1p(day) + C(publication); "
                        "sample×lineage aggregates"
                    ),
                }
            )
    return identity_sample, pd.DataFrame(model_rows)


def draw_panel_d_identity_time(
    axis: plt.Axes,
    identity_sample: pd.DataFrame,
    lineage_models: pd.DataFrame,
    outcome: str,
    ylabel: str,
    show_legend: bool,
) -> None:
    tick_days = np.array([0, 3, 7, 14, 28, 56, 98], dtype=float)
    tick_positions = np.log1p(tick_days)
    grid = np.linspace(
        identity_sample["log1p_day"].min(),
        identity_sample["log1p_day"].max(),
        200,
    )
    for branch in BRANCH_ORDER:
        subset = identity_sample[identity_sample["branch"].eq(branch)]
        if subset.empty:
            continue
        axis.scatter(
            subset["log1p_day"],
            subset[outcome],
            s=7,
            color=BRANCH_COLORS[branch],
            edgecolor="white",
            linewidth=0.15,
            alpha=0.55,
            label=branch,
            zorder=3,
        )
        model = lineage_models[
            lineage_models["outcome"].eq(outcome)
            & lineage_models["branch"].eq(branch)
        ]
        if model.empty:
            continue
        # Refit for publication-averaged prediction band on sample×lineage.
        grouped = subset.groupby(
            ["sample_id", "publication_display", "log1p_day"],
            observed=True,
        )
        if outcome == "median_nn_distance":
            lineage = grouped["median_nn_distance"].mean().reset_index()
        else:
            lineage = grouped["identity_fraction"].sum().reset_index()
        fit = smf.ols(
            f"{outcome} ~ log1p_day + publication_display",
            data=lineage,
        ).fit(cov_type="HC3")
        parameter_names = list(fit.params.index)
        publication_terms = [
            name
            for name in parameter_names
            if name.startswith("publication_display[T.")
        ]
        n_publications = lineage["publication_display"].nunique()
        means = []
        lowers = []
        uppers = []
        covariance = fit.cov_params().to_numpy()
        for value in grid:
            contrast = np.zeros(len(parameter_names), dtype=float)
            contrast[parameter_names.index("Intercept")] = 1.0
            contrast[parameter_names.index("log1p_day")] = value
            for name in publication_terms:
                contrast[parameter_names.index(name)] = 1 / n_publications
            mean = float(contrast @ fit.params.to_numpy())
            se = float(np.sqrt(contrast @ covariance @ contrast))
            means.append(mean)
            lowers.append(mean - 1.96 * se)
            uppers.append(mean + 1.96 * se)
        axis.plot(
            grid, means, color=BRANCH_COLORS[branch], lw=0.95, zorder=4
        )
        axis.fill_between(
            grid,
            lowers,
            uppers,
            color=BRANCH_COLORS[branch],
            alpha=0.12,
            linewidth=0,
        )
    axis.set_xticks(tick_positions, [str(int(day)) for day in tick_days])
    axis.set_xlabel("Numeric maturation day (log1p scale)")
    axis.set_ylabel(ylabel)
    # Keep CI ribbons from dominating the scale when a lineage is sparse.
    y_values = identity_sample[outcome].to_numpy(float)
    low = float(np.nanquantile(y_values, 0.01))
    high = float(np.nanquantile(y_values, 0.99))
    pad = 0.08 * (high - low if high > low else 1.0)
    axis.set_ylim(low - pad, high + pad)
    if show_legend:
        axis.legend(
            frameon=False,
            loc="best",
            fontsize=3.8,
            title="Lineage",
            title_fontsize=4.0,
        )
    sns.despine(ax=axis)


TIME_BIN_ORDER = ["≤14 d", "15–55 d", "≥56 d"]
SEGMENT_ORDER = ["Small Intestine", "Colon"]
MOSAIC_MIN_FRACTION = 0.015
# HCA blue (close / low distance) → white (far from adult reference).
HCA_BLUE = "#3A68AE"
MOSAIC_DISTANCE_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "hgca_distance_near_far",
    [HCA_BLUE, "#5B86C1", "#8EAFD6", "#C5D8EB", "#F7FBFF", "#FFFFFF"],
)
OTHER_CELL_FACE = "#D0D0D0"
MOSAIC_SHORT_LABELS = {
    "Intestinal Stem Cells (ISC)": "ISC",
    "Transiently Amplifying Cells (TA)": "TA",
    "EEC Progenitors": "EEC prog.",
    "Tuft Progenitors": "Tuft prog.",
    "Enterocyte Progenitors": "Ent. prog.",
    "Colonocyte Progenitors": "Col. prog.",
    "Secretory Progenitors": "Sec. prog.",
    "BEST4 Colonocytes": "BEST4 col.",
    "BEST4 Enterocytes": "BEST4 ent.",
    "Crypt Top Colonocytes": "Crypt-top",
    "Lower Crypt Colonocytes": "Lower-crypt",
    "Mid Crypt Colonocytes": "Mid-crypt",
    "Lower Villus Enterocytes": "Lower villus",
    "Mid Villus Enterocytes": "Mid villus",
    "Villus Tip Enterocytes": "Villus tip",
    "Goblet Cells": "Goblet",
    "Mature Goblet Cells": "Mature goblet",
    "Tuft Cells": "Tuft",
    "EEC Enterochromaffin (EC)": "EEC (EC)",
    "EEC L": "EEC L",
    "EEC I": "EEC I",
    "EEC K": "EEC K",
    "EEC N": "EEC N",
    "EEC S": "EEC S",
    "Microfold Cells (M cells)": "M cells",
    "Follicle associated enterocyte (FAE)": "FAE",
    "Brunners Gland Cells": "Brunner's",
    "Paneth Cells": "Paneth",
}


def mosaic_short_label(identity: str) -> str:
    if str(identity).startswith("Other"):
        return "Other"
    return MOSAIC_SHORT_LABELS.get(
        str(identity), short_identity_label(str(identity))
    )


def declared_segment(region_broad: str) -> str:
    """Map declared tissue to analysis segment.

    Nonspecific HEOCA ``Intestine`` labels are mid/hindgut-patterned HIOs and
    are treated as Colon, not small intestine.
    """
    text = str(region_broad).strip()
    if text in {"Small Intestine", "Duodenum", "Jejunum", "Ileum"}:
        return "Small Intestine"
    if text in {"Colon", "Intestine"}:
        return "Colon"
    return "Other"


def sequential_distance_norm(
    distances: np.ndarray,
) -> matplotlib.colors.Normalize:
    """Nature-style sequential scale from 0 to a nice max covering the data."""
    finite = np.asarray(distances, dtype=float)
    finite = finite[np.isfinite(finite)]
    vmax = float(np.max(finite)) if len(finite) else 1.0
    vmax = max(0.5, float(np.ceil(vmax * 2.0) / 2.0))
    return matplotlib.colors.Normalize(vmin=0.0, vmax=vmax)


def _mosaic_bin_rows(
    samples: pd.Series,
    fraction_long: pd.DataFrame,
    sample_distance: pd.DataFrame,
    identity_order: list[str],
    time_class: str,
    segment: str,
) -> pd.DataFrame:
    n_samples = int(samples.nunique())
    if n_samples == 0:
        return pd.DataFrame()
    bin_fractions = fraction_long[fraction_long["sample_id"].isin(samples)]
    mean_fraction = (
        bin_fractions.groupby(["hgca_celltype_v1", "branch"], observed=True)[
            "identity_fraction"
        ]
        .mean()
        .reset_index()
    )
    bin_distance = sample_distance[sample_distance["sample_id"].isin(samples)]
    median_distance = (
        bin_distance.groupby("hgca_celltype_v1", observed=True)[
            "median_nn_distance"
        ]
        .median()
        .rename("median_nn_distance")
    )
    n_distance_samples = (
        bin_distance.groupby("hgca_celltype_v1", observed=True)["sample_id"]
        .nunique()
        .rename("n_samples_with_identity")
    )
    mean_fraction = mean_fraction.join(
        median_distance, on="hgca_celltype_v1"
    ).join(n_distance_samples, on="hgca_celltype_v1")
    mean_fraction["n_samples_with_identity"] = (
        mean_fraction["n_samples_with_identity"].fillna(0).astype(int)
    )
    keep = mean_fraction["identity_fraction"].ge(MOSAIC_MIN_FRACTION)
    kept = mean_fraction.loc[keep].copy()
    for branch in BRANCH_ORDER:
        residual = float(
            mean_fraction.loc[
                mean_fraction["branch"].eq(branch) & ~keep,
                "identity_fraction",
            ].sum()
        )
        if residual >= MOSAIC_MIN_FRACTION / 2:
            kept = pd.concat(
                [
                    kept,
                    pd.DataFrame(
                        [
                            {
                                "hgca_celltype_v1": f"Other {branch}",
                                "branch": branch,
                                "identity_fraction": residual,
                                "median_nn_distance": np.nan,
                                "n_samples_with_identity": 0,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    if kept.empty:
        return pd.DataFrame()
    kept["segment"] = segment
    kept["time_class"] = time_class
    kept["n_samples_in_bin"] = n_samples
    kept["display"] = kept["hgca_celltype_v1"].map(mosaic_short_label)
    order_map = {identity: index for index, identity in enumerate(identity_order)}
    kept["taxonomy_rank"] = kept["hgca_celltype_v1"].map(
        lambda label: order_map.get(label, 10_000)
    )
    kept["branch_rank"] = kept["branch"].map(
        {name: index for index, name in enumerate(BRANCH_ORDER)}
    )
    return kept.sort_values(
        ["branch_rank", "taxonomy_rank", "identity_fraction"],
        ascending=[True, True, False],
    )


def build_psc_composition_mosaic_table() -> pd.DataFrame:
    """Mean PSC subtype fractions and adult-reference distance by time × segment."""
    metadata = pd.read_csv(C.DATA / "sample_metadata.csv")
    proportions = pd.read_csv(
        C.DATA / "sample_subtype_proportions_confident.csv"
    )
    hierarchy = pd.read_csv(
        C.DATA / "hgca_epithelial_hierarchy.csv", index_col=0
    )
    qc = pd.read_csv(
        C.DATA / "per_cell_mapping_qc_flags.csv.gz",
        usecols=[
            "sample_id",
            "hgca_pred_celltype_sysvi_knn_thresh",
            "confident",
            "d_nn1",
        ],
    )
    psc = metadata[
        metadata["source_standardized"].eq("PSC")
        & metadata["time_class"].isin(TIME_BIN_ORDER)
        & parse_numeric_culture_day(metadata["time"]).notna()
    ][
        [
            "sample_id",
            "publication_display",
            "time_class",
            "time",
            "region_broad",
        ]
    ].copy()
    psc["day"] = parse_numeric_culture_day(psc["time"])
    psc["segment"] = psc["region_broad"].map(declared_segment)
    psc = psc[psc["segment"].isin(SEGMENT_ORDER)].copy()
    if psc["sample_id"].nunique() < 10:
        raise RuntimeError("Too few numeric PSC samples for composition mosaic")

    subtype_columns = [
        column
        for column in proportions.columns
        if column not in {"sample_id", "Unknown"}
    ]
    fraction_long = proportions.melt(
        id_vars=["sample_id"],
        value_vars=subtype_columns,
        var_name="hgca_celltype_v1",
        value_name="identity_fraction",
    )
    fraction_long = fraction_long.merge(psc, on="sample_id", how="inner")
    fraction_long["branch"] = fraction_long["hgca_celltype_v1"].map(
        lambda label: branch_category(label, hierarchy)
    )
    fraction_long = fraction_long[
        fraction_long["branch"].isin(BRANCH_ORDER)
    ].copy()

    distance_cells = qc.merge(
        psc[["sample_id", "region_broad"]], on="sample_id", how="inner"
    )
    distance_cells = distance_cells[
        distance_cells["confident"]
        & distance_cells["hgca_pred_celltype_sysvi_knn_thresh"].ne("Unknown")
        & distance_cells["d_nn1"].notna()
        & ~cross_compartment_mask(
            distance_cells["hgca_pred_celltype_sysvi_knn_thresh"],
            distance_cells["region_broad"],
        )
    ].copy()
    sample_distance = (
        distance_cells.groupby(
            ["sample_id", "hgca_pred_celltype_sysvi_knn_thresh"],
            observed=True,
        )
        .agg(
            n_cells=("d_nn1", "size"),
            median_nn_distance=("d_nn1", "median"),
        )
        .reset_index()
        .rename(
            columns={
                "hgca_pred_celltype_sysvi_knn_thresh": "hgca_celltype_v1"
            }
        )
    )
    sample_distance = sample_distance[sample_distance["n_cells"] >= 10]

    identity_order = taxonomy_order(hierarchy)
    rows = []
    # Pooled (all segments) plus Colon / Small Intestine strata.
    strata = [("All", psc)] + [
        (segment, psc[psc["segment"].eq(segment)]) for segment in SEGMENT_ORDER
    ]
    for segment_label, segment_frame in strata:
        for time_class in TIME_BIN_ORDER:
            samples = segment_frame.loc[
                segment_frame["time_class"].eq(time_class), "sample_id"
            ]
            kept = _mosaic_bin_rows(
                samples,
                fraction_long,
                sample_distance,
                identity_order,
                time_class=time_class,
                segment=segment_label,
            )
            if kept.empty:
                rows.append(
                    pd.DataFrame(
                        [
                            {
                                "segment": segment_label,
                                "time_class": time_class,
                                "n_samples_in_bin": int(samples.nunique()),
                                "branch": "Progenitor",
                                "branch_fraction": np.nan,
                                "hgca_celltype_v1": "No samples",
                                "display": "No samples",
                                "identity_fraction": np.nan,
                                "median_nn_distance": np.nan,
                                "n_samples_with_identity": 0,
                                "branch_rank": 0,
                                "taxonomy_rank": 0,
                            }
                        ]
                    )
                )
            else:
                rows.append(kept)
    mosaic = pd.concat(rows, ignore_index=True)
    present = mosaic["identity_fraction"].notna()
    mosaic.loc[present, "branch_fraction"] = mosaic.loc[present].groupby(
        ["segment", "time_class", "branch"], observed=True
    )["identity_fraction"].transform("sum")
    return mosaic[
        [
            "segment",
            "time_class",
            "n_samples_in_bin",
            "branch",
            "branch_fraction",
            "hgca_celltype_v1",
            "display",
            "identity_fraction",
            "median_nn_distance",
            "n_samples_with_identity",
            "branch_rank",
            "taxonomy_rank",
        ]
    ]


def draw_nested_composition_mosaic(
    axis: plt.Axes,
    mosaic_bin: pd.DataFrame,
    norm: matplotlib.colors.Normalize,
    cmap: matplotlib.colors.Colormap,
    title: str,
    show_ylabel: bool,
    label_fontsize: float = 5.0,
) -> None:
    """Ordered nested mosaic: branch strips → subtype tiles; fill = distance."""
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

    usable = mosaic_bin[mosaic_bin["identity_fraction"].notna()].copy()
    total = float(usable["identity_fraction"].sum()) if len(usable) else 0.0
    if total <= 0:
        n_samples = int(mosaic_bin["n_samples_in_bin"].iloc[0]) if len(mosaic_bin) else 0
        axis.set_facecolor("#F4F4F4")
        axis.text(
            0.5,
            0.5,
            f"No samples\n(n={n_samples})",
            ha="center",
            va="center",
            fontsize=6,
            color="#666666",
        )
        axis.set_title(title, fontsize=6.5, fontweight="bold", pad=4)
        if show_ylabel:
            axis.set_ylabel("Composition\n(area = mean fraction)", fontsize=6)
        return

    scale = 1.0 / total
    y_cursor = 1.0
    text_stroke = [
        patheffects.withStroke(linewidth=1.6, foreground="white")
    ]
    for branch in BRANCH_ORDER:
        branch_rows = usable[usable["branch"].eq(branch)]
        if branch_rows.empty:
            continue
        branch_height = float(branch_rows["identity_fraction"].sum()) * scale
        if branch_height <= 0:
            continue
        y0 = y_cursor - branch_height
        rail = 0.05
        # Left lineage rail (fixed colour; fill is distance).
        axis.add_patch(
            matplotlib.patches.Rectangle(
                (0.0, y0),
                rail,
                branch_height,
                linewidth=0,
                facecolor=BRANCH_COLORS[branch],
                zorder=6,
            )
        )
        branch_total = float(branch_rows["identity_fraction"].sum())
        if branch_height >= 0.08:
            axis.text(
                rail / 2,
                y0 + 0.5 * branch_height,
                f"{100 * branch_total * scale:.0f}%",
                ha="center",
                va="center",
                fontsize=max(3.8, label_fontsize - 0.4),
                fontweight="bold",
                color="white",
                zorder=8,
                rotation=90,
            )
        x_cursor = rail
        plot_width = 1.0 - rail
        for _, row in branch_rows.iterrows():
            width = float(row["identity_fraction"]) * scale * plot_width
            if width <= 0:
                continue
            distance = row["median_nn_distance"]
            is_other = str(row["display"]).startswith("Other") or pd.isna(
                distance
            )
            face = OTHER_CELL_FACE if is_other else cmap(norm(float(distance)))
            axis.add_patch(
                matplotlib.patches.Rectangle(
                    (x_cursor, y0),
                    width,
                    branch_height,
                    linewidth=0.45,
                    edgecolor="white",
                    facecolor=face,
                    hatch="///" if is_other else None,
                    zorder=3,
                )
            )
            # Label every visible tile.
            if width >= 0.04 and branch_height >= 0.045:
                far = is_other or (
                    not pd.isna(distance)
                    and float(distance) > 0.45 * float(norm.vmax)
                )
                text_color = "#222222" if far else "white"
                axis.text(
                    x_cursor + width / 2,
                    y0 + branch_height / 2,
                    mosaic_short_label(str(row["hgca_celltype_v1"])),
                    ha="center",
                    va="center",
                    fontsize=max(
                        3.2, label_fontsize - (0.0 if width > 0.1 else 0.6)
                    ),
                    color=text_color,
                    zorder=7,
                    clip_on=True,
                    path_effects=text_stroke,
                )
            x_cursor += width
        axis.add_patch(
            matplotlib.patches.Rectangle(
                (rail, y0),
                plot_width,
                branch_height,
                linewidth=1.0,
                edgecolor=BRANCH_COLORS[branch],
                facecolor="none",
                zorder=5,
            )
        )
        y_cursor = y0

    axis.set_title(title, fontsize=6.5, fontweight="bold", pad=4)
    if show_ylabel:
        axis.set_ylabel("Composition\n(area = mean fraction)", fontsize=6)


def _add_mosaic_colorbar(figure, cax, norm, cmap) -> None:
    colorbar = figure.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
    )
    colorbar.set_label(
        "Median distance to adult HCA\n(blue = close, white = far)",
        fontsize=5.5,
    )
    vmax = float(norm.vmax)
    if abs(vmax - round(vmax)) < 1e-8:
        ticks = np.arange(0.0, vmax + 0.1, 1.0)
    else:
        ticks = np.linspace(0.0, vmax, num=5)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{tick:g}" for tick in ticks])
    colorbar.ax.tick_params(labelsize=5)
    colorbar.outline.set_linewidth(0.6)


def draw_panel_d_composition_mosaic(
    figure: plt.Figure,
    gridspec_slot,
    mosaic: pd.DataFrame,
    segments: list[str] | None = None,
    panel_tag: str = "d",
) -> None:
    """Draw mosaics; default = Small Intestine + Colon rows (stratified)."""
    if segments is None:
        segments = list(SEGMENT_ORDER)
    plot = mosaic[mosaic["segment"].isin(segments)].copy()
    if plot.empty:
        raise RuntimeError("No mosaic rows for requested segments")
    n_rows = len(segments)
    panel = gridspec_slot.subgridspec(
        n_rows,
        4,
        width_ratios=[1, 1, 1, 0.055],
        wspace=0.07,
        hspace=0.28,
    )
    distances = plot["median_nn_distance"].dropna().to_numpy(float)
    if len(distances) == 0:
        raise RuntimeError("No distances available for composition mosaic")
    norm = sequential_distance_norm(distances)
    cmap = MOSAIC_DISTANCE_CMAP
    first_axis = None
    for row_index, segment in enumerate(segments):
        for col_index, time_class in enumerate(TIME_BIN_ORDER):
            axis = figure.add_subplot(panel[row_index, col_index])
            bin_frame = plot[
                plot["segment"].eq(segment)
                & plot["time_class"].eq(time_class)
            ].copy()
            n_samples = (
                int(bin_frame["n_samples_in_bin"].iloc[0])
                if len(bin_frame)
                else 0
            )
            title = time_class if row_index == 0 else ""
            if row_index == 0:
                title = f"{time_class} (n={n_samples})"
            else:
                title = f"n={n_samples}"
            draw_nested_composition_mosaic(
                axis,
                bin_frame,
                norm=norm,
                cmap=cmap,
                title=title,
                show_ylabel=col_index == 0,
                label_fontsize=4.8 if n_rows == 1 else 4.4,
            )
            if col_index == 0:
                axis.text(
                    -0.22,
                    0.5,
                    segment.replace(" ", "\n"),
                    transform=axis.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=6,
                    fontweight="bold",
                )
            if first_axis is None:
                first_axis = axis
        # Shared colourbar column spans rows via last column of first row only
        # when single row; for multi-row attach once.
    cax = figure.add_subplot(panel[:, 3])
    _add_mosaic_colorbar(figure, cax, norm, cmap)
    if first_axis is not None:
        panel_label(first_axis, panel_tag)
    figure.legend(
        handles=[
            matplotlib.patches.Patch(
                facecolor=BRANCH_COLORS[branch],
                edgecolor="white",
                linewidth=0.3,
                label=branch,
            )
            for branch in BRANCH_ORDER
        ],
        loc="lower center",
        bbox_to_anchor=(0.62, 0.01),
        ncol=3,
        frameon=False,
        fontsize=5,
        title="Lineage (left rail)",
        title_fontsize=5.2,
        handlelength=0.9,
    )


def render_psc_composition_mosaic_standalone(
    mosaic: pd.DataFrame,
    segments: list[str] | None = None,
    stem: str = "fig5_psc_composition_mosaic",
    title: str | None = None,
) -> None:
    if segments is None:
        segments = ["All"]
    plot = mosaic[mosaic["segment"].isin(segments)].copy()
    n_rows = len(segments)
    height = 58 * MM if n_rows == 1 else 52 * MM * n_rows + 12 * MM
    figure = plt.figure(figsize=(180 * MM, height))
    grid = figure.add_gridspec(
        n_rows,
        4,
        width_ratios=[1, 1, 1, 0.05],
        wspace=0.08,
        hspace=0.32,
    )
    distances = plot["median_nn_distance"].dropna().to_numpy(float)
    norm = sequential_distance_norm(distances)
    cmap = MOSAIC_DISTANCE_CMAP
    for row_index, segment in enumerate(segments):
        for col_index, time_class in enumerate(TIME_BIN_ORDER):
            axis = figure.add_subplot(grid[row_index, col_index])
            bin_frame = plot[
                plot["segment"].eq(segment)
                & plot["time_class"].eq(time_class)
            ].copy()
            n_samples = (
                int(bin_frame["n_samples_in_bin"].iloc[0])
                if len(bin_frame)
                else 0
            )
            if n_rows == 1:
                panel_title = f"{time_class} (n={n_samples})"
            else:
                panel_title = (
                    f"{time_class} (n={n_samples})"
                    if row_index == 0
                    else f"n={n_samples}"
                )
            draw_nested_composition_mosaic(
                axis,
                bin_frame,
                norm=norm,
                cmap=cmap,
                title=panel_title,
                show_ylabel=col_index == 0,
                label_fontsize=5.2 if n_rows == 1 else 4.8,
            )
            if n_rows > 1 and col_index == 0:
                axis.text(
                    -0.2,
                    0.5,
                    segment.replace(" ", "\n"),
                    transform=axis.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                )
    cax = figure.add_subplot(grid[:, 3])
    _add_mosaic_colorbar(figure, cax, norm, cmap)
    figure.suptitle(
        title
        or (
            "PSC epithelial composition across maturation"
            if segments == ["All"]
            else "PSC composition by declared segment and maturation"
        ),
        fontsize=8,
        fontweight="bold",
        y=0.98,
    )
    figure.subplots_adjust(
        left=0.1 if n_rows > 1 else 0.08,
        right=0.93,
        top=0.88 if n_rows == 1 else 0.90,
        bottom=0.12,
    )
    for extension in ("pdf", "svg", "png"):
        figure.savefig(
            C.OUT / f"{stem}.{extension}",
            dpi=300,
            facecolor="white",
        )
    plt.close(figure)


def _densify_ring(coords, max_step: float = 0.004) -> list[tuple[float, float]]:
    """Insert vertices so curved Voronoi edges render smoothly in raster output."""
    if len(coords) < 2:
        return list(coords)
    densified: list[tuple[float, float]] = [tuple(coords[0])]
    for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
        dist = float(np.hypot(x1 - x0, y1 - y0))
        n_step = max(1, int(np.ceil(dist / max_step)))
        for step in range(1, n_step + 1):
            t = step / n_step
            densified.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return densified


def _polygon_to_pathpatch(polygon, **kwargs) -> PathPatch | None:
    if polygon is None or polygon.is_empty:
        return None
    if polygon.geom_type == "MultiPolygon":
        # Draw the largest piece; caller should union when needed.
        polygon = max(polygon.geoms, key=lambda geom: geom.area)
    if polygon.geom_type != "Polygon" or polygon.area <= 0:
        return None
    vertices = _densify_ring(list(polygon.exterior.coords))
    codes = (
        [MplPath.MOVETO]
        + [MplPath.LINETO] * (len(vertices) - 2)
        + [MplPath.CLOSEPOLY]
    )
    kwargs.setdefault("joinstyle", "round")
    kwargs.setdefault("capstyle", "round")
    patch = PathPatch(MplPath(vertices, codes), **kwargs)
    patch.set_antialiased(True)
    return patch


def _largest_polygon(geom) -> Polygon | None:
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda piece: piece.area)
    polys = [
        piece
        for piece in getattr(geom, "geoms", [])
        if piece.geom_type in {"Polygon", "MultiPolygon"}
    ]
    if not polys:
        return None
    return _largest_polygon(unary_union(polys))


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {
        key: max(float(value), 0.0)
        for key, value in weights.items()
        if float(value) > 0
    }
    total = sum(cleaned.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in cleaned.items()}


def _seed_sites_in_polygon(
    container: Polygon,
    target_areas: dict[str, float],
    rng: np.random.Generator,
    positioning: str,
    level_order: list[str] | None = None,
) -> dict[str, tuple[float, float]]:
    """Seed one site per cell; mirrors WeightedTreemaps positioning modes."""
    labels = list(target_areas)
    if not labels:
        return {}
    cx, cy = float(container.centroid.x), float(container.centroid.y)
    minx, miny, maxx, maxy = container.bounds
    radius = 0.45 * min(maxx - minx, maxy - miny)

    def _clamp_inside(x: float, y: float) -> tuple[float, float]:
        point = Point(x, y)
        if container.contains(point):
            return x, y
        # Pull toward centroid until inside.
        for alpha in np.linspace(0.9, 0.05, 18):
            trial = Point(cx + alpha * (x - cx), cy + alpha * (y - cy))
            if container.contains(trial):
                return float(trial.x), float(trial.y)
        rep = container.representative_point()
        return float(rep.x), float(rep.y)

    sites: dict[str, tuple[float, float]] = {}
    if positioning == "sector" and level_order is not None:
        ordered = [label for label in level_order if label in target_areas]
        angle = -np.pi / 2
        for label in ordered:
            sweep = 2 * np.pi * target_areas[label]
            mid = angle + 0.5 * sweep
            # Keep sites inward so large cells stay compact wedges.
            radial = radius * (0.28 + 0.18 * (1.0 - target_areas[label]))
            sites[label] = _clamp_inside(
                cx + radial * np.cos(mid), cy + radial * np.sin(mid)
            )
            angle += sweep
        return sites

    if positioning == "clustered_by_area":
        ordered = sorted(labels, key=lambda key: target_areas[key], reverse=True)
        golden = np.pi * (3 - np.sqrt(5))
        for index, label in enumerate(ordered):
            # Large targets near centre; small ones toward the rim.
            t = index / max(len(ordered) - 1, 1)
            radial = radius * (0.10 + 0.72 * (t**0.85))
            theta = index * golden + float(rng.uniform(-0.08, 0.08))
            sites[label] = _clamp_inside(
                cx + radial * np.cos(theta), cy + radial * np.sin(theta)
            )
        return sites

    # Random fallback.
    for label in labels:
        for _ in range(200):
            x = float(rng.uniform(minx, maxx))
            y = float(rng.uniform(miny, maxy))
            if container.contains(Point(x, y)):
                sites[label] = (x, y)
                break
        else:
            rep = container.representative_point()
            sites[label] = (float(rep.x), float(rep.y))
    return sites


def _aw_voronoi_from_grid(
    container: Polygon,
    sites: dict[str, tuple[float, float]],
    additive_weights: dict[str, float],
    grid_n: int = 240,
) -> dict[str, object]:
    """Gap-free additively weighted Voronoi via dense grid assignment.

    Cell of site i: {p : ||p - s_i|| - w_i <= ||p - s_j|| - w_j}.
    This is the WeightedTreemaps / CGAL AWVT principle, implemented without CGAL.
    """
    labels = [label for label in sites if label in additive_weights]
    if not labels:
        return {}
    if len(labels) == 1:
        return {labels[0]: container}

    minx, miny, maxx, maxy = container.bounds
    xs = np.linspace(minx, maxx, grid_n)
    ys = np.linspace(miny, maxy, grid_n)
    xx, yy = np.meshgrid(xs, ys)
    inside = shapely_vectorized.contains(container, xx, yy)
    scores = []
    for label in labels:
        sx, sy = sites[label]
        scores.append(np.hypot(xx - sx, yy - sy) - float(additive_weights[label]))
    winner = np.argmin(np.stack(scores, axis=0), axis=0)
    winner = np.where(inside, winner, -1)

    dx = (maxx - minx) / max(grid_n - 1, 1)
    dy = (maxy - miny) / max(grid_n - 1, 1)
    regions: dict[str, object] = {}
    for index, label in enumerate(labels):
        mask = winner == index
        if not mask.any():
            continue
        padded = np.pad(mask.astype(float), 1, mode="constant", constant_values=0.0)
        contours = sk_measure.find_contours(padded, 0.5)
        pieces = []
        for contour in contours:
            coords = [
                (minx + (col - 1.0) * dx, miny + (row - 1.0) * dy)
                for row, col in contour
            ]
            if len(coords) < 4:
                continue
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            poly = poly.intersection(container)
            if poly.is_empty:
                continue
            # Light topology cleanup only — avoid aggressive simplify (jagged look).
            if not poly.is_valid:
                poly = poly.buffer(0)
            # Morphological smooth + light densify for print-quality curves.
            smooth = min(dx, dy) * 0.9
            poly = poly.buffer(smooth).buffer(-smooth)
            poly = poly.intersection(container)
            if poly.is_empty:
                continue
            piece = _largest_polygon(poly)
            if piece is None or piece.is_empty:
                continue
            pieces.append(piece)
        if not pieces:
            continue
        merged = unary_union(pieces)
        if merged.is_empty:
            continue
        # Prefer a single compact cell; keep multipolygon only if needed.
        largest = _largest_polygon(merged)
        regions[label] = largest if largest is not None else merged
    return regions


def weighted_voronoi_regions(
    container: Polygon,
    weights: dict[str, float],
    rng: np.random.Generator,
    positioning: str = "clustered_by_area",
    level_order: list[str] | None = None,
    max_iter: int = 70,
    error_tol: float = 0.025,
    grid_n: int = 260,
    convergence: str = "intermediate",
) -> dict[str, object]:
    """Area-weighted Voronoi treemap cells (WeightedTreemaps-style AWVT).

    Principles used here:
    - one site per category (avoids sprawling multi-lobed merges)
    - additively weighted distance controls area
    - iterative weight updates until area error is small
    - gentle centroid chasing keeps cells compact
    - hierarchical callers nest children inside parent polygons
    """
    if container is None or container.is_empty or container.area <= 0:
        return {}
    target = _normalize_weights(weights)
    if not target:
        return {}
    if len(target) == 1:
        label = next(iter(target))
        return {label: container}

    sites = _seed_sites_in_polygon(
        container, target, rng, positioning=positioning, level_order=level_order
    )
    # Larger targets start with larger additive weights.
    scale = float(np.sqrt(container.area))
    add_w = {
        label: 0.20 * scale * np.sqrt(area) for label, area in target.items()
    }
    step = {"slow": 0.20, "intermediate": 0.35, "fast": 0.55}.get(
        convergence, 0.35
    )
    # Coarse grid while converging; high-resolution final extract for clean edges.
    iterate_grid = int(min(360, max(180, grid_n // 2)))
    regions: dict[str, object] = {}
    for _ in range(max_iter):
        regions = _aw_voronoi_from_grid(
            container, sites, add_w, grid_n=iterate_grid
        )
        max_abs_error = 0.0
        for label, area_target in target.items():
            geom = regions.get(label)
            got = (
                0.0
                if geom is None or geom.is_empty
                else geom.area / container.area
            )
            error = area_target - got
            max_abs_error = max(max_abs_error, abs(error))
            add_w[label] += step * error * scale
            add_w[label] = float(
                np.clip(add_w[label], -0.35 * scale, 0.85 * scale)
            )
            if geom is not None and not geom.is_empty:
                piece = _largest_polygon(geom)
                if piece is not None:
                    centre = piece.centroid
                    if container.contains(centre):
                        sx, sy = sites[label]
                        sites[label] = (
                            0.75 * sx + 0.25 * float(centre.x),
                            0.75 * sy + 0.25 * float(centre.y),
                        )
        if max_abs_error <= error_tol and len(regions) == len(target):
            break

    missing = [label for label in target if label not in regions]
    if missing:
        for label in missing:
            add_w[label] = max(add_w[label], 0.05 * scale)
    regions = _aw_voronoi_from_grid(container, sites, add_w, grid_n=grid_n)
    return regions


def hierarchical_voronoi_layout(
    mosaic_bin: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[dict[str, object], dict[str, object]]:
    """Nested AWVT: lineage sectors, then compact subtype cells inside."""
    usable = mosaic_bin[mosaic_bin["identity_fraction"].notna()].copy()
    if usable.empty:
        return {}, {}
    total = float(usable["identity_fraction"].sum())
    if total <= 0:
        return {}, {}
    container = Point(0.5, 0.5).buffer(0.49, resolution=128)
    branch_weights = {
        branch: float(
            usable.loc[usable["branch"].eq(branch), "identity_fraction"].sum()
        )
        for branch in BRANCH_ORDER
    }
    branch_regions = weighted_voronoi_regions(
        container,
        branch_weights,
        rng=rng,
        positioning="sector",
        level_order=BRANCH_ORDER,
        max_iter=80,
        error_tol=0.02,
        grid_n=1200,
        convergence="intermediate",
    )
    identity_regions: dict[str, object] = {}
    cleaned_branches: dict[str, object] = {}
    for branch in BRANCH_ORDER:
        branch_poly = branch_regions.get(branch)
        branch_poly = _largest_polygon(branch_poly)
        if branch_poly is None:
            continue
        cleaned_branches[branch] = branch_poly
        branch_rows = usable[usable["branch"].eq(branch)]
        if branch_rows.empty:
            continue
        child_weights = {
            row["hgca_celltype_v1"]: float(row["identity_fraction"])
            for _, row in branch_rows.iterrows()
            if float(row["identity_fraction"]) > 0
        }
        children = weighted_voronoi_regions(
            branch_poly,
            child_weights,
            rng=rng,
            positioning="clustered_by_area",
            max_iter=65,
            error_tol=0.03,
            grid_n=1000,
            convergence="intermediate",
        )
        identity_regions.update(children)
    return identity_regions, cleaned_branches


def _clamp_to_dish(
    x: float, y: float, margin: float = 0.07
) -> tuple[float, float]:
    """Keep label anchors inside the dish, away from the rim."""
    cx, cy = DISH_CENTER
    dx, dy = x - cx, y - cy
    dist = float(np.hypot(dx, dy))
    max_r = DISH_RADIUS - margin
    if dist <= max_r or dist < 1e-12:
        return float(x), float(y)
    scale = max_r / dist
    return cx + dx * scale, cy + dy * scale


def _clamp_text_box_to_dish(
    x: float,
    y: float,
    half_width: float,
    half_height: float = 0.022,
    margin: float = 0.02,
) -> tuple[float, float]:
    """Pull a label so its full text box stays inside the dish circle."""
    cx, cy = DISH_CENTER
    max_r = DISH_RADIUS - margin
    x, y = float(x), float(y)
    for _ in range(24):
        corners = [
            (x - half_width, y - half_height),
            (x - half_width, y + half_height),
            (x + half_width, y - half_height),
            (x + half_width, y + half_height),
        ]
        overflow = max(float(np.hypot(px - cx, py - cy)) - max_r for px, py in corners)
        if overflow <= 0:
            return x, y
        # Move toward dish centre enough to clear the worst corner.
        x = cx + (x - cx) * (max_r / (max_r + overflow + 1e-6))
        y = cy + (y - cy) * (max_r / (max_r + overflow + 1e-6))
    return _clamp_to_dish(x, y, margin=margin + half_width)


def _interior_anchor(polygon: Polygon, shrink: float = 0.03) -> Point:
    """Prefer the visual centre (pole of inaccessibility) inside a region."""
    if polygon is None or polygon.is_empty:
        return Point(*DISH_CENTER)
    candidate = polygon
    inset = min(shrink, 0.20 * float(np.sqrt(max(polygon.area, 1e-8))))
    shrunk = polygon.buffer(-inset)
    if not shrunk.is_empty:
        candidate = _largest_polygon(shrunk) or polygon
    try:
        return polylabel(candidate, tolerance=0.004)
    except Exception:
        return candidate.representative_point()


def _label_clearance(
    x: float, y: float, cell_anchors: list[tuple[float, float, float]]
) -> float:
    if not cell_anchors:
        return 1.0
    return min(
        float(np.hypot(x - cx, y - cy)) - pad for cx, cy, pad in cell_anchors
    )


def _lineage_label_anchor(
    branch_poly: Polygon,
    cell_anchors: list[tuple[float, float, float]],
) -> tuple[float, float]:
    """Center lineage labels in their region; nudge only if overlapping a cell label."""
    # Visual centre of a moderately inset polygon = true hierarchical centre.
    center = _interior_anchor(branch_poly, shrink=0.055)
    x0, y0 = _clamp_to_dish(center.x, center.y, margin=0.09)
    if _label_clearance(x0, y0, cell_anchors) >= 0.028:
        return x0, y0

    # Local search around the centre — stay near centre, avoid cell-type text.
    best_xy = (x0, y0)
    best_score = -np.inf
    for radius in (0.015, 0.03, 0.045, 0.06, 0.08):
        for angle in np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False):
            x = x0 + radius * float(np.cos(angle))
            y = y0 + radius * float(np.sin(angle))
            x, y = _clamp_to_dish(x, y, margin=0.09)
            if not branch_poly.buffer(0.01).contains(Point(x, y)):
                continue
            clearance = _label_clearance(x, y, cell_anchors)
            # Strongly prefer staying close to the visual centre.
            score = clearance - 4.0 * float(np.hypot(x - x0, y - y0))
            if score > best_score:
                best_score = score
                best_xy = (x, y)
    return best_xy


def _inset_lineage_outline(geom, inset: float = 0.011):
    """Shrink a lineage polygon so its border sits inside its own area."""
    if geom is None or geom.is_empty:
        return None
    piece = _largest_polygon(geom) or geom
    for scale in (1.0, 0.7, 0.45, 0.25):
        shrunk = piece.buffer(-inset * scale)
        if shrunk.is_empty:
            continue
        outline = _largest_polygon(shrunk)
        if outline is not None and not outline.is_empty:
            return outline
    return None


def draw_hierarchical_voronoi_treemap(
    axis: plt.Axes,
    mosaic_bin: pd.DataFrame,
    norm: matplotlib.colors.Normalize,
    cmap: matplotlib.colors.Colormap,
    title: str,
    show_ylabel: bool,
    seed: int,
) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

    usable = mosaic_bin[mosaic_bin["identity_fraction"].notna()].copy()
    n_samples = (
        int(mosaic_bin["n_samples_in_bin"].iloc[0]) if len(mosaic_bin) else 0
    )
    if usable.empty or float(usable["identity_fraction"].sum()) <= 0:
        axis.add_patch(
            plt.Circle(
                DISH_CENTER,
                DISH_RADIUS,
                facecolor="#F2F2F2",
                edgecolor="#BDBDBD",
                lw=0.8,
            )
        )
        axis.text(
            0.5,
            0.5,
            f"No samples\n(n={n_samples})",
            ha="center",
            va="center",
            fontsize=6,
            color="#666666",
        )
        axis.set_title(title, fontsize=6.5, fontweight="bold", pad=4)
        if show_ylabel:
            axis.set_ylabel(
                "Composition\n(area = mean fraction)", fontsize=6
            )
        return

    meta = usable.drop_duplicates("hgca_celltype_v1").set_index(
        "hgca_celltype_v1"
    )
    rng = np.random.default_rng(seed)
    identity_regions, branch_regions = hierarchical_voronoi_layout(
        usable, rng=rng
    )
    text_stroke_light = [
        patheffects.withStroke(linewidth=1.7, foreground="white")
    ]
    text_stroke_dark = [
        patheffects.withStroke(linewidth=1.4, foreground="#222222")
    ]
    total_area = sum(
        geom.area for geom in identity_regions.values() if not geom.is_empty
    ) or 1.0
    cell_anchors: list[tuple[float, float, float]] = []

    for identity, geom in identity_regions.items():
        if identity not in meta.index:
            continue
        row = meta.loc[identity]
        is_other = str(identity).startswith("Other") or str(
            row["display"]
        ).startswith("Other")
        distance = row["median_nn_distance"]
        # Grey hatched = residual rare subtypes below the display cutoff
        # (not a separate "Unknown" mapping class).
        if is_other or pd.isna(distance):
            face = OTHER_CELL_FACE
            hatch = "///"
            text_color = "#222222"
            stroke = text_stroke_light
        else:
            face = cmap(norm(float(distance)))
            hatch = None
            # Blue (close) needs light text; white (far) needs dark text.
            text_color = (
                "white"
                if float(distance) <= 0.45 * float(norm.vmax)
                else "#111111"
            )
            stroke = (
                text_stroke_dark if text_color == "white" else text_stroke_light
            )
        geoms = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for piece in geoms:
            patch = _polygon_to_pathpatch(
                piece,
                facecolor=face,
                edgecolor="white",
                linewidth=0.55,
                alpha=0.98,
                hatch=hatch,
                zorder=3,
            )
            if patch is not None:
                axis.add_patch(patch)
        # Label every cell; keep anchors inside the dish rim.
        point = _interior_anchor(
            _largest_polygon(geom) or geom, shrink=0.01
        )
        lx, ly = _clamp_to_dish(point.x, point.y, margin=0.055)
        rel = float(geom.area / total_area)
        fontsize = float(np.clip(3.1 + 12.0 * np.sqrt(rel), 3.1, 5.4))
        label = mosaic_short_label(str(identity))
        # Approximate label exclusion radius for lineage placement.
        cell_anchors.append((lx, ly, 0.040 + 0.012 * (fontsize / 5.0)))
        axis.text(
            lx,
            ly,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=text_color,
            zorder=6,
            path_effects=stroke,
            clip_on=True,
        )

    for branch, geom in branch_regions.items():
        # Slight inward offset so shared edges do not double-draw, but stay flush.
        outline = _inset_lineage_outline(geom, inset=0.0035)
        if outline is not None:
            patch = _polygon_to_pathpatch(
                outline,
                facecolor="none",
                edgecolor=BRANCH_COLORS.get(branch, "#333333"),
                linewidth=1.35,
                zorder=5,
            )
            if patch is not None:
                axis.add_patch(patch)
        pct = 100 * float(
            usable.loc[usable["branch"].eq(branch), "identity_fraction"].sum()
            / usable["identity_fraction"].sum()
        )
        label_geom = _largest_polygon(geom)
        if label_geom is None or label_geom.is_empty:
            continue
        tag_x, tag_y = _lineage_label_anchor(label_geom, cell_anchors)
        lineage_text = f"{branch} {pct:.0f}%"
        half_w = 0.011 * max(len(lineage_text), 8)
        tag_x, tag_y = _clamp_text_box_to_dish(
            tag_x, tag_y, half_width=half_w, half_height=0.02, margin=0.018
        )
        if not label_geom.buffer(0.02).contains(Point(tag_x, tag_y)):
            fallback = _interior_anchor(label_geom, shrink=0.06)
            tag_x, tag_y = _clamp_text_box_to_dish(
                fallback.x,
                fallback.y,
                half_width=half_w,
                half_height=0.02,
                margin=0.018,
            )
        axis.text(
            tag_x,
            tag_y,
            lineage_text,
            ha="center",
            va="center",
            fontsize=5.0,
            fontweight="bold",
            color=BRANCH_COLORS.get(branch, "#333333"),
            zorder=7,
            path_effects=text_stroke_light,
            clip_on=True,
        )

    # Outer circular frame.
    axis.add_patch(
        plt.Circle(
            DISH_CENTER,
            DISH_RADIUS,
            facecolor="none",
            edgecolor="#222222",
            lw=1.0,
            zorder=8,
        )
    )
    axis.set_title(title, fontsize=6.5, fontweight="bold", pad=4)
    if show_ylabel:
        axis.set_ylabel("Composition\n(area = mean fraction)", fontsize=6)


def render_psc_composition_voronoi_standalone(
    mosaic: pd.DataFrame,
    segments: list[str] | None = None,
    stem: str = "fig5_psc_composition_voronoi",
    title: str | None = None,
) -> None:
    if segments is None:
        segments = ["All"]
    plot = mosaic[mosaic["segment"].isin(segments)].copy()
    n_rows = len(segments)
    height = 70 * MM if n_rows == 1 else 62 * MM * n_rows + 14 * MM
    figure = plt.figure(figsize=(180 * MM, height))
    grid = figure.add_gridspec(
        n_rows,
        4,
        width_ratios=[1, 1, 1, 0.05],
        wspace=0.1,
        hspace=0.28,
    )
    distances = plot["median_nn_distance"].dropna().to_numpy(float)
    norm = sequential_distance_norm(distances)
    cmap = MOSAIC_DISTANCE_CMAP
    for row_index, segment in enumerate(segments):
        for col_index, time_class in enumerate(TIME_BIN_ORDER):
            axis = figure.add_subplot(grid[row_index, col_index])
            bin_frame = plot[
                plot["segment"].eq(segment)
                & plot["time_class"].eq(time_class)
            ].copy()
            n_samples = (
                int(bin_frame["n_samples_in_bin"].iloc[0])
                if len(bin_frame)
                else 0
            )
            panel_title = (
                f"{time_class} (n={n_samples})"
                if row_index == 0 or n_rows == 1
                else f"n={n_samples}"
            )
            seed = (
                17
                + 100 * row_index
                + 10 * col_index
                + abs(hash((segment, time_class))) % 997
            )
            draw_hierarchical_voronoi_treemap(
                axis,
                bin_frame,
                norm=norm,
                cmap=cmap,
                title=panel_title,
                show_ylabel=col_index == 0,
                seed=seed,
            )
            if n_rows > 1 and col_index == 0:
                axis.text(
                    -0.18,
                    0.5,
                    segment.replace(" ", "\n"),
                    transform=axis.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                )
    cax = figure.add_subplot(grid[:, 3])
    _add_mosaic_colorbar(figure, cax, norm, cmap)
    legend_handles = [
        matplotlib.patches.Patch(
            facecolor="none",
            edgecolor=BRANCH_COLORS[branch],
            linewidth=1.4,
            label=branch,
        )
        for branch in BRANCH_ORDER
    ]
    legend_handles.append(
        matplotlib.patches.Patch(
            facecolor=OTHER_CELL_FACE,
            edgecolor="#666666",
            linewidth=0.5,
            hatch="///",
            label="Other rare subtypes",
        )
    )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.48, 0.01),
        ncol=4,
        frameon=False,
        fontsize=5.2,
    )
    figure.suptitle(
        title or "PSC-derived gut organoid composition over time",
        fontsize=8,
        fontweight="bold",
        y=0.98,
    )
    figure.subplots_adjust(
        left=0.1 if n_rows > 1 else 0.07,
        right=0.93,
        top=0.86,
        bottom=0.16,
    )
    for extension in ("pdf", "svg", "png"):
        figure.savefig(
            C.OUT / f"{stem}.{extension}",
            dpi=400,
            facecolor="white",
        )
    plt.close(figure)


PREFERRED_WITHIN_IDENTITIES = [
    "Intestinal Stem Cells (ISC)",
    "Transiently Amplifying Cells (TA)",
    "EEC Progenitors",
    "BEST4 Colonocytes",
    "Tuft Progenitors",
    "Mid Villus Enterocytes",
    "Villus Tip Enterocytes",
    "Lower Villus Enterocytes",
    "BEST4 Enterocytes",
    "Crypt Top Colonocytes",
    "Lower Crypt Colonocytes",
]
ENTEROTYPE_IDENTITIES = {
    "BEST4 Enterocytes",
    "Enterocyte Progenitors",
    "Lower Villus Enterocytes",
    "Mid Villus Enterocytes",
    "Villus Tip Enterocytes",
}
COLONOTYPE_IDENTITIES = {
    "BEST4 Colonocytes",
    "Colonocyte Progenitors",
    "Crypt Top Colonocytes",
    "Lower Crypt Colonocytes",
    "Mid Crypt Colonocytes",
}


def tissue_segment_class(label: str) -> str:
    text = str(label).strip().lower()
    if text in SMALL_INTESTINE_TISSUE_LABELS or any(
        token in text for token in ("duoden", "jejun", "ileum")
    ):
        return "Small Intestine"
    if text in COLON_TISSUE_LABELS or "colon" in text or "rectum" in text:
        return "Colon"
    return "Other"


def rebuild_origin_proximity_with_small_intestine() -> pd.DataFrame:
    """Treat declared Intestine as Small Intestine; score SI vs colon proximity."""
    config = C.load_config()
    distance = pd.read_csv(
        config["inputs"]["sysvi_distances"],
        usecols=[
            "cell_id",
            "sample_id",
            "origin_tissue_label",
            "d_origin",
            "d_best_other",
            "nearest_tissue_label",
            "d_nearest_tissue",
            "second_nearest_tissue_label",
            "d_second_nearest_tissue",
            "origin_rank_among_seen_tissues",
            "n_origin_neighbors_seen",
        ],
    )
    qc = pd.read_csv(
        C.DATA / "per_cell_mapping_qc_flags.csv.gz",
        usecols=[
            "cell_id",
            "sample_id",
            "hgca_pred_celltype_sysvi_knn_thresh",
            "strict_mapping_pass",
            "d_origin",
            "d_best_other",
            "relative_origin_proximity",
            "origin_region",
        ],
    )
    metadata = pd.read_csv(C.DATA / "sample_metadata.csv")
    frame = distance.merge(
        qc.drop(columns=["sample_id", "d_origin", "d_best_other"]),
        on="cell_id",
        how="inner",
        validate="one_to_one",
    )
    frame = frame.merge(
        metadata[
            [
                "sample_id",
                "publication_display",
                "source_standardized",
                "region_broad",
            ]
        ],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    intestine_mask = frame["origin_tissue_label"].astype(str).str.lower().eq(
        "intestine"
    ) | frame["region_broad"].eq("Intestine")

    nearest_class = frame["nearest_tissue_label"].map(tissue_segment_class)
    second_class = frame["second_nearest_tissue_label"].map(tissue_segment_class)
    d_si = np.where(
        nearest_class.eq("Small Intestine"),
        frame["d_nearest_tissue"],
        np.where(
            second_class.eq("Small Intestine"),
            frame["d_second_nearest_tissue"],
            np.nan,
        ),
    )
    d_colon = np.where(
        nearest_class.eq("Colon"),
        frame["d_nearest_tissue"],
        np.where(
            second_class.eq("Colon"),
            frame["d_second_nearest_tissue"],
            np.nan,
        ),
    )
    # When both top tissues are SI, use their mean as the SI distance.
    both_si = nearest_class.eq("Small Intestine") & second_class.eq(
        "Small Intestine"
    )
    d_si = np.where(
        both_si,
        np.nanmean(
            np.vstack(
                [
                    frame["d_nearest_tissue"].to_numpy(float),
                    frame["d_second_nearest_tissue"].to_numpy(float),
                ]
            ),
            axis=0,
        ),
        d_si,
    )

    frame["origin_region_plot"] = frame["origin_region"]
    frame.loc[intestine_mask, "origin_region_plot"] = "Small Intestine"
    frame.loc[intestine_mask, "d_origin"] = d_si[intestine_mask.to_numpy()]
    frame.loc[intestine_mask, "d_best_other"] = d_colon[intestine_mask.to_numpy()]
    denominator = frame["d_best_other"] + frame["d_origin"]
    frame["relative_origin_proximity"] = (
        2
        * (frame["d_best_other"] - frame["d_origin"])
        / denominator.replace(0, np.nan)
    )

    label = "hgca_pred_celltype_sysvi_knn_thresh"
    origin_specific = frame.loc[
        frame["strict_mapping_pass"]
        & frame["origin_region_plot"].isin(
            ["Duodenum", "Ileum", "Small Intestine", "Colon"]
        )
        & frame["relative_origin_proximity"].notna()
    ].copy()
    cells_per_group = int(
        config["filters"]["origin_margin_cells_per_sample_subtype"]
    )
    rng = np.random.default_rng(int(config["project"]["seed"]))
    rarefied_groups = []
    for _, group in origin_specific.groupby(
        ["sample_id", label], observed=True, sort=True
    ):
        if len(group) < cells_per_group:
            continue
        rarefied_groups.append(
            group.sample(
                n=cells_per_group,
                replace=False,
                random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
            )
        )
    if not rarefied_groups:
        raise RuntimeError("No origin-proximity groups after SI remapping")
    rarefied = pd.concat(rarefied_groups, ignore_index=True)
    rarefied["hgca_celltype_v1"] = rarefied[label].astype(str)
    rarefied["origin_region"] = rarefied["origin_region_plot"].astype(str)
    rarefied[
        [
            "cell_id",
            "sample_id",
            "publication_display",
            "source_standardized",
            "origin_region",
            "hgca_celltype_v1",
            "d_origin",
            "d_best_other",
            "relative_origin_proximity",
            "origin_rank_among_seen_tissues",
            "n_origin_neighbors_seen",
        ]
    ].to_csv(
        C.DATA / "fig5b_origin_proximity_rarefied_cells.csv.gz",
        index=False,
        compression="gzip",
    )
    summary = (
        rarefied.groupby(
            ["source_standardized", "origin_region", "hgca_celltype_v1"],
            observed=True,
        )
        .agg(
            n_cells=("cell_id", "size"),
            n_samples=("sample_id", "nunique"),
            n_publications=("publication_display", "nunique"),
            median_relative_origin_proximity=(
                "relative_origin_proximity",
                "median",
            ),
            fraction_cells_origin_closer=(
                "relative_origin_proximity",
                lambda values: (values > 0).mean(),
            ),
        )
        .reset_index()
    )
    summary.to_csv(C.DATA / "fig5b_origin_proximity_summary.csv", index=False)
    return summary


def day14_to_day56_delta() -> float:
    return float(np.log1p(56.0) - np.log1p(14.0))


def scale_log1p_effect_to_day14_56(coefficient: float) -> float:
    return float(coefficient * day14_to_day56_delta())


def fit_publication_adjusted_day_effect(
    frame: pd.DataFrame,
    outcome: str,
) -> tuple[float, float, float, float]:
    """Fit day effect; drop publication term when only one study is present."""
    if frame["publication_display"].nunique() >= 2:
        formula = f"{outcome} ~ log1p_day + publication_display"
    else:
        formula = f"{outcome} ~ log1p_day"
    fit = smf.ols(formula, data=frame).fit(cov_type="HC3")
    confidence = fit.conf_int().loc["log1p_day"]
    return (
        float(fit.params["log1p_day"]),
        float(confidence.iloc[0]),
        float(confidence.iloc[1]),
        float(fit.pvalues["log1p_day"]),
    )


def build_panel_f_decomposition(
    maturation_samples: pd.DataFrame,
    segment: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hierarchy = pd.read_csv(
        C.DATA / "hgca_epithelial_hierarchy.csv"
    ).set_index("hgca_celltype_v1")
    branches = pd.read_csv(C.DATA / "fig5h_sample_branch_composition.csv")
    metadata = pd.read_csv(C.DATA / "sample_metadata.csv")
    maturation_samples = maturation_samples.copy()
    if "segment" not in maturation_samples.columns:
        maturation_samples["segment"] = maturation_samples["region_broad"].map(
            declared_segment
        )
    if segment is not None:
        maturation_samples = maturation_samples[
            maturation_samples["segment"].eq(segment)
        ].copy()
    model_samples = maturation_samples[
        maturation_samples["model_included"]
    ].copy()
    all_numeric_samples = maturation_samples.copy()
    model_samples = model_samples.merge(
        branches[
            [
                "sample_id",
                "absorptive_fraction",
                "secretory_fraction",
                "progenitor_fraction",
            ]
        ],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if segment is None:
        if len(model_samples) != 31:
            raise RuntimeError(
                "Expected 31 within-publication samples for panel f"
            )
        if len(all_numeric_samples) != 41:
            raise RuntimeError(
                "Expected 41 numeric-time PSC samples for within-identity panel f"
            )
    elif len(model_samples) < 3 or all_numeric_samples["day"].nunique() < 2:
        raise RuntimeError(
            f"Insufficient {segment} PSC samples for panel f "
            f"(model n={len(model_samples)})"
        )
    min_identity_samples = 8 if segment != "Colon" else 3
    min_identity_pubs = 2 if segment != "Colon" else 1
    min_balanced_coverage = 12 if segment != "Colon" else 3
    segment_label = segment or "All"

    effect_rows = []
    coef, ci_low, ci_high, p_value = fit_publication_adjusted_day_effect(
        model_samples, "median_nn_distance"
    )
    effect_rows.append(
        {
            "segment": segment_label,
            "block": "Overall organoid convergence",
            "scope": "Sample-level distance",
            "identity": "All cells",
            "adjustment": "Publication",
            "display": "Unadjusted sample distance",
            "n_samples": len(model_samples),
            "n_publications": model_samples["publication_display"].nunique(),
            "coefficient_log1p_day": coef,
            "ci_low_log1p_day": ci_low,
            "ci_high_log1p_day": ci_high,
            "delta_day14_to_day56": scale_log1p_effect_to_day14_56(coef),
            "ci_low_day14_to_day56": scale_log1p_effect_to_day14_56(ci_low),
            "ci_high_day14_to_day56": scale_log1p_effect_to_day14_56(ci_high),
            "p_value": p_value,
            "branch": "",
        }
    )
    # Two of the three branch fractions are sufficient because they sum to one.
    if model_samples["publication_display"].nunique() >= 2:
        composition_formula = (
            "median_nn_distance ~ log1p_day + absorptive_fraction "
            "+ secretory_fraction + publication_display"
        )
    else:
        composition_formula = (
            "median_nn_distance ~ log1p_day + absorptive_fraction "
            "+ secretory_fraction"
        )
    try:
        composition_adjusted = smf.ols(
            composition_formula, data=model_samples
        ).fit(cov_type="HC3")
        composition_ci = composition_adjusted.conf_int().loc["log1p_day"]
        coef = float(composition_adjusted.params["log1p_day"])
        ci_low = float(composition_ci.iloc[0])
        ci_high = float(composition_ci.iloc[1])
        effect_rows.append(
            {
                "segment": segment_label,
                "block": "Overall organoid convergence",
                "scope": "Sample-level distance",
                "identity": "All cells",
                "adjustment": (
                    "Publication + absorptive and secretory fractions "
                    "(progenitor implied by unit sum)"
                ),
                "display": (
                    "Adjusted for progenitor, absorptive\nand secretory fractions"
                ),
                "n_samples": len(model_samples),
                "n_publications": model_samples["publication_display"].nunique(),
                "coefficient_log1p_day": coef,
                "ci_low_log1p_day": ci_low,
                "ci_high_log1p_day": ci_high,
                "delta_day14_to_day56": scale_log1p_effect_to_day14_56(coef),
                "ci_low_day14_to_day56": scale_log1p_effect_to_day14_56(ci_low),
                "ci_high_day14_to_day56": scale_log1p_effect_to_day14_56(
                    ci_high
                ),
                "p_value": float(composition_adjusted.pvalues["log1p_day"]),
                "branch": "",
            }
        )
    except Exception:
        pass

    cells = pd.read_csv(
        C.DATA / "per_cell_mapping_qc_flags.csv.gz",
        usecols=[
            "sample_id",
            "hgca_pred_celltype_sysvi_knn",
            "hgca_pred_celltype_sysvi_knn_thresh",
            "d_nn1",
            "confident",
        ],
    )
    cells = cells.merge(
        metadata[
            [
                "sample_id",
                "publication_display",
                "source_standardized",
                "region_broad",
                "time",
                "time_class",
            ]
        ],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    cells["day"] = parse_numeric_culture_day(cells["time"])
    # Within-identity fits use all numeric-time PSC samples so enterocyte
    # subtypes remain estimable; overall rows above stay on the n=31 set.
    cells = cells[
        cells["sample_id"].isin(all_numeric_samples["sample_id"])
        & cells["confident"]
        & cells["hgca_pred_celltype_sysvi_knn_thresh"].ne("Unknown")
        & cells["d_nn1"].notna()
        & ~cross_compartment_mask(
            cells["hgca_pred_celltype_sysvi_knn"],
            cells["region_broad"],
        )
    ].copy()
    cells["log1p_day"] = np.log1p(cells["day"])
    identity_sample = (
        cells.groupby(
            [
                "sample_id",
                "publication_display",
                "hgca_pred_celltype_sysvi_knn",
                "log1p_day",
            ],
            observed=True,
        )
        .agg(
            n_cells=("d_nn1", "size"),
            median_nn_distance=("d_nn1", "median"),
        )
        .reset_index()
        .rename(columns={"hgca_pred_celltype_sysvi_knn": "hgca_celltype_v1"})
    )
    identity_sample = identity_sample[identity_sample["n_cells"] >= 20].copy()

    # Fixed equal-weight identity panel for abundance-independent sample distance.
    identity_coverage = (
        identity_sample.groupby("hgca_celltype_v1", observed=True)["sample_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    balanced_panel = identity_coverage[
        identity_coverage >= min_balanced_coverage
    ].index.tolist()
    if len(balanced_panel) >= 3:
        balanced_rows = []
        for sample_id, frame in identity_sample.groupby(
            "sample_id", observed=True
        ):
            present = frame[
                frame["hgca_celltype_v1"].isin(balanced_panel)
            ].copy()
            if len(present) < max(
                2, int(np.ceil(0.5 * len(balanced_panel)))
            ):
                continue
            balanced_rows.append(
                {
                    "sample_id": sample_id,
                    "publication_display": present[
                        "publication_display"
                    ].iloc[0],
                    "log1p_day": present["log1p_day"].iloc[0],
                    "identity_balanced_distance": float(
                        present["median_nn_distance"].mean()
                    ),
                    "n_identities_averaged": int(len(present)),
                }
            )
        balanced = pd.DataFrame(balanced_rows)
        if (
            len(balanced) >= 3
            and balanced["log1p_day"].nunique() >= 2
            and (
                segment == "Colon"
                or balanced["publication_display"].nunique() >= 2
            )
        ):
            coef, ci_low, ci_high, p_value = fit_publication_adjusted_day_effect(
                balanced, "identity_balanced_distance"
            )
            effect_rows.append(
                {
                    "segment": segment_label,
                    "block": "Overall organoid convergence",
                    "scope": "Identity-balanced sample distance",
                    "identity": "Equal-weight recurrent identities",
                    "adjustment": "Publication; equal-weight identity means",
                    "display": "Identity-balanced distance",
                    "n_samples": int(len(balanced)),
                    "n_publications": int(
                        balanced["publication_display"].nunique()
                    ),
                    "coefficient_log1p_day": coef,
                    "ci_low_log1p_day": ci_low,
                    "ci_high_log1p_day": ci_high,
                    "delta_day14_to_day56": scale_log1p_effect_to_day14_56(
                        coef
                    ),
                    "ci_low_day14_to_day56": scale_log1p_effect_to_day14_56(
                        ci_low
                    ),
                    "ci_high_day14_to_day56": scale_log1p_effect_to_day14_56(
                        ci_high
                    ),
                    "p_value": p_value,
                    "branch": "",
                    "n_identities_in_panel": len(balanced_panel),
                }
            )

    identity_rows = []
    for identity, frame in identity_sample.groupby(
        "hgca_celltype_v1", observed=True
    ):
        if frame["sample_id"].nunique() < min_identity_samples:
            continue
        if frame["publication_display"].nunique() < min_identity_pubs:
            continue
        if frame["log1p_day"].nunique() < 2:
            continue
        varying_publications = (
            frame.groupby("publication_display")["log1p_day"].nunique() >= 2
        ).sum()
        if segment != "Colon" and varying_publications < 1:
            continue
        coef, ci_low, ci_high, p_value = fit_publication_adjusted_day_effect(
            frame, "median_nn_distance"
        )
        branch = branch_category(str(identity), hierarchy)
        identity_rows.append(
            {
                "segment": segment_label,
                "block": "Convergence within matched identities",
                "scope": "Within-identity distance",
                "identity": identity,
                "branch": branch,
                "adjustment": "Publication",
                "display": short_identity_label(str(identity)),
                "n_samples": int(frame["sample_id"].nunique()),
                "n_publications": int(frame["publication_display"].nunique()),
                "coefficient_log1p_day": coef,
                "ci_low_log1p_day": ci_low,
                "ci_high_log1p_day": ci_high,
                "delta_day14_to_day56": scale_log1p_effect_to_day14_56(coef),
                "ci_low_day14_to_day56": scale_log1p_effect_to_day14_56(ci_low),
                "ci_high_day14_to_day56": scale_log1p_effect_to_day14_56(
                    ci_high
                ),
                "p_value": p_value,
            }
        )
    identity_effects = pd.DataFrame(identity_rows)
    if identity_effects.empty and segment is None:
        raise RuntimeError("No within-identity maturation models were eligible")
    if not identity_effects.empty:
        enterocyte = identity_effects[
            identity_effects["identity"].isin(ENTEROTYPE_IDENTITIES)
        ].sort_values("delta_day14_to_day56")
        colonocyte = identity_effects[
            identity_effects["identity"].isin(COLONOTYPE_IDENTITIES)
        ]
        ci_crosses_zero = colonocyte["ci_low_day14_to_day56"].le(
            0
        ) & colonocyte["ci_high_day14_to_day56"].ge(0)
        weak_colonocyte = colonocyte[
            colonocyte["delta_day14_to_day56"].ge(-0.5) | ci_crosses_zero
        ]
        keep_identities = set(PREFERRED_WITHIN_IDENTITIES)
        keep_identities.update(enterocyte.head(4)["identity"])
        keep_identities.update(weak_colonocyte["identity"])
        # For sparse Colon strata keep every estimable identity.
        if segment == "Colon":
            keep_identities.update(identity_effects["identity"])
        identity_effects = identity_effects[
            identity_effects["identity"].isin(keep_identities)
        ].copy()

        def _panel_f_story_rank(row: pd.Series) -> tuple:
            identity = row["identity"]
            delta = float(row["delta_day14_to_day56"])
            if identity in ENTEROTYPE_IDENTITIES:
                return (0, delta, identity)
            if identity in COLONOTYPE_IDENTITIES and (
                delta >= -0.5
                or (
                    float(row["ci_low_day14_to_day56"]) <= 0
                    and float(row["ci_high_day14_to_day56"]) >= 0
                )
            ):
                return (1, -delta, identity)
            preferred_rank = {
                name: index
                for index, name in enumerate(PREFERRED_WITHIN_IDENTITIES)
            }
            return (2, preferred_rank.get(identity, 1000), identity)

        rank_frame = identity_effects.apply(
            _panel_f_story_rank, axis=1, result_type="expand"
        )
        rank_frame.columns = ["story_block", "story_key", "story_name"]
        identity_effects = (
            pd.concat([identity_effects, rank_frame], axis=1)
            .sort_values(["story_block", "story_key", "story_name"])
            .drop(columns=["story_block", "story_key", "story_name"])
        )
    effects = pd.concat(
        [pd.DataFrame(effect_rows), identity_effects],
        ignore_index=True,
        sort=False,
    )
    adjusted = effects.loc[
        effects["display"].str.contains("Adjusted for progenitor", na=False),
        "delta_day14_to_day56",
    ]
    effects["show_composition_title"] = (
        bool(len(adjusted)) and abs(float(adjusted.iloc[0])) >= 0.5
    )
    effects["segment"] = segment_label
    return identity_sample, effects


def draw_origin_proximity(
    axis: plt.Axes,
    compact: bool = False,
    color_labels_by_branch: bool = False,
) -> matplotlib.colorbar.Colorbar:
    origin = pd.read_csv(C.DATA / "fig5b_origin_proximity_summary.csv")
    hierarchy = pd.read_csv(
        C.DATA / "supp_subtype_capability_table.csv", index_col=0
    )
    origin = origin[
        (origin["n_samples"] >= 3) & (origin["n_publications"] >= 2)
    ].copy()
    small_intestine = origin["origin_region"].isin(
        ["Duodenum", "Ileum", "Small Intestine"]
    )
    origin = origin[
        ~(
            small_intestine
            & origin["hgca_celltype_v1"].str.contains(
                "Colonocyte", case=False, na=False
            )
        )
        & ~(
            origin["origin_region"].eq("Colon")
            & origin["hgca_celltype_v1"].str.contains(
                "Enterocyte", case=False, na=False
            )
        )
    ].copy()
    columns = [
        ("ASC", "Duodenum"),
        ("ASC", "Ileum"),
        ("ASC", "Colon"),
        ("FSC", "Duodenum"),
        ("FSC", "Ileum"),
        ("PSC", "Small Intestine"),
        ("PSC", "Colon"),
    ]
    column_lookup = {value: index for index, value in enumerate(columns)}
    origin = origin[
        [
            (source, region) in column_lookup
            for source, region in zip(
                origin["source_standardized"], origin["origin_region"]
            )
        ]
    ].copy()
    subtype_order = [
        subtype
        for subtype in taxonomy_order(hierarchy)
        if subtype in set(origin["hgca_celltype_v1"])
    ]
    origin["x"] = [
        column_lookup[(source, region)]
        for source, region in zip(
            origin["source_standardized"], origin["origin_region"]
        )
    ]
    origin["y"] = origin["hgca_celltype_v1"].map(
        {value: index for index, value in enumerate(subtype_order)}
    )
    proximity_limit = max(
        0.1,
        float(
            np.nanquantile(
                np.abs(origin["median_relative_origin_proximity"]), 0.98
            )
        ),
    )
    proximity_cmap = sns.color_palette("vlag", as_cmap=True)
    sns.scatterplot(
        data=origin,
        x="x",
        y="y",
        hue="median_relative_origin_proximity",
        size="fraction_cells_origin_closer",
        palette=proximity_cmap,
        hue_norm=(-proximity_limit, proximity_limit),
        sizes=(4, 36) if compact else (6, 62),
        edgecolor="#333333",
        linewidth=0.18 if compact else 0.25,
        legend=False,
        ax=axis,
    )
    x_labels = [
        region.replace("Duodenum", "Duod.")
        .replace("Small Intestine", "SI" if compact else "Small\nint.")
        for _, region in columns
    ]
    axis.set_xticks(range(len(columns)), x_labels)
    for tick, (_, region) in zip(axis.get_xticklabels(), columns):
        tick.set_color(SEGMENT_COLORS.get(region, "#333333"))
    axis.set_yticks(
        range(len(subtype_order)),
        [short_identity_label(value) for value in subtype_order],
    )
    if color_labels_by_branch:
        for tick, subtype in zip(axis.get_yticklabels(), subtype_order):
            branch = branch_category(subtype, hierarchy)
            tick.set_color(BRANCH_COLORS.get(branch, "#333333"))
    y_top = -0.55 if compact else -1.0
    axis.set_ylim(len(subtype_order) - 0.45, y_top)
    axis.set_xlim(-0.45, len(columns) - 0.55)
    axis.set_xlabel(
        "Declared origin" if compact else "Declared segment of origin",
        fontsize=4.2 if compact else 6,
        labelpad=1 if compact else 3,
    )
    axis.set_ylabel(
        "HGCA subtype" if compact else "HGCA epithelial subtype",
        fontsize=4.2 if compact else 6,
        labelpad=1 if compact else 3,
    )
    axis.tick_params(length=0, labelsize=3.4 if compact else 4.2)
    for boundary in [2.5, 4.5]:
        axis.axvline(boundary, color="#BDBDBD", lw=0.4 if compact else 0.45)
    source_y = -0.42 if compact else -0.85
    source_size = 3.6 if compact else 4.5
    for center, text in [(1, "ASC"), (3.5, "FSC"), (5.5, "PSC")]:
        axis.text(
            center,
            source_y,
            text,
            ha="center",
            va="bottom",
            fontsize=source_size,
            fontweight="bold",
        )
    for spine in axis.spines.values():
        spine.set_visible(False)
    if compact:
        color_axis = axis.inset_axes([1.01, 0.55, 0.03, 0.38])
    else:
        color_axis = axis.inset_axes([1.02, 0.42, 0.025, 0.4])
    colorbar = axis.figure.colorbar(
        plt.cm.ScalarMappable(
            norm=matplotlib.colors.TwoSlopeNorm(
                vmin=-proximity_limit,
                vcenter=0,
                vmax=proximity_limit,
            ),
            cmap=proximity_cmap,
        ),
        cax=color_axis,
    )
    colorbar.set_label(
        "Origin proximity" if compact else "Median relative origin proximity",
        fontsize=3.6 if compact else 4.2,
        labelpad=2 if compact else 3,
    )
    colorbar.ax.tick_params(labelsize=3.2 if compact else 3.8, length=1.5)
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="#333333",
            markeredgewidth=0.25 if compact else 0.3,
            markersize=size,
            label=label,
        )
        for size, label in (
            [(1.8, "25%"), (3.2, "50%"), (5.0, "100%")]
            if compact
            else [(2.5, "25%"), (4.5, "50%"), (7, "100%")]
        )
    ]
    size_legend = axis.legend(
        handles=size_handles,
        title="Closer to origin",
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0),
        fontsize=3.2 if compact else 3.8,
        title_fontsize=3.4 if compact else 4,
        borderaxespad=0,
        handletextpad=0.3,
        labelspacing=0.25,
    )
    if compact and color_labels_by_branch:
        axis.add_artist(size_legend)
        axis.legend(
            handles=[
                matplotlib.patches.Patch(
                    facecolor=BRANCH_COLORS[branch],
                    edgecolor="white",
                    linewidth=0.2,
                    label=branch,
                )
                for branch in BRANCH_ORDER
            ],
            loc="upper left",
            bbox_to_anchor=(1.01, 0.48),
            frameon=False,
            fontsize=3.2,
            handlelength=0.7,
            handletextpad=0.3,
            borderaxespad=0,
            title="Lineage",
            title_fontsize=3.4,
        )
    return colorbar


def render_panel_b_compact() -> None:
    """Standalone compact origin-proximity matrix matching composite panel b."""
    configure_style()
    origin = pd.read_csv(C.DATA / "fig5b_origin_proximity_summary.csv")
    hierarchy = pd.read_csv(
        C.DATA / "supp_subtype_capability_table.csv", index_col=0
    )
    keep = origin[
        (origin["n_samples"] >= 3) & (origin["n_publications"] >= 2)
    ]
    n_subtypes = len(
        [
            subtype
            for subtype in taxonomy_order(hierarchy)
            if subtype in set(keep["hgca_celltype_v1"])
        ]
    )
    width = 68 * MM
    height = max(38 * MM, (6.5 + 1.35 * n_subtypes) * MM)
    figure = plt.figure(figsize=(width, height))
    axis = figure.add_subplot(111)
    draw_origin_proximity(
        axis, compact=True, color_labels_by_branch=True
    )
    figure.suptitle(
        "Origin proximity by subtype, source and segment",
        fontsize=5.0,
        fontweight="bold",
        x=0.02,
        y=0.995,
        ha="left",
        va="top",
    )
    figure.subplots_adjust(left=0.28, right=0.76, top=0.88, bottom=0.11)
    for extension in ("pdf", "svg", "png"):
        figure.savefig(
            C.OUT / f"fig5_b_origin_proximity.{extension}",
            dpi=400,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(figure)


def _forest_row_color(row: pd.Series) -> str:
    branch = str(row.get("branch", "") or "").strip()
    if branch in BRANCH_COLORS:
        return BRANCH_COLORS[branch]
    return "#333333"


def _short_forest_label(display: str, n_samples: float, compact: bool) -> str:
    text = " ".join(str(display).split())
    if compact:
        replacements = {
            "Adjusted for progenitor, absorptive and secretory fractions": (
                "Composition-adjusted"
            ),
            "Unadjusted sample distance": "Unadjusted",
            "Identity-balanced distance": "Identity-balanced",
        }
        for old, new in replacements.items():
            if text.startswith(old) or text == old:
                text = new
                break
        return f"{text} (n={int(n_samples)})"
    return f"{text}  (n={int(n_samples)} samples)"


def draw_forest_panel(
    axis: plt.Axes,
    decomposition_effects: pd.DataFrame,
    title: str | None,
    panel_tag: str | None = "f",
    within_color: str = "#4C78A8",
    compact: bool = False,
    color_by_branch: bool = False,
) -> None:
    overall = decomposition_effects[
        decomposition_effects["block"].eq("Overall organoid convergence")
    ].copy().reset_index(drop=True)
    within = decomposition_effects[
        decomposition_effects["block"].eq(
            "Convergence within matched identities"
        )
    ].copy().reset_index(drop=True)
    if color_by_branch and not within.empty and "branch" in within.columns:
        branch_rank = {
            name: index for index, name in enumerate(BRANCH_ORDER)
        }
        within["_branch_rank"] = within["branch"].map(
            lambda value: branch_rank.get(str(value), 99)
        )
        within = within.sort_values(
            ["_branch_rank", "delta_day14_to_day56"]
        ).drop(columns=["_branch_rank"])
    header_overall = pd.DataFrame(
        [
            {
                "block": "header",
                "display": (
                    "Overall" if compact else "Overall organoid convergence"
                ),
                "branch": "",
                "n_samples": np.nan,
                "delta_day14_to_day56": np.nan,
                "ci_low_day14_to_day56": np.nan,
                "ci_high_day14_to_day56": np.nan,
            }
        ]
    )
    header_within = pd.DataFrame(
        [
            {
                "block": "header",
                "display": (
                    "Within identity"
                    if compact
                    else "Convergence within matched identities"
                ),
                "branch": "",
                "n_samples": np.nan,
                "delta_day14_to_day56": np.nan,
                "ci_low_day14_to_day56": np.nan,
                "ci_high_day14_to_day56": np.nan,
            }
        ]
    )
    parts = [header_overall, overall]
    if len(within):
        parts.extend([header_within, within])
    forest = pd.concat(parts, ignore_index=True, sort=False)
    if forest["delta_day14_to_day56"].notna().sum() == 0:
        axis.text(
            0.5,
            0.5,
            "Insufficient samples\nfor day-effect estimates",
            ha="center",
            va="center",
            fontsize=5.5,
            color="#666666",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        if title:
            axis.set_title(title, fontsize=6, fontweight="bold", loc="left")
        if panel_tag:
            panel_label(axis, panel_tag)
        return

    row_gap = 0.52 if compact else 1.0
    forest["y"] = np.arange(len(forest), dtype=float)[::-1] * row_gap
    point_size = 7 if compact else 12
    line_width = 0.65 if compact else 0.85
    finite = forest["delta_day14_to_day56"].notna()
    x_vals = np.concatenate(
        [
            forest.loc[finite, "delta_day14_to_day56"].to_numpy(float),
            forest.loc[finite, "ci_low_day14_to_day56"].to_numpy(float),
            forest.loc[finite, "ci_high_day14_to_day56"].to_numpy(float),
        ]
    )
    x_vals = x_vals[np.isfinite(x_vals)]
    if len(x_vals):
        x_pad = (0.08 if compact else 0.15) * max(float(np.ptp(x_vals)), 1.0)
        axis.set_xlim(float(np.min(x_vals)) - x_pad, float(np.max(x_vals)) + x_pad)

    for _, row in forest.iterrows():
        if row["block"] == "header" or pd.isna(row["delta_day14_to_day56"]):
            continue
        if color_by_branch:
            color = _forest_row_color(row)
        else:
            is_overall = row["block"] == "Overall organoid convergence"
            color = "#222222" if is_overall else within_color
        axis.plot(
            [row["ci_low_day14_to_day56"], row["ci_high_day14_to_day56"]],
            [row["y"], row["y"]],
            color=color,
            lw=line_width,
            solid_capstyle="round",
        )
        axis.scatter(
            row["delta_day14_to_day56"],
            row["y"],
            s=point_size,
            color=color,
            edgecolor="white",
            linewidth=0.15,
            zorder=3,
        )
    axis.axvline(0, color="#999999", lw=0.45)
    labels = []
    for _, row in forest.iterrows():
        if row["block"] == "header":
            labels.append(row["display"])
        else:
            labels.append(
                _short_forest_label(
                    row["display"], row["n_samples"], compact=compact
                )
            )
    axis.set_yticks(forest["y"], labels)
    label_size = 3.7 if compact else 3.9
    header_size = 4.0 if compact else 4.4
    for tick, (_, row) in zip(axis.get_yticklabels(), forest.iterrows()):
        if row["block"] == "header":
            tick.set_fontweight("bold")
            tick.set_fontsize(header_size)
        else:
            tick.set_fontsize(label_size)
            if color_by_branch and compact:
                tick.set_color(_forest_row_color(row))
    if len(within) and len(overall):
        header_within_y = forest.loc[
            forest["display"].eq("Within identity")
            | forest["display"].eq("Convergence within matched identities"),
            "y",
        ].iloc[0]
        axis.axhline(
            header_within_y + 0.32 * row_gap, color="#BDBDBD", lw=0.5
        )
    axis.tick_params(axis="y", length=0, pad=1.0)
    axis.tick_params(axis="x", labelsize=3.8 if compact else 5)
    axis.set_xlabel(
        "Change in HGCA distance (d14-d56)" if compact else (
            "Change in HGCA distance, day 14 to 56\n"
            "(negative = closer to adult)"
        ),
        fontsize=4.2 if compact else 5.5,
        labelpad=1 if compact else 2,
    )
    if title is None and bool(
        decomposition_effects["show_composition_title"].iloc[0]
    ):
        title = (
            "Maturation-associated convergence is only\n"
            "partly explained by epithelial composition"
        )
    if title:
        axis.set_title(
            title,
            fontsize=5.2 if compact else 5.5,
            fontweight="bold",
            pad=1 if compact else 4,
            loc="left",
        )
    y_min = float(forest["y"].min()) - 0.28 * row_gap
    y_max = float(forest["y"].max()) + 0.32 * row_gap
    axis.set_ylim(y_min, y_max)
    sns.despine(ax=axis)
    if panel_tag:
        panel_label(axis, panel_tag)


def _branch_forest_legend_handles() -> list[matplotlib.patches.Patch]:
    handles = [
        matplotlib.patches.Patch(
            facecolor=BRANCH_COLORS[branch],
            edgecolor="white",
            linewidth=0.2,
            label=branch,
        )
        for branch in BRANCH_ORDER
    ]
    handles.append(
        matplotlib.patches.Patch(
            facecolor="#333333",
            edgecolor="white",
            linewidth=0.2,
            label="Overall",
        )
    )
    return handles


def render_panel_f_by_segment(
    effects_by_segment: dict[str, pd.DataFrame],
) -> None:
    """Write one compact forest plot per declared segment."""
    stem_by_segment = {
        "Small Intestine": "fig5_f_maturation_decomposition_intestine",
        "Colon": "fig5_f_maturation_decomposition_colon",
    }
    title_by_segment = {
        "Small Intestine": "Intestine",
        "Colon": "Colon",
    }
    legend_handles = _branch_forest_legend_handles()
    for segment in SEGMENT_ORDER:
        effects = effects_by_segment.get(segment, pd.DataFrame())
        n_rows = 2 if effects.empty else int(len(effects) + 2)
        height = max(24 * MM, (1.8 + 1.55 * n_rows) * MM)
        width = 44 * MM
        figure = plt.figure(figsize=(width, height))
        axis = figure.add_subplot(111)
        if effects.empty:
            axis.text(
                0.5,
                0.5,
                f"No {segment} estimates",
                ha="center",
                va="center",
                fontsize=5.5,
            )
            axis.set_axis_off()
        else:
            draw_forest_panel(
                axis,
                effects,
                title="",
                panel_tag=None,
                compact=True,
                color_by_branch=True,
            )
        figure.suptitle(
            title_by_segment.get(segment, segment),
            fontsize=5.4,
            fontweight="bold",
            x=0.02,
            y=0.995,
            ha="left",
            va="top",
        )
        figure.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.99, 0.995),
            ncol=4,
            frameon=False,
            fontsize=3.2,
            handlelength=0.65,
            handletextpad=0.25,
            columnspacing=0.45,
            borderaxespad=0.0,
        )
        figure.subplots_adjust(
            left=0.52, right=0.99, top=0.88, bottom=0.13
        )
        stem = stem_by_segment[segment]
        for extension in ("pdf", "svg", "png"):
            figure.savefig(
                C.OUT / f"{stem}.{extension}",
                dpi=400,
                facecolor="white",
                bbox_inches="tight",
                pad_inches=0.02,
            )
        plt.close(figure)

    # Keep a tiny side-by-side index figure for convenience.
    figure = plt.figure(figsize=(88 * MM, 36 * MM))
    grid = figure.add_gridspec(1, 2, wspace=0.40)
    for index, segment in enumerate(SEGMENT_ORDER):
        axis = figure.add_subplot(grid[0, index])
        effects = effects_by_segment.get(segment, pd.DataFrame())
        if effects.empty:
            axis.text(0.5, 0.5, f"No {segment}", ha="center", va="center")
            axis.set_axis_off()
            continue
        draw_forest_panel(
            axis,
            effects,
            title=title_by_segment.get(segment, segment),
            panel_tag=None,
            compact=True,
            color_by_branch=True,
        )
    figure.subplots_adjust(left=0.30, right=0.99, top=0.86, bottom=0.16)
    for extension in ("pdf", "svg", "png"):
        figure.savefig(
            C.OUT / f"fig5_f_maturation_decomposition_by_segment.{extension}",
            dpi=350,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(figure)


def render_fig5_selective_maturation_composite(
    branch_distance: pd.DataFrame,
    branch_contrasts: pd.DataFrame,
    composition_mosaic: pd.DataFrame,
    maturation_samples: pd.DataFrame,
    maturation_models: pd.DataFrame,
    decomposition_effects: pd.DataFrame,
) -> None:
    configure_style()
    figure = plt.figure(figsize=(180 * MM, 280 * MM))
    outer = figure.add_gridspec(
        3,
        1,
        height_ratios=[1.0, 1.25, 0.95],
        hspace=0.36,
    )

    # Row 1: a blank ARBOL + b origin proximity
    top = outer[0].subgridspec(1, 2, width_ratios=[1.05, 1.2], wspace=0.28)
    axis_a = figure.add_subplot(top[0, 0])
    axis_a.axis("off")
    panel_label(axis_a, "a")
    axis_a.text(
        0.5,
        0.5,
        "ARBOL taxonomy mapping\n(insert external graphic)",
        ha="center",
        va="center",
        fontsize=6,
        color="#777777",
        transform=axis_a.transAxes,
    )

    axis_b = figure.add_subplot(top[0, 1])
    draw_origin_proximity(axis_b)
    panel_label(axis_b, "b")

    # Row 2: c branch distances + d identity distance/fraction vs time
    middle = outer[1].subgridspec(1, 2, width_ratios=[0.9, 1.35], wspace=0.32)
    axis_c = figure.add_subplot(middle[0, 0])
    plot_c = branch_distance.copy()
    plot_c["branch_x"] = plot_c["branch"].map(
        {value: index for index, value in enumerate(BRANCH_ORDER)}
    ).astype(float)
    # Small horizontal jitter by source for readability.
    source_offset = {"ASC": -0.12, "FSC": 0.0, "PSC": 0.12}
    plot_c["x"] = plot_c["branch_x"] + plot_c["source_standardized"].map(
        source_offset
    )
    for sample_id, frame in plot_c.groupby("sample_id", observed=True):
        if len(frame) < 2:
            continue
        ordered = frame.sort_values("branch_x")
        axis_c.plot(
            ordered["x"],
            ordered["median_nn_distance"],
            color="#D0D0D0",
            lw=0.35,
            alpha=0.55,
            zorder=1,
        )
    for source, color in {
        "ASC": "#0072B2",
        "FSC": "#009E73",
        "PSC": "#D55E00",
    }.items():
        subset = plot_c[plot_c["source_standardized"].eq(source)]
        axis_c.scatter(
            subset["x"],
            subset["median_nn_distance"],
            s=8,
            color=color,
            edgecolor="white",
            linewidth=0.2,
            alpha=0.85,
            zorder=3,
            label=source,
        )
    axis_c.set_xticks(
        range(3),
        ["Progenitor", "Differentiated\nabsorptive", "Differentiated\nsecretory"],
    )
    axis_c.set_ylabel(
        "Sample median nearest-HGCA-reference distance\n(lower = closer)"
    )
    axis_c.set_xlabel("")
    axis_c.legend(
        frameon=False,
        loc="upper right",
        fontsize=4,
        title="Source",
        title_fontsize=4.2,
    )
    if not branch_contrasts.empty:
        bits = []
        for _, row in branch_contrasts.iterrows():
            q = row.get("q_value", np.nan)
            bits.append(
                f"{row['category'][0]}−{row['baseline'][0]} "
                f"Δ={row['median_paired_difference']:.2f}"
                + (f", q={q:.2g}" if pd.notna(q) else "")
            )
        axis_c.text(
            0.02,
            0.98,
            "Paired Wilcoxon FDR:\n" + "\n".join(bits),
            transform=axis_c.transAxes,
            va="top",
            fontsize=3.8,
        )
    sns.despine(ax=axis_c)
    panel_label(axis_c, "c")

    draw_panel_d_composition_mosaic(
        figure,
        middle[0, 1],
        composition_mosaic,
        segments=SEGMENT_ORDER,
        panel_tag="d",
    )

    # Row 3: e maturation distance + f decomposition forest
    bottom = outer[2].subgridspec(
        1, 2, width_ratios=[1.25, 1.0], wspace=0.32
    )
    e_grid = bottom[0, 0].subgridspec(1, 2, wspace=0.35)
    across = maturation_models[
        maturation_models["model_type"].eq("across_study_ols_hc3")
    ].iloc[0]
    within = maturation_models[
        maturation_models["model_type"].eq(
            "within_study_publication_adjusted_ols_hc3"
        )
    ].iloc[0]
    e_axes = [
        figure.add_subplot(e_grid[0, 0]),
        figure.add_subplot(e_grid[0, 1]),
    ]
    within_samples = maturation_samples[
        maturation_samples["model_included"]
    ].copy()
    tick_days = np.array([0, 3, 7, 14, 28, 56, 98], dtype=float)
    tick_positions = np.log1p(tick_days)
    for axis_index, (axis, plot_samples, title, model) in enumerate(
        [
            (
                e_axes[0],
                maturation_samples,
                "Across studies",
                across,
            ),
            (
                e_axes[1],
                within_samples,
                "Within studies",
                within,
            ),
        ]
    ):
        for time_class in ["≤14 d", "15–55 d", "≥56 d"]:
            subset = plot_samples[plot_samples["time_class"].eq(time_class)]
            axis.scatter(
                subset["log1p_day"],
                subset["median_nn_distance"],
                s=11,
                color=TIME_COLORS[time_class],
                edgecolor="white",
                linewidth=0.2,
                zorder=3,
            )
        if axis_index == 0:
            fit = smf.ols(
                "median_nn_distance ~ log1p_day",
                data=plot_samples,
            ).fit(cov_type="HC3")
            pred = fit.get_prediction(
                pd.DataFrame(
                    {
                        "log1p_day": np.linspace(
                            plot_samples["log1p_day"].min(),
                            plot_samples["log1p_day"].max(),
                            200,
                        )
                    }
                )
            ).summary_frame(alpha=0.05)
            x_grid = np.linspace(
                plot_samples["log1p_day"].min(),
                plot_samples["log1p_day"].max(),
                200,
            )
            axis.plot(x_grid, pred["mean"], color="#222222", lw=0.8)
            axis.fill_between(
                x_grid,
                pred["mean_ci_lower"],
                pred["mean_ci_upper"],
                color="#777777",
                alpha=0.15,
                linewidth=0,
            )
        else:
            # Publication-specific slopes plus equal-publication marginal.
            for publication, frame in plot_samples.groupby(
                "publication_display", observed=True
            ):
                if frame["log1p_day"].nunique() < 2:
                    continue
                slope = smf.ols(
                    "median_nn_distance ~ log1p_day", data=frame
                ).fit()
                x_values = np.linspace(
                    frame["log1p_day"].min(),
                    frame["log1p_day"].max(),
                    50,
                )
                axis.plot(
                    x_values,
                    slope.params["Intercept"]
                    + slope.params["log1p_day"] * x_values,
                    color="#BDBDBD",
                    lw=0.55,
                    zorder=2,
                )
            adjusted = smf.ols(
                "median_nn_distance ~ log1p_day + publication_display",
                data=plot_samples,
            ).fit(cov_type="HC3")
            parameter_names = list(adjusted.params.index)
            publication_terms = [
                name
                for name in parameter_names
                if name.startswith("publication_display[T.")
            ]
            n_publications = plot_samples["publication_display"].nunique()
            x_grid = np.linspace(
                maturation_samples["log1p_day"].min(),
                maturation_samples["log1p_day"].max(),
                200,
            )
            fitted = []
            for value in x_grid:
                contrast = np.zeros(len(parameter_names), dtype=float)
                contrast[parameter_names.index("Intercept")] = 1.0
                contrast[parameter_names.index("log1p_day")] = value
                for name in publication_terms:
                    contrast[parameter_names.index(name)] = 1 / n_publications
                fitted.append(
                    float(contrast @ adjusted.params.to_numpy())
                )
            axis.plot(x_grid, fitted, color="#222222", lw=0.9, zorder=4)
        axis.set_xticks(tick_positions, [str(int(day)) for day in tick_days])
        axis.set_xlabel("Maturation day")
        if axis_index == 0:
            axis.set_ylabel(
                "Sample median HGCA distance\n(lower = closer)"
            )
            panel_label(axis, "e")
        else:
            axis.set_ylabel("")
        axis.text(
            0.02,
            0.98,
            f"{title}\n"
            f"β={model['coefficient']:.2f} "
            f"[{model['ci_low']:.2f}, {model['ci_high']:.2f}]",
            transform=axis.transAxes,
            va="top",
            fontsize=3.9,
            fontweight="bold",
        )
        sns.despine(ax=axis)

    axis_f = figure.add_subplot(bottom[0, 1])
    draw_forest_panel(
        axis_f,
        decomposition_effects,
        title=None,
        panel_tag="f",
    )

    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=TIME_COLORS[key],
                markeredgecolor="white",
                markersize=4,
                label=key,
            )
            for key in ["≤14 d", "15–55 d", "≥56 d"]
        ],
        title="Maturation/time",
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.38, 0.012),
        ncol=3,
        fontsize=4,
        title_fontsize=4.2,
    )
    figure.subplots_adjust(
        left=0.1, right=0.92, top=0.98, bottom=0.06
    )
    for extension in ("pdf", "svg", "png"):
        figure.savefig(
            C.OUT / f"fig5_selective_maturation_composite.{extension}",
            dpi=300,
            facecolor="white",
        )
    plt.close(figure)


def main() -> None:
    logger = C.setup_logging("06_fig5_selective_maturation_composite")
    configure_style()
    origin_summary = rebuild_origin_proximity_with_small_intestine()
    logger.info(
        "Rebuilt origin proximity with Small Intestine remapping: %s rows",
        len(origin_summary),
    )
    branch_distance, branch_contrasts = build_panel_c_branch_distance()
    branch_distance.to_csv(
        C.DATA / "fig5_c_branch_reference_distance_by_sample.csv",
        index=False,
    )
    branch_contrasts.to_csv(
        C.DATA / "fig5_c_branch_reference_distance_contrasts.csv",
        index=False,
    )
    # Keep lineage distance/fraction tables for supplementary reuse.
    identity_time_samples, identity_time_models = (
        build_panel_d_identity_time_tables()
    )
    identity_time_samples.to_csv(
        C.DATA / "fig5_d_psc_identity_distance_fraction_by_time.csv",
        index=False,
    )
    identity_time_models.to_csv(
        C.DATA / "fig5_d_psc_identity_lineage_time_models.csv",
        index=False,
    )
    composition_mosaic = build_psc_composition_mosaic_table()
    composition_mosaic.to_csv(
        C.DATA / "fig5_d_psc_composition_mosaic_by_time.csv",
        index=False,
    )
    render_psc_composition_mosaic_standalone(
        composition_mosaic,
        segments=["All"],
        stem="fig5_psc_composition_mosaic",
        title="PSC epithelial composition across maturation",
    )
    render_psc_composition_mosaic_standalone(
        composition_mosaic,
        segments=SEGMENT_ORDER,
        stem="fig5_d_psc_composition_mosaic_by_segment",
        title="PSC composition by declared segment and maturation",
    )
    render_psc_composition_voronoi_standalone(
        composition_mosaic,
        segments=["All"],
        stem="fig5_psc_composition_voronoi",
        title="PSC-derived gut organoid composition over time",
    )
    render_psc_composition_voronoi_standalone(
        composition_mosaic,
        segments=SEGMENT_ORDER,
        stem="fig5_d_psc_composition_voronoi_by_segment",
        title="PSC-derived gut organoid composition over time",
    )
    render_psc_composition_voronoi_standalone(
        composition_mosaic,
        segments=["Small Intestine"],
        stem="fig5_psc_composition_voronoi_intestine",
        title="PSC-derived gut organoid composition over time (Intestine)",
    )
    render_psc_composition_voronoi_standalone(
        composition_mosaic,
        segments=["Colon"],
        stem="fig5_psc_composition_voronoi_colon",
        title="PSC-derived gut organoid composition over time (Colon)",
    )
    maturation_samples = pd.read_csv(
        C.DATA / "fig5h_f_maturation_distance_samples.csv"
    )
    maturation_models = pd.read_csv(
        C.DATA / "fig5h_f_maturation_distance_models.csv"
    )
    identity_sample, decomposition_effects = build_panel_f_decomposition(
        maturation_samples
    )
    identity_sample.to_csv(
        C.DATA / "fig5_f_within_identity_distance_by_sample.csv",
        index=False,
    )
    decomposition_effects.to_csv(
        C.DATA / "fig5_f_maturation_distance_decomposition.csv",
        index=False,
    )
    effects_by_segment: dict[str, pd.DataFrame] = {}
    segment_effect_frames = []
    for segment in SEGMENT_ORDER:
        try:
            _, segment_effects = build_panel_f_decomposition(
                maturation_samples, segment=segment
            )
            effects_by_segment[segment] = segment_effects
            segment_effect_frames.append(segment_effects)
            logger.info(
                "Panel f %s: %s effect rows",
                segment,
                len(segment_effects),
            )
        except Exception as exc:
            logger.warning("Panel f %s skipped: %s", segment, exc)
            effects_by_segment[segment] = pd.DataFrame()
    if segment_effect_frames:
        pd.concat(segment_effect_frames, ignore_index=True).to_csv(
            C.DATA / "fig5_f_maturation_distance_decomposition_by_segment.csv",
            index=False,
        )
    render_panel_b_compact()
    render_panel_f_by_segment(effects_by_segment)
    render_fig5_selective_maturation_composite(
        branch_distance,
        branch_contrasts,
        composition_mosaic,
        maturation_samples,
        maturation_models,
        decomposition_effects,
    )
    logger.info(
        "Wrote selective-maturation Figure 5 with segment-stratified d/f"
    )


if __name__ == "__main__":
    main()
