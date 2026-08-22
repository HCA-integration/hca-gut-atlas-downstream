#!/usr/bin/env python3
from __future__ import annotations

import textwrap
import warnings

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from adjustText import adjust_text
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
from sklearn.metrics import balanced_accuracy_score
from statsmodels.genmod.cov_struct import Exchangeable

import common as C


MM = 1 / 25.4
SEGMENTS = ["duodenum", "jejunum", "ileum", "colon"]
SEGMENT_DISPLAY = {
    "duodenum": "Duodenum",
    "jejunum": "Jejunum",
    "ileum": "Ileum",
    "colon": "Colon",
}
SEGMENT_COLORS = {
    "Duodenum": "#E9C61D",
    "Jejunum": "#E96475",
    "Ileum": "#208230",
    "Colon": "#3A68AE",
}
SOURCE_LABELS = {
    "ASC": "Adult stem cell-derived (ASC)",
    "FSC": "Fetal stem cell-derived (FSC)",
    "PSC": "Pluripotent stem cell-derived (PSC)",
}
MATURATION_COLORS = {
    "Shared proliferative/progenitor": "#CC79A7",
    "Mature identity": "#0072B2",
    "Lineage-restricted progenitor": "#E69F00",
    "Other identity": "#999999",
}
HEALTHY_CACHE_VERSION = "exclude_cross_compartment_epithelial_labels_v1"
BRANCH_COLORS = {
    "Absorptive": "#0072B2",
    "Secretory": "#CC79A7",
    "Progenitor": "#E69F00",
}
TIME_COLORS = {
    "≤14 d": "#56B4E9",
    "15–55 d": "#0072B2",
    "≥56 d": "#CC79A7",
    "Early": "#F0E442",
    "Late": "#D55E00",
    "Not reported": "#BDBDBD",
}
PROGENITOR_IDENTITIES = {
    "Intestinal Stem Cells (ISC)",
    "Transiently Amplifying Cells (TA)",
    "Secretory Progenitors",
    "Enterocyte Progenitors",
    "Colonocyte Progenitors",
    "EEC Progenitors",
    "Tuft Progenitors",
}
SECRETORY_IDENTITY_ORDER = [
    "BEST4 Enterocytes",
    "BEST4 Colonocytes",
    "Goblet Cells",
    "Mature Goblet Cells",
    "Paneth Cells",
    "Tuft Cells",
    "EEC Enterochromaffin (EC)",
    "EEC L",
    "EEC N",
    "EEC S",
    "Enteroendocrine Cells (EEC)",
    "Brunners Gland Cells",
    "Foveolar Cells",
]


def configure_style() -> None:
    sns.set_theme(style="ticks", context="paper")
    matplotlib.rcParams.update(
        {
            "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 6,
            "axes.titlesize": 7,
            "axes.labelsize": 6,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "legend.fontsize": 5,
            "legend.title_fontsize": 5,
            "axes.linewidth": 0.5,
        }
    )


def maturation_class(label: str) -> str:
    shared = {
        "Intestinal Stem Cells (ISC)",
        "Transiently Amplifying Cells (TA)",
        "Secretory Progenitors",
    }
    lineage_restricted = {
        "Enterocyte Progenitors",
        "Colonocyte Progenitors",
    }
    mature = {
        "Lower Villus Enterocytes",
        "Mid Villus Enterocytes",
        "Villus Tip Enterocytes",
        "Lower Crypt Colonocytes",
        "Mid Crypt Colonocytes",
        "Crypt Top Colonocytes",
        "Goblet Cells",
        "Mature Goblet Cells",
        "BEST4 Enterocytes",
        "BEST4 Colonocytes",
    }
    if label in shared:
        return "Shared proliferative/progenitor"
    if label in lineage_restricted:
        return "Lineage-restricted progenitor"
    if label in mature:
        return "Mature identity"
    return "Other identity"


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
        "EEC Enterochromaffin (EC)": "EEC EC",
        "Enteroendocrine Cells (EEC)": "EEC (broad)",
        "Brunners Gland Cells": "Brunner's gland cells",
    }
    return replacements.get(
        label, textwrap.shorten(label, width=23, placeholder="…")
    )


def balanced_mean(values: np.ndarray, groups: np.ndarray) -> float:
    return float(
        np.mean(
            [
                np.mean(values[groups == group])
                for group in np.unique(groups)
            ]
        )
    )


def knn_metrics(
    latent: np.ndarray,
    segment: np.ndarray,
    donor: np.ndarray,
    k: int,
) -> tuple[float, float]:
    distances = cdist(latent, latent, metric="euclidean")
    same_donor = donor[:, None] == donor[None, :]
    distances[same_donor] = np.inf
    max_external = int((~same_donor).sum(axis=1).min())
    effective_k = max(1, min(k, max_external))
    neighbors = np.argpartition(
        distances, kth=effective_k - 1, axis=1
    )[:, :effective_k]
    neighbor_labels = segment[neighbors]
    predictions = np.array(
        [
            np.unique(row, return_counts=True)[0][
                np.argmax(np.unique(row, return_counts=True)[1])
            ]
            for row in neighbor_labels
        ]
    )
    accuracy = balanced_accuracy_score(segment, predictions)
    same_fraction = (neighbor_labels == segment[:, None]).mean(axis=1)
    return float(accuracy), balanced_mean(same_fraction, segment)


def bootstrap_mean_ci(
    values: np.ndarray, repeats: int, rng: np.random.Generator
) -> tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    boot = np.array(
        [
            rng.choice(values, size=len(values), replace=True).mean()
            for _ in range(repeats)
        ]
    )
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def healthy_segment_separability(
    config: dict, shared_labels: set[str], logger
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference = ad.read_h5ad(config["inputs"]["hgca_epithelial"], backed="r")
    latent_key = "X_scVI"
    if latent_key not in reference.obsm:
        raise RuntimeError(
            "Current HGCA epithelial object has no validated X_scVI latent; "
            "healthy segment separability was not computed."
        )
    if reference.obsm[latent_key].shape[1] < 2:
        raise RuntimeError("HGCA X_scVI latent is not a usable representation")
    obs = reference.obs
    base_keep = (
        obs["sampled_site_condition"].astype(str).eq("healthy")
        & obs["tissue_level_1"].astype(str).isin(SEGMENTS)
        & obs["hgca_celltype_v1"].astype(str).isin(shared_labels)
        & obs["donor_id"].notna()
    )
    labels = obs["hgca_celltype_v1"].astype(str)
    regions = obs["tissue_level_1"].astype(str)
    cross_compartment = (
        labels.str.contains("Colonocyte", case=False, na=False)
        & regions.isin(["duodenum", "jejunum", "ileum"])
    ) | (
        labels.str.contains("Enterocyte", case=False, na=False)
        & regions.eq("colon")
    )
    excluded_obs = obs.loc[
        base_keep & cross_compartment,
        ["hgca_celltype_v1", "tissue_level_1", "donor_id"],
    ].copy()
    excluded_obs["exclusion_reason"] = np.where(
        excluded_obs["hgca_celltype_v1"]
        .astype(str)
        .str.contains("Colonocyte", case=False, na=False),
        "Colonocyte-labelled identity in small intestine",
        "Enterocyte-labelled identity in colon",
    )
    excluded = (
        excluded_obs.groupby(
            [
                "hgca_celltype_v1",
                "tissue_level_1",
                "exclusion_reason",
            ],
            observed=True,
        )
        .agg(n_cells=("donor_id", "size"), n_donors=("donor_id", "nunique"))
        .reset_index()
    )
    keep = base_keep & ~cross_compartment
    positions = np.flatnonzero(keep.to_numpy())
    frame = obs.iloc[positions][
        ["hgca_celltype_v1", "tissue_level_1", "donor_id"]
    ].copy()
    frame["position"] = positions
    latent = np.asarray(reference.obsm[latent_key][positions], dtype=np.float32)
    if not np.isfinite(latent).all():
        raise RuntimeError("HGCA X_scVI latent contains non-finite values")
    frame["local_position"] = np.arange(len(frame))

    rng = np.random.default_rng(int(config["project"]["seed"]))
    n_repeats = 100
    repeat_rows = []
    summary_rows = []
    for identity, identity_frame in frame.groupby(
        "hgca_celltype_v1", observed=True
    ):
        unit_counts = (
            identity_frame.groupby(
                ["tissue_level_1", "donor_id"], observed=True
            )
            .size()
            .rename("n_cells")
            .reset_index()
        )
        unit_counts = unit_counts[unit_counts["n_cells"] >= 10]
        donors_by_segment = {
            segment: unit_counts.loc[
                unit_counts["tissue_level_1"] == segment, "donor_id"
            ].astype(str).tolist()
            for segment in SEGMENTS
        }
        eligible_segments = [
            segment
            for segment in SEGMENTS
            if len(donors_by_segment[segment]) >= 2
        ]
        if len(eligible_segments) < 2:
            summary_rows.append(
                {
                    "hgca_celltype_v1": identity,
                    "cache_version": HEALTHY_CACHE_VERSION,
                    "latent_key": latent_key,
                    "n_segments": len(eligible_segments),
                    "supported": False,
                    "reason": "Fewer than two segments with two donors and 10 cells per donor",
                }
            )
            continue
        donors_per_segment = min(
            6, min(len(donors_by_segment[x]) for x in eligible_segments)
        )
        identity_lookup = identity_frame.set_index(
            ["tissue_level_1", "donor_id"]
        ).sort_index()
        for repeat in range(n_repeats):
            sampled_parts = []
            for segment in eligible_segments:
                selected_donors = rng.choice(
                    donors_by_segment[segment],
                    size=donors_per_segment,
                    replace=False,
                )
                selected_units = []
                for donor_id in selected_donors:
                    unit = identity_lookup.loc[(segment, donor_id)]
                    if isinstance(unit, pd.Series):
                        unit = unit.to_frame().T
                    selected_units.append((segment, donor_id, unit))
                cells_per_unit = min(
                    25, min(len(unit) for _, _, unit in selected_units)
                )
                for segment_value, donor_id, unit in selected_units:
                    chosen = rng.choice(
                        unit["local_position"].to_numpy(int),
                        size=cells_per_unit,
                        replace=False,
                    )
                    sampled_parts.append(
                        pd.DataFrame(
                            {
                                "local_position": chosen,
                                "segment": segment_value,
                                "donor_id": donor_id,
                                "unit": f"{segment_value}::{donor_id}",
                            }
                        )
                    )
            sampled = pd.concat(sampled_parts, ignore_index=True)
            x = latent[sampled["local_position"].to_numpy(int)]
            segment = sampled["segment"].to_numpy(str)
            donor = sampled["donor_id"].to_numpy(str)
            observed_accuracy, observed_neighbor = knn_metrics(
                x, segment, donor, k=15
            )
            units = sampled[["unit", "segment"]].drop_duplicates()
            permuted_units = units.copy()
            permuted_units["segment"] = rng.permutation(
                permuted_units["segment"].to_numpy()
            )
            permuted = sampled["unit"].map(
                permuted_units.set_index("unit")["segment"]
            ).to_numpy(str)
            perm_accuracy, perm_neighbor = knn_metrics(
                x, permuted, donor, k=15
            )
            repeat_rows.append(
                {
                    "hgca_celltype_v1": identity,
                    "repeat": repeat,
                    "n_segments": len(eligible_segments),
                    "donors_per_segment": donors_per_segment,
                    "cells_per_donor_segment": cells_per_unit,
                    "balanced_accuracy": observed_accuracy,
                    "permuted_balanced_accuracy": perm_accuracy,
                    "balanced_accuracy_above_permutation": (
                        observed_accuracy - perm_accuracy
                    ),
                    "same_segment_neighbor_fraction": observed_neighbor,
                    "permuted_same_segment_neighbor_fraction": perm_neighbor,
                    "neighbor_fraction_above_permutation": (
                        observed_neighbor - perm_neighbor
                    ),
                }
            )
        identity_repeats = pd.DataFrame(repeat_rows)
        identity_repeats = identity_repeats[
            identity_repeats["hgca_celltype_v1"] == identity
        ]
        accuracy_ci = bootstrap_mean_ci(
            identity_repeats[
                "balanced_accuracy_above_permutation"
            ].to_numpy(float),
            500,
            rng,
        )
        neighbor_ci = bootstrap_mean_ci(
            identity_repeats[
                "neighbor_fraction_above_permutation"
            ].to_numpy(float),
            500,
            rng,
        )
        summary_rows.append(
            {
                "hgca_celltype_v1": identity,
                "cache_version": HEALTHY_CACHE_VERSION,
                "latent_key": latent_key,
                "n_segments": len(eligible_segments),
                "segments": ";".join(eligible_segments),
                "donors_per_segment_per_repeat": donors_per_segment,
                "n_balanced_repeats": n_repeats,
                "balanced_accuracy": identity_repeats[
                    "balanced_accuracy"
                ].mean(),
                "permuted_balanced_accuracy": identity_repeats[
                    "permuted_balanced_accuracy"
                ].mean(),
                "balanced_accuracy_above_permutation": identity_repeats[
                    "balanced_accuracy_above_permutation"
                ].mean(),
                "balanced_accuracy_above_permutation_ci_low": accuracy_ci[0],
                "balanced_accuracy_above_permutation_ci_high": accuracy_ci[1],
                "same_segment_neighbor_fraction": identity_repeats[
                    "same_segment_neighbor_fraction"
                ].mean(),
                "permuted_same_segment_neighbor_fraction": identity_repeats[
                    "permuted_same_segment_neighbor_fraction"
                ].mean(),
                "neighbor_fraction_above_permutation": identity_repeats[
                    "neighbor_fraction_above_permutation"
                ].mean(),
                "neighbor_fraction_above_permutation_ci_low": neighbor_ci[0],
                "neighbor_fraction_above_permutation_ci_high": neighbor_ci[1],
                "supported": True,
                "reason": "",
            }
        )
        logger.info(
            "Healthy separability %s: %s segments, BA above null %.3f",
            identity,
            len(eligible_segments),
            summary_rows[-1]["balanced_accuracy_above_permutation"],
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(repeat_rows), excluded


def publication_bootstrap_median(
    frame: pd.DataFrame,
    value: str,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[float, float, str]:
    if frame["publication_display"].nunique() >= 2:
        units = frame["publication_display"].unique()
        groups = {
            unit: frame.loc[
                frame["publication_display"] == unit, value
            ].to_numpy(float)
            for unit in units
        }
        boot = [
            np.median(
                np.concatenate(
                    [
                        groups[unit]
                        for unit in rng.choice(
                            units, size=len(units), replace=True
                        )
                    ]
                )
            )
            for _ in range(repeats)
        ]
        unit = "publication"
    else:
        values = frame[value].to_numpy(float)
        boot = [
            np.median(rng.choice(values, size=len(values), replace=True))
            for _ in range(repeats)
        ]
        unit = "sample"
    return (
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
        unit,
    )


def organoid_origin_retention(
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cells = pd.read_csv(
        C.DATA / "fig5b_origin_proximity_rarefied_cells.csv.gz"
    )
    qc = pd.read_csv(
        C.DATA / "per_cell_mapping_qc_flags.csv.gz",
        usecols=["cell_id", config["columns"]["confidence"]],
    )
    cells = cells.merge(qc, on="cell_id", how="left", validate="one_to_one")
    cross_compartment = (
        cells["hgca_celltype_v1"]
        .astype(str)
        .str.contains("Colonocyte", case=False, na=False)
        & cells["origin_region"].isin(["Duodenum", "Jejunum", "Ileum"])
    ) | (
        cells["hgca_celltype_v1"]
        .astype(str)
        .str.contains("Enterocyte", case=False, na=False)
        & cells["origin_region"].eq("Colon")
    )
    excluded_cells = cells[cross_compartment].copy()
    excluded_cells["exclusion_reason"] = np.where(
        excluded_cells["hgca_celltype_v1"]
        .astype(str)
        .str.contains("Colonocyte", case=False, na=False),
        "Colonocyte-labelled identity in small-intestinal organoid",
        "Enterocyte-labelled identity in colonic organoid",
    )
    excluded = (
        excluded_cells.groupby(
            [
                "hgca_celltype_v1",
                "source_standardized",
                "origin_region",
                "exclusion_reason",
            ],
            observed=True,
        )
        .agg(
            n_balanced_cells=("cell_id", "size"),
            n_samples=("sample_id", "nunique"),
            n_publications=("publication_display", "nunique"),
        )
        .reset_index()
    )
    cells = cells[~cross_compartment].copy()
    sample = (
        cells.groupby(
            [
                "sample_id",
                "publication_display",
                "source_standardized",
                "origin_region",
                "hgca_celltype_v1",
            ],
            observed=True,
        )
        .agg(
            n_balanced_cells=("cell_id", "size"),
            median_relative_origin_proximity=(
                "relative_origin_proximity",
                "median",
            ),
            fraction_positive=("relative_origin_proximity", lambda x: (x > 0).mean()),
            median_mapping_confidence=(
                config["columns"]["confidence"],
                "median",
            ),
        )
        .reset_index()
    )
    rng = np.random.default_rng(int(config["project"]["seed"]) + 1)
    rows = []
    for keys, frame in sample.groupby(
        ["hgca_celltype_v1", "source_standardized", "origin_region"],
        observed=True,
    ):
        proximity_ci = publication_bootstrap_median(
            frame,
            "median_relative_origin_proximity",
            500,
            rng,
        )
        positive_ci = publication_bootstrap_median(
            frame, "fraction_positive", 500, rng
        )
        rows.append(
            {
                "hgca_celltype_v1": keys[0],
                "source_standardized": keys[1],
                "origin_region": keys[2],
                "n_samples": frame["sample_id"].nunique(),
                "n_publications": frame["publication_display"].nunique(),
                "median_relative_origin_proximity": frame[
                    "median_relative_origin_proximity"
                ].median(),
                "median_relative_origin_proximity_ci_low": proximity_ci[0],
                "median_relative_origin_proximity_ci_high": proximity_ci[1],
                "fraction_positive": frame["fraction_positive"].median(),
                "fraction_positive_ci_low": positive_ci[0],
                "fraction_positive_ci_high": positive_ci[1],
                "median_mapping_confidence": frame[
                    "median_mapping_confidence"
                ].median(),
                "bootstrap_unit": proximity_ci[2],
                "well_supported": (
                    frame["sample_id"].nunique() >= 3
                    and frame["publication_display"].nunique() >= 2
                ),
            }
        )
    return sample, pd.DataFrame(rows), excluded


def bootstrap_slope(
    frame: pd.DataFrame,
    formula: str,
    coefficient: str,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    publications = frame["publication_display"].unique()
    values = []
    for _ in range(repeats):
        selected = rng.choice(
            publications, size=len(publications), replace=True
        )
        pieces = []
        for copy_index, publication in enumerate(selected):
            piece = frame[
                frame["publication_display"] == publication
            ].copy()
            piece["bootstrap_publication"] = (
                f"{publication}__{copy_index}"
            )
            pieces.append(piece)
        boot = pd.concat(pieces, ignore_index=True)
        try:
            fit = smf.ols(
                f"{formula} + bootstrap_publication", data=boot
            ).fit()
            if coefficient in fit.params:
                values.append(float(fit.params[coefficient]))
        except (np.linalg.LinAlgError, ValueError):
            continue
    if not values:
        return np.nan, np.nan, 0
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        len(values),
    )


def fit_publication_random_intercept(
    formula: str, frame: pd.DataFrame
) -> tuple[object, str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm(
                formula,
                data=frame,
                groups=frame["publication_display"],
            ).fit(reml=False, method="lbfgs", maxiter=500, disp=False)
        if not np.isfinite(fit.params).all():
            raise ValueError("Non-finite mixed-model coefficients")
        return fit, "Publication random-intercept linear model"
    except (np.linalg.LinAlgError, ValueError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.gee(
                formula,
                groups="publication_display",
                data=frame,
                cov_struct=Exchangeable(),
                family=sm.families.Gaussian(),
            ).fit()
        return fit, "Publication-clustered GEE fallback"


def association_models(
    joined_sample: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed + 2)
    outcome = "median_relative_origin_proximity"
    predictor = "balanced_accuracy_above_permutation"
    rows = []
    specifications = [("Overall", joined_sample)]
    specifications.extend(
        [
            (source, joined_sample[joined_sample["source_standardized"] == source])
            for source in ["ASC", "FSC", "PSC"]
        ]
    )
    for label, frame in specifications:
        frame = frame.dropna(
            subset=[
                outcome,
                predictor,
                "origin_region",
                "publication_display",
            ]
        ).copy()
        if (
            len(frame) < 12
            or frame[predictor].nunique() < 3
            or frame["publication_display"].nunique() < 2
        ):
            continue
        if label == "Overall":
            formula = (
                f"{outcome} ~ {predictor} + "
                "source_standardized + origin_region"
            )
        else:
            formula = f"{outcome} ~ {predictor} + origin_region"
        fit, model_type = fit_publication_random_intercept(
            formula, frame
        )
        ci = bootstrap_slope(
            frame, formula, predictor, 500, rng
        )
        rows.append(
            {
                "scope": label,
                "n_sample_identity_observations": len(frame),
                "n_samples": frame["sample_id"].nunique(),
                "n_publications": frame["publication_display"].nunique(),
                "coefficient": fit.params[predictor],
                "standard_error": fit.bse[predictor],
                "p_value": fit.pvalues[predictor],
                "publication_bootstrap_ci_low": ci[0],
                "publication_bootstrap_ci_high": ci[1],
                "successful_bootstraps": ci[2],
                "model": formula,
                "model_type": model_type,
            }
        )

    comparison = joined_sample[
        joined_sample["maturation_class"].isin(
            ["Shared proliferative/progenitor", "Mature identity"]
        )
    ].copy()
    comparison = comparison[
        comparison["identity_well_supported"]
    ]
    comparison_rows = []
    if (
        comparison["maturation_class"].nunique() == 2
        and comparison["publication_display"].nunique() >= 2
    ):
        comparison["maturation_class"] = pd.Categorical(
            comparison["maturation_class"],
            categories=[
                "Mature identity",
                "Shared proliferative/progenitor",
            ],
            ordered=True,
        )
        coefficient = "maturation_class[T.Shared proliferative/progenitor]"
        formula = (
            f"{outcome} ~ maturation_class + "
            "source_standardized + origin_region"
        )
        fit, model_type = fit_publication_random_intercept(
            formula, comparison
        )
        ci = bootstrap_slope(
            comparison, formula, coefficient, 500, rng
        )
        comparison_rows.append(
            {
                "contrast": "Shared proliferative/progenitor minus mature identity",
                "n_sample_identity_observations": len(comparison),
                "n_samples": comparison["sample_id"].nunique(),
                "n_publications": comparison[
                    "publication_display"
                ].nunique(),
                "coefficient": fit.params.get(coefficient, np.nan),
                "standard_error": fit.bse.get(coefficient, np.nan),
                "p_value": fit.pvalues.get(coefficient, np.nan),
                "publication_bootstrap_ci_low": ci[0],
                "publication_bootstrap_ci_high": ci[1],
                "successful_bootstraps": ci[2],
                "model_type": model_type,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(comparison_rows)


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


def build_branch_composition_tables() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    counts = pd.read_csv(
        C.DATA / "sample_subtype_counts_confident.csv", index_col=0
    )
    metadata = pd.read_csv(C.DATA / "sample_metadata.csv", index_col=0)
    metadata = metadata.reindex(counts.index)
    hierarchy = pd.read_csv(
        C.DATA / "hgca_epithelial_hierarchy.csv"
    ).set_index("hgca_celltype_v1")
    long = (
        counts.rename_axis("sample_id")
        .reset_index()
        .melt(
            id_vars="sample_id",
            var_name="hgca_celltype_v1",
            value_name="n_cells",
        )
        .merge(
            metadata[
                [
                    "publication_display",
                    "source_standardized",
                    "region_broad",
                    "time",
                    "time_class",
                    "molecular",
                    "gel",
                    "protocol",
                ]
            ].reset_index(),
            on="sample_id",
            how="left",
            validate="many_to_one",
        )
    )
    long["branch"] = long["hgca_celltype_v1"].map(
        lambda value: branch_category(value, hierarchy)
    )
    long["cross_compartment_excluded"] = cross_compartment_mask(
        long["hgca_celltype_v1"], long["region_broad"]
    )
    long.loc[long["cross_compartment_excluded"], "n_cells"] = 0
    filtered_counts = long.pivot(
        index="sample_id", columns="hgca_celltype_v1", values="n_cells"
    ).reindex(index=counts.index, columns=counts.columns, fill_value=0)
    branches = (
        long[long["branch"].isin(BRANCH_COLORS)]
        .groupby(["sample_id", "branch"], observed=True)["n_cells"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=list(BRANCH_COLORS), fill_value=0)
    )
    branch_total = branches.sum(axis=1)
    for branch in BRANCH_COLORS:
        branches[f"{branch.lower()}_fraction"] = (
            branches[branch] / branch_total.replace(0, np.nan)
        )
    branches["branch_total_cells"] = branch_total
    branches["secretory_balance"] = np.log(
        (branches["Secretory"] + 0.5)
        / np.sqrt(
            (branches["Absorptive"] + 0.5)
            * (branches["Progenitor"] + 0.5)
        )
    )
    branches = branches.join(metadata, how="left")
    branches.index.name = "sample_id"

    summary_rows = []
    for field in [
        "source_standardized",
        "region_broad",
        "time_class",
        "gel",
        "molecular",
    ]:
        for value, frame in branches.dropna(subset=[field]).groupby(
            field, observed=True
        ):
            summary_rows.append(
                {
                    "field": field,
                    "value": value,
                    "n_samples": frame.index.nunique(),
                    "n_publications": frame[
                        "publication_display"
                    ].nunique(),
                    "median_absorptive_fraction": frame[
                        "absorptive_fraction"
                    ].median(),
                    "median_secretory_fraction": frame[
                        "secretory_fraction"
                    ].median(),
                    "median_progenitor_fraction": frame[
                        "progenitor_fraction"
                    ].median(),
                    "median_secretory_balance": frame[
                        "secretory_balance"
                    ].median(),
                }
            )
    definitions = pd.DataFrame(
        [
            {
                "hgca_celltype_v1": label,
                "branch": branch_category(label, hierarchy),
                "definition_note": (
                    "Progenitor overrides lineage branch"
                    if label in PROGENITOR_IDENTITIES
                    else ""
                ),
            }
            for label in counts.columns
        ]
    )
    return (
        branches.reset_index(),
        pd.DataFrame(summary_rows),
        definitions,
        filtered_counts,
    )


def publication_bootstrap_paired_difference(
    frame: pd.DataFrame,
    metric: str,
    category: str,
    baseline: str,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    wide = frame.pivot(
        index="sample_id", columns="branch", values=metric
    )
    values = (wide[category] - wide[baseline]).dropna()
    if values.empty:
        return np.nan, np.nan, 0
    publications = (
        frame[["sample_id", "publication_display"]]
        .drop_duplicates()
        .set_index("sample_id")["publication_display"]
        .reindex(values.index)
    )
    units = publications.dropna().unique()
    boot = []
    for _ in range(repeats):
        selected = rng.choice(units, size=len(units), replace=True)
        sampled = np.concatenate(
            [
                values.loc[publications[publications == unit].index].to_numpy(
                    float
                )
                for unit in selected
            ]
        )
        boot.append(np.median(sampled))
    return (
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
        len(values),
    )


def build_branch_mapping_quality(
    metadata: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hierarchy = pd.read_csv(
        C.DATA / "hgca_epithelial_hierarchy.csv"
    ).set_index("hgca_celltype_v1")
    columns = [
        "sample_id",
        "hgca_pred_celltype_sysvi_knn",
        "hgca_pred_celltype_sysvi_knn_thresh",
        "hgca_pred_conf_sysvi_knn",
        "distance_robust_z_within_label",
        "confident",
        "distance_outlier",
        "strict_mapping_pass",
    ]
    cells = pd.read_csv(
        C.DATA / "per_cell_mapping_qc_flags.csv.gz", usecols=columns
    )
    cells = cells.merge(
        metadata[
            [
                "sample_id",
                "publication_display",
                "source_standardized",
                "region_broad",
                "time_class",
            ]
        ],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    cells["branch"] = cells["hgca_pred_celltype_sysvi_knn"].map(
        lambda value: branch_category(value, hierarchy)
    )
    cells = cells[
        cells["branch"].isin(BRANCH_COLORS)
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
                "time_class",
                "branch",
            ],
            observed=True,
            dropna=False,
        )
        .agg(
            n_raw_assignments=("branch", "size"),
            median_mapping_confidence=(
                "hgca_pred_conf_sysvi_knn",
                "median",
            ),
            fraction_confident=("confident", "mean"),
            median_reference_distance_z=(
                "distance_robust_z_within_label",
                "median",
            ),
            fraction_distance_outlier=("distance_outlier", "mean"),
            fraction_strict_mapping=("strict_mapping_pass", "mean"),
        )
        .reset_index()
    )
    sample["mapping_summary_eligible"] = (
        sample["n_raw_assignments"] >= 20
    )
    eligible = sample[sample["mapping_summary_eligible"]].copy()
    summary = (
        eligible.groupby("branch", observed=True)
        .agg(
            n_samples=("sample_id", "nunique"),
            n_publications=("publication_display", "nunique"),
            median_mapping_confidence=(
                "median_mapping_confidence",
                "median",
            ),
            median_fraction_confident=("fraction_confident", "median"),
            median_reference_distance_z=(
                "median_reference_distance_z",
                "median",
            ),
            median_fraction_distance_outlier=(
                "fraction_distance_outlier",
                "median",
            ),
            median_fraction_strict_mapping=(
                "fraction_strict_mapping",
                "median",
            ),
        )
        .reset_index()
    )
    rng = np.random.default_rng(seed + 3)
    contrast_rows = []
    for category in ["Secretory", "Progenitor"]:
        for metric in [
            "median_mapping_confidence",
            "fraction_confident",
            "median_reference_distance_z",
            "fraction_distance_outlier",
            "fraction_strict_mapping",
        ]:
            wide = eligible.pivot(
                index="sample_id", columns="branch", values=metric
            )
            difference = (
                wide[category] - wide["Absorptive"]
            ).dropna()
            ci = publication_bootstrap_paired_difference(
                eligible,
                metric,
                category,
                "Absorptive",
                500,
                rng,
            )
            contrast_rows.append(
                {
                    "contrast": f"{category} minus Absorptive",
                    "metric": metric,
                    "n_paired_samples": len(difference),
                    "median_paired_difference": difference.median(),
                    "publication_bootstrap_ci_low": ci[0],
                    "publication_bootstrap_ci_high": ci[1],
                }
            )
    confident_cells = cells[
        cells["hgca_pred_celltype_sysvi_knn_thresh"].ne("Unknown")
    ].copy()
    return sample, summary, pd.DataFrame(contrast_rows), confident_cells


def condition_definitions() -> list[dict]:
    return [
        {
            "contrast_id": "capeling_enr_vs_egf",
            "display": "Capeling\nENR − EGF\n(days 7/14)",
            "family": "Molecular",
            "publication": "Capeling_CellRep_2022",
            "test": [
                "Capeling_CellRep_2022_Day7_HIO_ENR_algenate",
                "Capeling_CellRep_2022_Day14_HIO_ENR_algenate",
            ],
            "control": [
                "Capeling_CellRep_2022_Day7_HIO_EGF_algenate",
                "Capeling_CellRep_2022_Day14_HIO_EGF_algenate",
            ],
            "verification": "Structured molecular metadata; matched culture days",
        },
        {
            "contrast_id": "yu_nrg1_vs_egf_d40",
            "display": "Yu\nNRG1 − EGF\n(day 40)",
            "family": "Molecular",
            "publication": "Yu_Cell_2021",
            "test": [
                "Yu_Cell_2021_H9_HIO_0EGF_100NRG1_D40",
                "Yu_Cell_2021_H9_HIO_1EGF_100NRG1_D40",
            ],
            "control": [
                "Yu_Cell_2021_H9_HIO_100EGF_0NRG1_D40",
                "Yu_Cell_2021_H9_HIO_100EGF_1NRG1_D40",
            ],
            "verification": "Structured molecular metadata; dose variants",
        },
        {
            "contrast_id": "yu_enr_vs_egf_d28",
            "display": "Yu\nENR − EGF\n(day 28)",
            "family": "Molecular",
            "publication": "Yu_Cell_2021",
            "test": ["Yu_Cell_2021_H9_HIO_ENR_D28"],
            "control": ["Yu_Cell_2021_H9_HIO_EGF_matrigel_D28"],
            "verification": "Structured molecular metadata; one sample per arm",
        },
        {
            "contrast_id": "kilik_nrg1_vs_egf",
            "display": "Kilik\nNRG1 − EGF",
            "family": "Molecular",
            "publication": "Kilik_BioRxiv_2021",
            "test": ["Kilik_BioRxiv_2021_UK_HIO_NRG1"],
            "control": ["Kilik_BioRxiv_2021_UK_HIO_EGF"],
            "verification": "Structured molecular metadata; one sample per arm",
        },
        {
            "contrast_id": "he_il22_pos_vs_neg",
            "display": "He\nIL-22+ − IL-22−",
            "family": "Molecular",
            "publication": "He_CellStemCell_2022",
            "test": ["He_CellStemCell_2022_hSIO_IL22pos"],
            "control": ["He_CellStemCell_2022_hSIO_IL22neg"],
            "verification": "Sample-name-defined published contrast; one sample per arm",
        },
        {
            "contrast_id": "heoca_hioec_vs_control",
            "display": "HEOCA\nHIO-EC − control",
            "family": "Co-culture",
            "publication": "HEOCA newly generated",
            "test": [
                "Thispaper_Thispaper_2023_H9_22d_tHIO_8wk_kidneycapsule_HIOEC",
                "Thispaper_Thispaper_2023_H9_22d_tHIO_8wk_mesentery_HIOEC",
            ],
            "control": [
                "Thispaper_Thispaper_2023_H9_22d_tHIO_8wk_kidneycapsule_control",
                "Thispaper_Thispaper_2023_H9_22d_tHIO_8wk_mesentery_control",
            ],
            "verification": "Structured molecular metadata; matched transplant sites",
        },
        {
            "contrast_id": "capeling_day28_vs_early",
            "display": "Capeling EGF\nday 28 − days 7/14",
            "family": "Maturation",
            "publication": "Capeling_CellRep_2022",
            "test": [
                "Capeling_CellRep_2022_h9_HIO_Suspension_EGF_28days"
            ],
            "control": [
                "Capeling_CellRep_2022_Day7_HIO_EGF_algenate",
                "Capeling_CellRep_2022_Day14_HIO_EGF_algenate",
            ],
            "verification": "Structured time metadata; one late sample",
        },
        {
            "contrast_id": "yu_week8_vs_week4",
            "display": "Yu ENR\nweek 8 − week 4",
            "family": "Maturation",
            "publication": "Yu_Cell_2021",
            "test": ["Yu_Cell_2021_H9_tHIO_WK8"],
            "control": ["Yu_Cell_2021_H9_tHIO_WK4"],
            "verification": "Structured time metadata; one sample per time",
        },
    ]


def bootstrap_two_group_effect(
    test: np.ndarray,
    control: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    if len(test) < 2 or len(control) < 2:
        return np.nan, np.nan, 0
    values = [
        rng.choice(test, size=len(test), replace=True).mean()
        - rng.choice(control, size=len(control), replace=True).mean()
        for _ in range(repeats)
    ]
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        len(values),
    )


def build_condition_contrast_tables(
    branches: pd.DataFrame,
    filtered_counts: pd.DataFrame,
    confident_cells: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    branch_frame = branches.set_index("sample_id")
    rng = np.random.default_rng(seed + 4)
    effect_rows = []
    subtype_rows = []
    n_features = filtered_counts.shape[1]
    totals = filtered_counts.sum(axis=1)
    log_abundance = np.log2(
        filtered_counts.add(0.5).div(
            totals + 0.5 * n_features, axis=0
        )
    )
    confidence = (
        confident_cells.groupby(
            ["sample_id", "hgca_pred_celltype_sysvi_knn_thresh"],
            observed=True,
        )["hgca_pred_conf_sysvi_knn"]
        .median()
        .unstack()
    )
    available = set(branch_frame.index)
    for definition in condition_definitions():
        test_ids = [x for x in definition["test"] if x in available]
        control_ids = [
            x for x in definition["control"] if x in available
        ]
        test = branch_frame.loc[test_ids, "secretory_balance"].to_numpy(
            float
        )
        control = branch_frame.loc[
            control_ids, "secretory_balance"
        ].to_numpy(float)
        ci = bootstrap_two_group_effect(test, control, 500, rng)
        test_fraction = branch_frame.loc[
            test_ids, "secretory_fraction"
        ].to_numpy(float)
        control_fraction = branch_frame.loc[
            control_ids, "secretory_fraction"
        ].to_numpy(float)
        fraction_ci = bootstrap_two_group_effect(
            test_fraction, control_fraction, 500, rng
        )
        effect_rows.append(
            {
                **{
                    key: definition[key]
                    for key in [
                        "contrast_id",
                        "display",
                        "family",
                        "publication",
                        "verification",
                    ]
                },
                "n_test_samples": len(test_ids),
                "n_control_samples": len(control_ids),
                "test_sample_ids": ";".join(test_ids),
                "control_sample_ids": ";".join(control_ids),
                "mean_test_secretory_fraction": branch_frame.loc[
                    test_ids, "secretory_fraction"
                ].mean(),
                "mean_control_secretory_fraction": branch_frame.loc[
                    control_ids, "secretory_fraction"
                ].mean(),
                "secretory_fraction_difference": (
                    test_fraction.mean() - control_fraction.mean()
                ),
                "secretory_fraction_difference_ci_low": fraction_ci[0],
                "secretory_fraction_difference_ci_high": fraction_ci[1],
                "secretory_balance_effect": test.mean() - control.mean(),
                "bootstrap_ci_low": ci[0],
                "bootstrap_ci_high": ci[1],
                "successful_bootstraps": ci[2],
                "bootstrap_interval_available": (
                    len(test_ids) >= 2 and len(control_ids) >= 2
                ),
                "independent_publication_replication": False,
                "inference_supported": False,
            }
        )
        for subtype in SECRETORY_IDENTITY_ORDER:
            if subtype not in filtered_counts:
                continue
            effect = (
                log_abundance.loc[test_ids, subtype].mean()
                - log_abundance.loc[control_ids, subtype].mean()
            )
            selected_ids = test_ids + control_ids
            subtype_rows.append(
                {
                    "contrast_id": definition["contrast_id"],
                    "display": definition["display"],
                    "family": definition["family"],
                    "hgca_celltype_v1": subtype,
                    "log2_proportion_effect": effect,
                    "n_mapped_cells": int(
                        filtered_counts.loc[selected_ids, subtype].sum()
                    ),
                    "n_samples_with_detection": int(
                        (
                            filtered_counts.loc[selected_ids, subtype] > 0
                        ).sum()
                    ),
                    "median_mapping_confidence": np.nan,
                }
            )
            values = (
                confidence.reindex(index=selected_ids, columns=[subtype])
                .iloc[:, 0]
                .dropna()
                .to_numpy(float)
            )
            if len(values):
                subtype_rows[-1]["median_mapping_confidence"] = float(
                    np.median(values)
                )
    return pd.DataFrame(effect_rows), pd.DataFrame(subtype_rows)


def build_psc_maturation_tables(
    branches: pd.DataFrame,
    filtered_counts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    branch_frame = branches.set_index("sample_id")
    best4_cells = filtered_counts[
        ["BEST4 Enterocytes", "BEST4 Colonocytes"]
    ].sum(axis=1)
    total_cells = filtered_counts.sum(axis=1)
    best4 = pd.DataFrame(
        {
            "best4_cells": best4_cells,
            "best4_fraction": best4_cells
            / total_cells.replace(0, np.nan),
            "best4_balance": np.log(
                (best4_cells + 0.5)
                / (total_cells - best4_cells + 0.5)
            ),
        }
    )
    frame = branch_frame.join(best4, how="left")
    time_order = {"≤14 d": 0, "15–55 d": 1, "≥56 d": 2}
    frame = frame[
        frame["source_standardized"].eq("PSC")
        & frame["time_class"].isin(time_order)
    ].copy()
    frame["time_ordinal"] = frame["time_class"].map(time_order).astype(float)
    summary = (
        frame.groupby("time_class", observed=True)
        .agg(
            n_samples=("best4_fraction", "size"),
            n_publications=("publication_display", "nunique"),
            median_secretory_fraction=("secretory_fraction", "median"),
            mean_secretory_fraction=("secretory_fraction", "mean"),
            median_best4_fraction=("best4_fraction", "median"),
            mean_best4_fraction=("best4_fraction", "mean"),
            best4_detection_rate=("best4_cells", lambda x: (x > 0).mean()),
        )
        .reset_index()
    )
    within = (
        frame.groupby(
            ["publication_display", "time_class"], observed=True
        )
        .agg(
            n_samples=("best4_fraction", "size"),
            median_secretory_fraction=("secretory_fraction", "median"),
            median_best4_fraction=("best4_fraction", "median"),
            total_best4_cells=("best4_cells", "sum"),
        )
        .reset_index()
    )
    varying_publications = (
        within.groupby("publication_display", observed=True)[
            "time_class"
        ]
        .nunique()
        .loc[lambda x: x >= 2]
        .index
    )
    within = within[
        within["publication_display"].isin(varying_publications)
    ].copy()

    model_rows = []
    for outcome in ["secretory_balance", "best4_balance"]:
        mixed = smf.mixedlm(
            f"{outcome} ~ time_ordinal",
            data=frame,
            groups=frame["publication_display"],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mixed_fit = mixed.fit(
                reml=False, method="lbfgs", maxiter=500, disp=False
            )
        fixed_fit = smf.ols(
            f"{outcome} ~ time_ordinal + publication_display",
            data=frame,
        ).fit(cov_type="HC3")
        model_rows.append(
            {
                "outcome": outcome,
                "n_samples": len(frame),
                "n_publications": frame["publication_display"].nunique(),
                "random_intercept_time_coefficient": mixed_fit.params[
                    "time_ordinal"
                ],
                "random_intercept_standard_error": mixed_fit.bse[
                    "time_ordinal"
                ],
                "random_intercept_p_value": mixed_fit.pvalues[
                    "time_ordinal"
                ],
                "publication_fixed_time_coefficient": fixed_fit.params[
                    "time_ordinal"
                ],
                "publication_fixed_ci_low": fixed_fit.conf_int().loc[
                    "time_ordinal", 0
                ],
                "publication_fixed_ci_high": fixed_fit.conf_int().loc[
                    "time_ordinal", 1
                ],
                "publication_fixed_p_value": fixed_fit.pvalues[
                    "time_ordinal"
                ],
                "interpretation": (
                    "Ordinal time effect after publication adjustment; "
                    "descriptive because time bins and protocols differ by study."
                ),
            }
        )
    return summary, within, pd.DataFrame(model_rows)


def best4_protocol_definitions() -> list[dict]:
    return [
        {
            "contrast_id": "charlie_ereg_vs_egf",
            "display": "Charlie EREG − EGF",
            "source": "FSC",
            "time_display": "Late",
            "test": ["Charlie_JCI_2023_hfDO_EREG"],
            "control": ["Charlie_JCI_2023_hfDO_EGF"],
        },
        {
            "contrast_id": "capeling_enr_vs_egf",
            "display": "Capeling ENR − EGF",
            "source": "PSC",
            "time_display": "Days 7/14",
            "test": [
                "Capeling_CellRep_2022_Day7_HIO_ENR_algenate",
                "Capeling_CellRep_2022_Day14_HIO_ENR_algenate",
            ],
            "control": [
                "Capeling_CellRep_2022_Day7_HIO_EGF_algenate",
                "Capeling_CellRep_2022_Day14_HIO_EGF_algenate",
            ],
        },
        {
            "contrast_id": "capeling_day28_vs_early",
            "display": "Capeling EGF day 28 − days 7/14",
            "source": "PSC",
            "time_display": "Day 28 vs 7/14",
            "test": [
                "Capeling_CellRep_2022_h9_HIO_Suspension_EGF_28days"
            ],
            "control": [
                "Capeling_CellRep_2022_Day7_HIO_EGF_algenate",
                "Capeling_CellRep_2022_Day14_HIO_EGF_algenate",
            ],
        },
        {
            "contrast_id": "yu_nrg1_vs_egf_d40",
            "display": "Yu NRG1 − EGF, day 40",
            "source": "PSC",
            "time_display": "Day 40",
            "test": [
                "Yu_Cell_2021_H9_HIO_0EGF_100NRG1_D40",
                "Yu_Cell_2021_H9_HIO_1EGF_100NRG1_D40",
            ],
            "control": [
                "Yu_Cell_2021_H9_HIO_100EGF_0NRG1_D40",
                "Yu_Cell_2021_H9_HIO_100EGF_1NRG1_D40",
            ],
        },
        {
            "contrast_id": "yu_enr_vs_egf_d28",
            "display": "Yu ENR − EGF, day 28",
            "source": "PSC",
            "time_display": "Day 28",
            "test": ["Yu_Cell_2021_H9_HIO_ENR_D28"],
            "control": ["Yu_Cell_2021_H9_HIO_EGF_matrigel_D28"],
        },
        {
            "contrast_id": "kilik_nrg1_vs_egf",
            "display": "Kilik NRG1 − EGF",
            "source": "PSC",
            "time_display": "Day 77",
            "test": ["Kilik_BioRxiv_2021_UK_HIO_NRG1"],
            "control": ["Kilik_BioRxiv_2021_UK_HIO_EGF"],
        },
        {
            "contrast_id": "he_il22_pos_vs_neg",
            "display": "He IL-22+ − IL-22−",
            "source": "ASC",
            "time_display": "Not reported",
            "test": ["He_CellStemCell_2022_hSIO_IL22pos"],
            "control": ["He_CellStemCell_2022_hSIO_IL22neg"],
        },
    ]


def build_best4_protocol_response(
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts = pd.read_csv(
        C.DATA / "sample_subtype_counts_confident.csv", index_col=0
    )
    metadata = pd.read_csv(C.DATA / "sample_metadata.csv", index_col=0)
    hierarchy = pd.read_csv(
        C.DATA / "hgca_epithelial_hierarchy.csv"
    ).set_index("hgca_celltype_v1")
    definitions = best4_protocol_definitions()
    selected_ids = list(
        dict.fromkeys(
            sample_id
            for definition in definitions
            for arm in ["test", "control"]
            for sample_id in definition[arm]
        )
    )
    missing = sorted(set(selected_ids) - set(counts.index))
    if missing:
        raise RuntimeError(
            "BEST4 protocol samples absent from composition table: "
            + ", ".join(missing)
        )
    category = {
        label: branch_category(label, hierarchy)
        for label in counts.columns
    }
    best4_enterocyte = "BEST4 Enterocytes"
    best4_colonocyte = "BEST4 Colonocytes"
    mature_absorptive = [
        label
        for label, branch in category.items()
        if branch == "Absorptive"
        and label not in {best4_enterocyte, best4_colonocyte}
    ]
    secretory = [
        label for label, branch in category.items() if branch == "Secretory"
    ]
    progenitor = [
        label for label, branch in category.items() if branch == "Progenitor"
    ]
    rng = np.random.default_rng(seed + 5)
    sample_metric_rows = []
    rarefaction_depth = 200
    rarefaction_repeats = 100
    for sample_id in selected_ids:
        sample_counts = counts.loc[sample_id].to_numpy(int)
        if sample_counts.sum() < rarefaction_depth:
            raise RuntimeError(
                f"{sample_id} has fewer than {rarefaction_depth} confident cells"
            )
        draws = np.vstack(
            [
                rng.multivariate_hypergeometric(
                    sample_counts, rarefaction_depth
                )
                for _ in range(rarefaction_repeats)
            ]
        )
        draw_frame = pd.DataFrame(draws, columns=counts.columns)
        region = str(metadata.loc[sample_id, "region_broad"])
        invalid_mature = [
            label
            for label in mature_absorptive
            if (
                "Colonocyte" in label
                and region in {"Duodenum", "Jejunum", "Ileum"}
            )
            or ("Enterocyte" in label and region == "Colon")
        ]
        valid_mature = [
            label for label in mature_absorptive if label not in invalid_mature
        ]
        invalid_progenitor = [
            label
            for label in progenitor
            if (
                "Colonocyte" in label
                and region in {"Duodenum", "Jejunum", "Ileum"}
            )
            or ("Enterocyte" in label and region == "Colon")
        ]
        valid_progenitor = [
            label for label in progenitor if label not in invalid_progenitor
        ]
        metric_values = {
            "total_best4": 100
            * draw_frame[[best4_enterocyte, best4_colonocyte]]
            .sum(axis=1)
            .mean()
            / rarefaction_depth,
            "best4_enterocytes": 100
            * draw_frame[best4_enterocyte].mean()
            / rarefaction_depth,
            "best4_colonocytes": 100
            * draw_frame[best4_colonocyte].mean()
            / rarefaction_depth,
            "mature_absorptive": 100
            * draw_frame[valid_mature].sum(axis=1).mean()
            / rarefaction_depth,
            "differentiated_secretory": 100
            * draw_frame[secretory].sum(axis=1).mean()
            / rarefaction_depth,
            "progenitor": 100
            * draw_frame[valid_progenitor].sum(axis=1).mean()
            / rarefaction_depth,
        }
        for metric, value in metric_values.items():
            sample_metric_rows.append(
                {
                    "sample_id": sample_id,
                    "metric": metric,
                    "value": value,
                    "unit": "percent of rarefied confident mappings",
                    "rarefaction_depth": rarefaction_depth,
                    "rarefaction_repeats": rarefaction_repeats,
                }
            )

    qc = pd.read_csv(
        C.DATA / "per_cell_mapping_qc_flags.csv.gz",
        usecols=[
            "sample_id",
            "hgca_pred_celltype_sysvi_knn_thresh",
            "hgca_pred_conf_sysvi_knn",
            "d_knn_mean",
        ],
    )
    qc = qc[
        qc["sample_id"].isin(selected_ids)
        & qc["hgca_pred_celltype_sysvi_knn_thresh"].isin(
            [best4_enterocyte, best4_colonocyte]
        )
    ]
    qc_summary = (
        qc.groupby("sample_id", observed=True)
        .agg(
            median_best4_mapping_confidence=(
                "hgca_pred_conf_sysvi_knn",
                "median",
            ),
            median_best4_reference_distance=("d_knn_mean", "median"),
            n_confident_best4_cells=(
                "hgca_pred_celltype_sysvi_knn_thresh",
                "size",
            ),
        )
        .reindex(selected_ids)
    )
    for sample_id, row in qc_summary.iterrows():
        sample_metric_rows.extend(
            [
                {
                    "sample_id": sample_id,
                    "metric": "median_mapping_confidence",
                    "value": 100 * row["median_best4_mapping_confidence"],
                    "unit": "confidence percentage points",
                    "rarefaction_depth": np.nan,
                    "rarefaction_repeats": np.nan,
                },
                {
                    "sample_id": sample_id,
                    "metric": "reference_distance",
                    "value": row["median_best4_reference_distance"],
                    "unit": "median sysVI k-nearest-reference distance",
                    "rarefaction_depth": np.nan,
                    "rarefaction_repeats": np.nan,
                },
            ]
        )
    sample_metrics = pd.DataFrame(sample_metric_rows)
    sample_metrics = sample_metrics.merge(
        metadata[
            [
                "publication_display",
                "source_standardized",
                "region_broad",
                "time",
                "time_class",
            ]
        ].reset_index(),
        on="sample_id",
        how="left",
        validate="many_to_one",
    )

    effect_rows = []
    raw_rows = []
    for definition in definitions:
        for arm in ["test", "control"]:
            for sample_id in definition[arm]:
                current = sample_metrics[
                    sample_metrics["sample_id"].eq(sample_id)
                ].copy()
                current["contrast_id"] = definition["contrast_id"]
                current["contrast_display"] = definition["display"]
                current["arm"] = arm
                raw_rows.append(current)
        test = sample_metrics[
            sample_metrics["sample_id"].isin(definition["test"])
        ]
        control = sample_metrics[
            sample_metrics["sample_id"].isin(definition["control"])
        ]
        sidecar_ids = definition["test"] + definition["control"]
        sidecar = metadata.loc[sidecar_ids]
        region_values = sorted(
            sidecar["region_broad"].dropna().astype(str).unique()
        )
        derive_original_values = sorted(
            sidecar["derive_original"].dropna().astype(str).unique()
        )
        for metric in sample_metrics["metric"].unique():
            test_values = test.loc[test["metric"].eq(metric), "value"]
            control_values = control.loc[
                control["metric"].eq(metric), "value"
            ]
            effect_rows.append(
                {
                    "contrast_id": definition["contrast_id"],
                    "contrast_display": definition["display"],
                    "source_standardized": definition["source"],
                    "derive_original": "/".join(derive_original_values),
                    "region": "/".join(region_values) or "Not reported",
                    "time": definition["time_display"],
                    "n_test_samples": len(definition["test"]),
                    "n_control_samples": len(definition["control"]),
                    "replicated_both_arms": (
                        len(definition["test"]) >= 2
                        and len(definition["control"]) >= 2
                    ),
                    "metric": metric,
                    "test_mean": test_values.mean(),
                    "control_mean": control_values.mean(),
                    "effect": test_values.mean() - control_values.mean(),
                    "unit": test.loc[
                        test["metric"].eq(metric), "unit"
                    ].iloc[0],
                    "best4_colonocyte_calls_retained": True,
                }
            )
    return pd.DataFrame(effect_rows), pd.concat(raw_rows), qc_summary.reset_index()


def render_best4_protocol_response(
    effects: pd.DataFrame,
    raw: pd.DataFrame,
) -> None:
    configure_style()
    metric_order = [
        "total_best4",
        "best4_enterocytes",
        "best4_colonocytes",
        "mature_absorptive",
        "differentiated_secretory",
        "progenitor",
        "median_mapping_confidence",
        "reference_distance",
    ]
    metric_labels = {
        "total_best4": "Total\nBEST4",
        "best4_enterocytes": "BEST4\nenterocytes",
        "best4_colonocytes": "BEST4\ncolonocytes",
        "mature_absorptive": "Mature\nabsorptive",
        "differentiated_secretory": "Differentiated\nsecretory",
        "progenitor": "Progenitor",
        "median_mapping_confidence": "Median mapping\nconfidence",
        "reference_distance": "Reference\ndistance",
    }
    contrast_order = [
        definition["contrast_id"]
        for definition in best4_protocol_definitions()
    ]
    display = {
        definition["contrast_id"]: definition["display"]
        for definition in best4_protocol_definitions()
    }
    matrix = effects.pivot(
        index="contrast_id", columns="metric", values="effect"
    ).reindex(index=contrast_order, columns=metric_order)
    composition_values = matrix.iloc[:, :7].to_numpy(float)
    composition_limit = max(
        5.0, float(np.nanquantile(np.abs(composition_values), 0.95))
    )
    distance_values = matrix["reference_distance"].to_numpy(float)
    distance_limit = max(
        0.1, float(np.nanquantile(np.abs(distance_values), 0.95))
    )
    composition_norm = matplotlib.colors.TwoSlopeNorm(
        vmin=-composition_limit, vcenter=0, vmax=composition_limit
    )
    distance_norm = matplotlib.colors.TwoSlopeNorm(
        vmin=-distance_limit, vcenter=0, vmax=distance_limit
    )
    composition_cmap = sns.color_palette("vlag", as_cmap=True)
    distance_cmap = sns.color_palette("BrBG_r", as_cmap=True)

    fig, ax = plt.subplots(figsize=(180 * MM, 98 * MM))
    row_lookup = {
        contrast: len(contrast_order) - 1 - index
        for index, contrast in enumerate(contrast_order)
    }
    raw_ranges = (
        raw.groupby("metric", observed=True)["value"]
        .agg(["min", "max"])
        .to_dict("index")
    )
    for contrast in contrast_order:
        y = row_lookup[contrast]
        row = effects[effects["contrast_id"].eq(contrast)].iloc[0]
        replicated = bool(row["replicated_both_arms"])
        for x, metric in enumerate(metric_order):
            effect = matrix.loc[contrast, metric]
            if metric == "reference_distance":
                facecolor = distance_cmap(distance_norm(effect))
            else:
                facecolor = composition_cmap(composition_norm(effect))
            ax.add_patch(
                matplotlib.patches.Rectangle(
                    (x - 0.5, y - 0.45),
                    1,
                    0.9,
                    facecolor=facecolor,
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=0,
                )
            )
            points = raw[
                raw["contrast_id"].eq(contrast)
                & raw["metric"].eq(metric)
            ]
            limits = raw_ranges[metric]
            value_range = limits["max"] - limits["min"]
            for arm, arm_frame in points.groupby("arm", observed=True):
                x_center = x + (0.16 if arm == "test" else -0.16)
                offsets = np.linspace(
                    -0.045,
                    0.045,
                    max(1, len(arm_frame)),
                )
                for offset, (_, point) in zip(
                    offsets, arm_frame.iterrows()
                ):
                    normalized = (
                        0.5
                        if value_range == 0
                        else (point["value"] - limits["min"]) / value_range
                    )
                    point_y = y - 0.28 + 0.56 * normalized
                    ax.scatter(
                        x_center + offset,
                        point_y,
                        s=5,
                        marker="o",
                        facecolor=(
                            "#222222" if arm == "test" else "white"
                        ),
                        edgecolor="#222222",
                        linewidth=0.35,
                        zorder=3,
                    )
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (-0.5, y - 0.45),
                len(metric_order),
                0.9,
                fill=False,
                edgecolor="#222222" if replicated else "#777777",
                linewidth=0.8 if replicated else 0.55,
                linestyle="-" if replicated else "--",
                zorder=4,
            )
        )
        ax.text(
            len(metric_order) + 0.05,
            y,
            (
                f"{row['source_standardized']} ({row['derive_original']})"
                if str(row["derive_original"])
                not in {"", str(row["source_standardized"])}
                else str(row["source_standardized"])
            ),
            color={
                "ASC": "#0072B2",
                "FSC": "#D55E00",
                "PSC": "#CC79A7",
            }.get(str(row["source_standardized"]), "#333333"),
            va="center",
            fontsize=5,
        )
        ax.text(
            len(metric_order) + 1.2,
            y,
            str(row["region"]),
            va="center",
            fontsize=5,
        )
        ax.text(
            len(metric_order) + 2.45,
            y,
            str(row["time"]),
            va="center",
            fontsize=5,
        )
        ax.text(
            len(metric_order) + 3.75,
            y,
            f"{int(row['n_test_samples'])}/{int(row['n_control_samples'])}",
            va="center",
            ha="center",
            fontsize=5,
        )
    ax.axvline(5.5, color="white", lw=1.2)
    ax.axvline(6.5, color="#555555", lw=0.7)
    ax.set_xlim(-0.55, len(metric_order) + 4.25)
    ax.set_ylim(-0.55, len(contrast_order) - 0.45)
    ax.set_xticks(
        range(len(metric_order)),
        [metric_labels[metric] for metric in metric_order],
    )
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_yticks(
        range(len(contrast_order)),
        [display[value] for value in contrast_order[::-1]],
    )
    ax.tick_params(axis="y", length=0)
    sidecar_headers = [
        (len(metric_order) + 0.05, "Source"),
        (len(metric_order) + 1.2, "Segment"),
        (len(metric_order) + 2.45, "Time"),
        (len(metric_order) + 3.75, "n test/control"),
    ]
    for x, label in sidecar_headers:
        ax.text(
            x,
            len(contrast_order) - 0.28,
            label,
            ha="left" if label != "n test/control" else "center",
            va="bottom",
            fontsize=5,
            fontweight="bold",
        )
    ax.text(
        -0.08,
        1.12,
        "e",
        transform=ax.transAxes,
        fontsize=7,
        fontweight="bold",
    )
    ax.set_title(
        "Within-study protocol responses separate BEST4 abundance from mapping quality",
        loc="left",
        fontsize=7,
        fontweight="bold",
        pad=26,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    composition_mappable = plt.cm.ScalarMappable(
        norm=composition_norm, cmap=composition_cmap
    )
    distance_mappable = plt.cm.ScalarMappable(
        norm=distance_norm, cmap=distance_cmap
    )
    composition_bar = fig.colorbar(
        composition_mappable,
        ax=ax,
        orientation="horizontal",
        fraction=0.045,
        pad=0.16,
        aspect=30,
    )
    composition_bar.set_label(
        "Difference in composition or confidence (percentage points)",
        fontsize=5,
    )
    composition_bar.ax.tick_params(labelsize=4)
    distance_bar = fig.colorbar(
        distance_mappable,
        ax=ax,
        orientation="horizontal",
        fraction=0.045,
        pad=0.04,
        aspect=30,
    )
    distance_bar.set_label(
        "Difference in median reference distance (test − comparator)",
        fontsize=5,
    )
    distance_bar.ax.tick_params(labelsize=4)
    fig.text(
        0.18,
        0.09,
        "Filled points: test samples; open points: comparator samples. "
        "Solid outline: ≥2 samples in both arms; dashed: a one-sample arm.\n"
        "BEST4-colonocyte calls are retained to expose segment mismatch. "
        "ASC, adult stem cell-derived; FSC, fetal stem cell-derived; "
        "PSC (IPS), pluripotent/iPSC-derived.",
        fontsize=4.5,
    )
    fig.subplots_adjust(
        left=0.18, right=0.98, top=0.8, bottom=0.34
    )
    for extension in ("pdf", "svg", "png"):
        fig.savefig(
            C.OUT / f"fig5_h_e_best4_protocol_response.{extension}",
            dpi=300,
            facecolor="white",
        )
    plt.close(fig)


def parse_numeric_culture_day(values: pd.Series) -> pd.Series:
    extracted = values.astype(str).str.extract(
        r"(?i)^\s*(\d+(?:\.\d+)?)\s*day\s*$"
    )[0]
    return pd.to_numeric(extracted, errors="coerce")


def build_source_maturation_distance_table() -> pd.DataFrame:
    frame = pd.read_csv(C.DATA / "supp_mapping_qc_by_sample.csv")
    frame["day"] = parse_numeric_culture_day(frame["time"])
    frame["maturation_display"] = "Not reported"
    frame.loc[
        frame["source_standardized"].eq("FSC")
        & frame["time"].astype(str).str.lower().eq("early"),
        "maturation_display",
    ] = "Reported early"
    frame.loc[
        frame["source_standardized"].eq("FSC")
        & frame["time"].astype(str).str.lower().eq("late"),
        "maturation_display",
    ] = "Reported late"
    frame.loc[
        frame["source_standardized"].eq("PSC") & frame["day"].le(14),
        "maturation_display",
    ] = "PSC ≤14 d"
    frame.loc[
        frame["source_standardized"].eq("PSC")
        & frame["day"].between(15, 55),
        "maturation_display",
    ] = "PSC 15–55 d"
    frame.loc[
        frame["source_standardized"].eq("PSC") & frame["day"].ge(56),
        "maturation_display",
    ] = "PSC ≥56 d"
    frame["maturation_metadata_type"] = np.select(
        [
            frame["source_standardized"].eq("PSC")
            & frame["day"].notna(),
            frame["source_standardized"].eq("FSC")
            & frame["time"].notna(),
        ],
        ["Numeric day", "Qualitative early/late"],
        default="Not reported",
    )
    return frame[
        [
            "sample_id",
            "publication_display",
            "source_standardized",
            "region_broad",
            "time",
            "day",
            "maturation_display",
            "maturation_metadata_type",
            "median_nn_distance",
            "n_cells",
        ]
    ].copy()


def build_maturation_reference_distance_tables() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    metadata = pd.read_csv(C.DATA / "sample_metadata.csv")
    distance = pd.read_csv(C.DATA / "supp_mapping_qc_by_sample.csv")
    distance["day"] = parse_numeric_culture_day(distance["time"])
    numeric_psc = distance[
        distance["source_standardized"].eq("PSC")
        & distance["day"].notna()
    ].copy()
    time_counts = numeric_psc.groupby(
        "publication_display", observed=True
    )["day"].nunique()
    varying_publications = time_counts[time_counts >= 2].index
    samples = numeric_psc.copy()
    samples["log1p_day"] = np.log1p(samples["day"])
    samples["publication_has_multiple_numeric_days"] = samples[
        "publication_display"
    ].isin(varying_publications)
    samples["model_included"] = samples[
        "publication_has_multiple_numeric_days"
    ]
    samples["model_exclusion_reason"] = np.where(
        samples["model_included"],
        "",
        "Publication contains only one numeric maturation day",
    )
    samples["within_publication_log1p_day"] = np.nan
    samples["within_publication_distance"] = np.nan
    model_mask = samples["model_included"]
    samples.loc[
        model_mask, "within_publication_log1p_day"
    ] = samples.loc[model_mask, "log1p_day"] - samples.loc[
        model_mask
    ].groupby("publication_display")["log1p_day"].transform("mean")
    samples.loc[
        model_mask, "within_publication_distance"
    ] = samples.loc[model_mask, "median_nn_distance"] - samples.loc[
        model_mask
    ].groupby("publication_display")[
        "median_nn_distance"
    ].transform("mean")
    samples = samples[
        [
            "sample_id",
            "publication_display",
            "source_standardized",
            "region_broad",
            "time",
            "time_class",
            "day",
            "log1p_day",
            "median_nn_distance",
            "n_cells",
            "within_publication_log1p_day",
            "within_publication_distance",
            "publication_has_multiple_numeric_days",
            "model_included",
            "model_exclusion_reason",
        ]
    ].sort_values(["publication_display", "day", "sample_id"])
    if len(samples) != 41 or samples["publication_display"].nunique() != 9:
        raise RuntimeError(
            "Unexpected numeric-time denominator: expected 41 PSC samples "
            "from nine publications"
        )
    model_samples = samples[samples["model_included"]].copy()
    if (
        len(model_samples) != 31
        or model_samples["publication_display"].nunique() != 5
    ):
        raise RuntimeError(
            "Unexpected maturation-distance denominator: expected 31 samples "
            "from five time-varying publications"
        )
    across_fit = smf.ols(
        "median_nn_distance ~ log1p_day",
        data=samples,
    ).fit(cov_type="HC3")
    across_confidence = across_fit.conf_int().loc["log1p_day"]
    fit = smf.ols(
        "median_nn_distance ~ log1p_day + publication_display",
        data=model_samples,
    ).fit(cov_type="HC3")
    coefficient = float(fit.params["log1p_day"])
    confidence = fit.conf_int().loc["log1p_day"]
    model_rows = [
        {
            "model_type": "across_study_ols_hc3",
            "scope": "all_numeric_time_psc_samples",
            "formula": "median_nn_distance ~ log1p(day)",
            "term": "log1p_day",
            "n_samples": len(samples),
            "n_publications": samples["publication_display"].nunique(),
            "coefficient": across_fit.params["log1p_day"],
            "standard_error": across_fit.bse["log1p_day"],
            "ci_low": across_confidence.iloc[0],
            "ci_high": across_confidence.iloc[1],
            "p_value": across_fit.pvalues["log1p_day"],
            "all_publication_specific_slopes_negative": np.nan,
            "descriptive_median": np.nan,
        },
        {
            "model_type": "within_study_publication_adjusted_ols_hc3",
            "scope": "all_time_varying_publications",
            "formula": (
                "median_nn_distance ~ log1p(day) + "
                "C(publication_display)"
            ),
            "term": "log1p_day",
            "n_samples": len(model_samples),
            "n_publications": model_samples[
                "publication_display"
            ].nunique(),
            "coefficient": coefficient,
            "standard_error": fit.bse["log1p_day"],
            "ci_low": confidence.iloc[0],
            "ci_high": confidence.iloc[1],
            "p_value": fit.pvalues["log1p_day"],
            "all_publication_specific_slopes_negative": True,
            "descriptive_median": np.nan,
        }
    ]
    publication_slopes = []
    for publication, frame in model_samples.groupby(
        "publication_display", observed=True
    ):
        publication_fit = smf.ols(
            "median_nn_distance ~ log1p_day", data=frame
        ).fit()
        slope = float(publication_fit.params["log1p_day"])
        publication_slopes.append(slope)
        if len(frame) > 2:
            publication_se = publication_fit.bse["log1p_day"]
            publication_ci = publication_fit.conf_int().loc[
                "log1p_day"
            ]
            publication_p = publication_fit.pvalues["log1p_day"]
        else:
            publication_se = np.nan
            publication_ci = pd.Series([np.nan, np.nan])
            publication_p = np.nan
        model_rows.append(
            {
                "model_type": "publication_specific_descriptive_slope",
                "scope": publication,
                "formula": "median_nn_distance ~ log1p(day)",
                "term": "log1p_day",
                "n_samples": len(frame),
                "n_publications": 1,
                "coefficient": slope,
                "standard_error": publication_se,
                "ci_low": publication_ci.iloc[0],
                "ci_high": publication_ci.iloc[1],
                "p_value": publication_p,
                "all_publication_specific_slopes_negative": np.nan,
                "descriptive_median": np.nan,
            }
        )
    if not all(slope < 0 for slope in publication_slopes):
        raise RuntimeError(
            "Expected all five publication-specific maturation slopes to be negative"
        )
    for time_class in ["≤14 d", "15–55 d", "≥56 d"]:
        frame = samples[samples["time_class"].eq(time_class)]
        model_rows.append(
            {
                "model_type": "descriptive_time_class_median",
                "scope": time_class,
                "formula": "",
                "term": "median_nn_distance",
                "n_samples": len(frame),
                "n_publications": frame["publication_display"].nunique(),
                "coefficient": np.nan,
                "standard_error": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_value": np.nan,
                "all_publication_specific_slopes_negative": np.nan,
                "descriptive_median": frame[
                    "median_nn_distance"
                ].median(),
            }
        )

    origin = pd.read_csv(
        C.DATA / "fig5g_organoid_origin_proximity_by_sample.csv"
    )
    origin_sample = (
        origin.groupby("sample_id", observed=True)
        .agg(
            median_relative_origin_proximity=(
                "median_relative_origin_proximity",
                "median",
            ),
            n_evaluable_subtypes=("hgca_celltype_v1", "nunique"),
            n_balanced_cells=("n_balanced_cells", "sum"),
        )
        .reset_index()
    )
    nearest = pd.read_csv(
        C.DATA / "fig5b_sample_nearest_region_proportions.csv"
    )
    coverage = samples[
        [
            "sample_id",
            "publication_display",
            "source_standardized",
            "region_broad",
            "time",
            "time_class",
            "day",
            "log1p_day",
        ]
    ].merge(
        origin_sample,
        on="sample_id",
        how="left",
        validate="one_to_one",
    ).merge(
        nearest,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    coverage["declared_segment_specific"] = coverage[
        "region_broad"
    ].isin(["Duodenum", "Jejunum", "Ileum", "Colon"])
    coverage["origin_proximity_available"] = coverage[
        "median_relative_origin_proximity"
    ].notna()
    coverage["regression_estimable"] = False
    coverage["coverage_limit"] = np.where(
        coverage["origin_proximity_available"],
        "Numeric time and segment-of-origin proximity available",
        (
            "Declared region is nonspecific 'Intestine'; "
            "segment-of-origin proximity is undefined"
        ),
    )
    coverage["nearest_segment_profile_available"] = coverage[
        ["Duodenum", "Jejunum", "Ileum", "Colon"]
    ].notna().all(axis=1)
    if (
        coverage["origin_proximity_available"].sum() != 4
        or coverage.loc[
            coverage["origin_proximity_available"],
            "publication_display",
        ].nunique()
        != 2
    ):
        raise RuntimeError(
            "Unexpected origin-time denominator: expected four samples "
            "from two publications"
        )
    if len(coverage) != 41 or not coverage[
        "nearest_segment_profile_available"
    ].all():
        raise RuntimeError(
            "Expected nearest-segment profiles for all 41 numeric-time samples"
        )
    return samples, pd.DataFrame(model_rows), coverage


def short_publication_label(value: str) -> str:
    parts = str(value).split("_")
    year = next(
        (part for part in parts if part.isdigit() and len(part) == 4),
        "",
    )
    return " ".join([parts[0], year]).strip()


def render_maturation_reference_distance(
    samples: pd.DataFrame,
    models: pd.DataFrame,
    origin: pd.DataFrame,
) -> None:
    configure_style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(180 * MM, 76 * MM),
        gridspec_kw={
            "wspace": 0.48,
            "width_ratios": [1.15, 1.0, 0.9],
        },
    )
    time_palette = {
        key: TIME_COLORS[key]
        for key in ["≤14 d", "15–55 d", "≥56 d"]
    }
    tick_days = np.array([0, 3, 7, 14, 28, 56, 98], dtype=float)
    tick_positions = np.log1p(tick_days)

    ax = axes[0]
    for time_class in ["≤14 d", "15–55 d", "≥56 d"]:
        frame = samples[samples["time_class"].eq(time_class)]
        included = frame[frame["model_included"]]
        excluded = frame[~frame["model_included"]]
        ax.scatter(
            included["log1p_day"],
            included["median_nn_distance"],
            s=15,
            color=time_palette[time_class],
            edgecolor="white",
            linewidth=0.3,
            label=time_class,
            zorder=3,
        )
        ax.scatter(
            excluded["log1p_day"],
            excluded["median_nn_distance"],
            s=15,
            facecolor="white",
            edgecolor=time_palette[time_class],
            linewidth=0.75,
            zorder=3,
        )
    model = models[
        models["model_type"].eq("publication_adjusted_ols_hc3")
    ].iloc[0]
    model_samples = samples[samples["model_included"]]
    grid = np.linspace(
        model_samples["log1p_day"].min(),
        model_samples["log1p_day"].max(),
        200,
    )
    x_center = model_samples["log1p_day"].mean()
    y_center = model_samples["median_nn_distance"].mean()
    fitted = y_center + model["coefficient"] * (grid - x_center)
    uncertainty = 1.96 * model["standard_error"] * np.abs(
        grid - x_center
    )
    ax.plot(grid, fitted, color="#222222", lw=0.9, zorder=4)
    ax.fill_between(
        grid,
        fitted - uncertainty,
        fitted + uncertainty,
        color="#777777",
        alpha=0.16,
        linewidth=0,
        zorder=2,
    )
    ax.set_xticks(tick_positions, [str(int(day)) for day in tick_days])
    ax.set_xlabel("Numeric maturation day (log1p scale)")
    ax.set_ylabel(
        "Sample median nearest-HGCA-reference distance\n"
        "(lower = closer to adult HGCA)"
    )
    ax.set_title(
        "Overall nearest-reference distance",
        fontweight="bold",
    )
    ax.text(
        0.98,
        0.98,
        "All numeric-time PSC: n=41 samples, 9 publications\n"
        "Model: n=31 samples; 5 time-varying publications\n"
        f"β={model['coefficient']:.2f} "
        f"[{model['ci_low']:.2f}, {model['ci_high']:.2f}]\n"
        f"HC3 P={model['p_value']:.2g}; all 5 slopes <0\n"
        "Medians: ≤14 d 4.27; 15–55 d 2.22; ≥56 d 2.00",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=4.5,
    )
    sns.despine(ax=ax)
    ax.text(
        -0.16,
        1.05,
        "f",
        transform=ax.transAxes,
        fontsize=7,
        fontweight="bold",
    )

    ax = axes[1]
    region_order = ["Duodenum", "Jejunum", "Ileum", "Colon"]
    region_colors = {
        "Duodenum": SEGMENT_COLORS["Duodenum"],
        "Jejunum": SEGMENT_COLORS["Jejunum"],
        "Ileum": SEGMENT_COLORS["Ileum"],
        "Colon": SEGMENT_COLORS["Colon"],
    }
    day_offsets = {}
    for day, frame in origin.groupby("day", observed=True):
        ordered_ids = sorted(frame["sample_id"].astype(str))
        offsets = np.linspace(-0.035, 0.035, len(ordered_ids))
        day_offsets.update(
            {
                sample_id: offset
                for sample_id, offset in zip(ordered_ids, offsets)
            }
        )
    for region_index, region in enumerate(region_order):
        fractions = origin[region].to_numpy(float)
        x_values = origin["log1p_day"].to_numpy(float) + origin[
            "sample_id"
        ].map(day_offsets).to_numpy(float)
        ax.scatter(
            x_values,
            np.full(len(origin), len(region_order) - 1 - region_index),
            s=4 + 52 * fractions,
            color=region_colors[region],
            edgecolor="white",
            linewidth=0.2,
            alpha=0.45,
        )
    ax.set_yticks(
        range(len(region_order)),
        region_order[::-1],
    )
    ax.set_xticks(tick_positions, [str(int(day)) for day in tick_days])
    ax.set_xlabel("Numeric maturation day (log1p scale)")
    ax.set_ylabel("Nearest HGCA segment")
    ax.set_title(
        "Nearest-segment profile\n(origin-independent)",
        fontweight="bold",
    )
    ax.text(
        0.03,
        0.97,
        'n=41 samples, 9 publications\n37 declared only as "Intestine"\n'
        "Bubble area = within-sample fraction (25–100%)",
        transform=ax.transAxes,
        va="top",
        fontsize=4.5,
    )
    sns.despine(ax=ax)

    ax = axes[2]
    origin_available = origin[origin["origin_proximity_available"]].copy()
    for time_class, frame in origin_available.groupby(
        "time_class", observed=True
    ):
        ax.scatter(
            frame["log1p_day"],
            frame["median_relative_origin_proximity"],
            s=18,
            color=time_palette[time_class],
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
    ax.axhline(0, color="#777777", lw=0.55)
    labels = []
    for _, row in origin_available.iterrows():
        labels.append(
            ax.text(
                row["log1p_day"],
                row["median_relative_origin_proximity"],
                (
                    f"{short_publication_label(row['publication_display'])}"
                    f"\n{row['source_standardized']} · "
                    f"{row['region_broad']}"
                ),
                fontsize=3.9,
                color="#444444",
                ha="right",
            )
        )
    adjust_text(
        labels,
        ax=ax,
        force_text=(0.35, 0.6),
        expand=(1.05, 1.12),
        max_move=12,
        ensure_inside_axes=True,
    )
    ax.set_xticks(tick_positions, [str(int(day)) for day in tick_days])
    ax.set_xlabel("Numeric maturation day (log1p scale)")
    ax.set_ylabel(
        "Median relative origin proximity\n"
        "(positive = declared origin is closer)"
    )
    ax.set_title(
        "Segment-of-origin proximity",
        fontweight="bold",
    )
    ax.text(
        0.03,
        0.97,
        "Insufficient numeric-time and segment coverage:\n"
        "n=4 samples, 2 publications.\n"
        "Not estimable; 37/41 lack a specific declared segment.",
        transform=ax.transAxes,
        fontsize=4.7,
        va="top",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.9,
            "pad": 1.0,
        },
    )
    sns.despine(ax=ax)
    fig.suptitle(
        "PSC-organoid maturation time and adult-reference correspondence",
        x=0.08,
        y=0.99,
        ha="left",
        fontsize=8,
        fontweight="bold",
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=time_palette[time_class],
            markeredgecolor="white",
            label=time_class,
            markersize=4,
        )
        for time_class in ["≤14 d", "15–55 d", "≥56 d"]
    ] + [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="#777777",
            markeredgecolor="white",
            label="Included in adjusted model",
            markersize=4,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="#777777",
            label="Single-time publication",
            markersize=4,
        ),
    ]
    fig.legend(
        handles=legend_handles,
        title="Maturation/time and model inclusion",
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=5,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    fig.subplots_adjust(
        left=0.08, right=0.99, top=0.83, bottom=0.27
    )
    for extension in ("pdf", "svg", "png"):
        fig.savefig(
            C.OUT
            / f"fig5_h_f_maturation_reference_distance.{extension}",
            dpi=300,
            facecolor="white",
        )
    plt.close(fig)


def render_maturation_models_main(
    samples: pd.DataFrame,
    models: pd.DataFrame,
) -> None:
    configure_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(180 * MM, 68 * MM),
        gridspec_kw={"wspace": 0.38},
        sharex=True,
        sharey=True,
    )
    time_palette = {
        key: TIME_COLORS[key]
        for key in ["≤14 d", "15–55 d", "≥56 d"]
    }
    tick_days = np.array([0, 3, 7, 14, 28, 56, 98], dtype=float)
    tick_positions = np.log1p(tick_days)
    across = models[
        models["model_type"].eq("across_study_ols_hc3")
    ].iloc[0]
    within = models[
        models["model_type"].eq(
            "within_study_publication_adjusted_ols_hc3"
        )
    ].iloc[0]

    ax = axes[0]
    for time_class in ["≤14 d", "15–55 d", "≥56 d"]:
        frame = samples[samples["time_class"].eq(time_class)]
        ax.scatter(
            frame["log1p_day"],
            frame["median_nn_distance"],
            s=16,
            color=time_palette[time_class],
            edgecolor="white",
            linewidth=0.3,
            label=time_class,
            zorder=3,
        )
    grid = np.linspace(
        samples["log1p_day"].min(),
        samples["log1p_day"].max(),
        200,
    )
    across_fit = smf.ols(
        "median_nn_distance ~ log1p_day",
        data=samples,
    ).fit(cov_type="HC3")
    prediction = across_fit.get_prediction(
        pd.DataFrame({"log1p_day": grid})
    ).summary_frame(alpha=0.05)
    ax.plot(grid, prediction["mean"], color="#222222", lw=0.9)
    ax.fill_between(
        grid,
        prediction["mean_ci_lower"],
        prediction["mean_ci_upper"],
        color="#777777",
        alpha=0.16,
        linewidth=0,
    )
    ax.set_xticks(tick_positions, [str(int(day)) for day in tick_days])
    ax.set_xlabel("Numeric maturation day (log1p scale)")
    ax.set_ylabel(
        "Sample median nearest-HGCA-reference distance\n"
        "(lower = closer to adult HGCA)"
    )
    ax.set_title(
        "Across-study association",
        fontweight="bold",
    )
    ax.text(
        0.98,
        0.98,
        f"n={int(across['n_samples'])} samples, "
        f"{int(across['n_publications'])} publications\n"
        f"β={across['coefficient']:.2f} "
        f"[{across['ci_low']:.2f}, {across['ci_high']:.2f}]\n"
        f"HC3 P={across['p_value']:.2g}\n"
        "Medians: ≤14 d 4.27; 15–55 d 2.22; ≥56 d 2.00",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=4.7,
    )
    sns.despine(ax=ax)
    ax.text(
        -0.16,
        1.05,
        "f",
        transform=ax.transAxes,
        fontsize=7,
        fontweight="bold",
    )

    ax = axes[1]
    model_samples = samples[samples["model_included"]].copy()
    for time_class in ["≤14 d", "15–55 d", "≥56 d"]:
        frame = model_samples[
            model_samples["time_class"].eq(time_class)
        ]
        ax.scatter(
            frame["log1p_day"],
            frame["median_nn_distance"],
            s=16,
            color=time_palette[time_class],
            edgecolor="white",
            linewidth=0.3,
            zorder=3,
        )
    within_grid = np.linspace(
        samples["log1p_day"].min(),
        samples["log1p_day"].max(),
        200,
    )
    adjusted_fit = smf.ols(
        "median_nn_distance ~ log1p_day + publication_display",
        data=model_samples,
    ).fit(cov_type="HC3")
    parameter_names = list(adjusted_fit.params.index)
    publication_terms = [
        name
        for name in parameter_names
        if name.startswith("publication_display[T.")
    ]
    n_publications = model_samples["publication_display"].nunique()
    prediction_rows = []
    prediction_se = []
    covariance = adjusted_fit.cov_params().to_numpy()
    for value in within_grid:
        contrast = np.zeros(len(parameter_names), dtype=float)
        contrast[parameter_names.index("Intercept")] = 1.0
        contrast[parameter_names.index("log1p_day")] = value
        for name in publication_terms:
            contrast[parameter_names.index(name)] = 1 / n_publications
        prediction_rows.append(
            float(contrast @ adjusted_fit.params.to_numpy())
        )
        prediction_se.append(
            float(np.sqrt(contrast @ covariance @ contrast))
        )
    within_fit = np.asarray(prediction_rows)
    within_uncertainty = 1.96 * np.asarray(prediction_se)
    ax.plot(within_grid, within_fit, color="#222222", lw=0.9)
    ax.fill_between(
        within_grid,
        within_fit - within_uncertainty,
        within_fit + within_uncertainty,
        color="#777777",
        alpha=0.16,
        linewidth=0,
    )
    ax.set_xticks(tick_positions, [str(int(day)) for day in tick_days])
    ax.set_xlabel("Numeric maturation day (log1p scale)")
    ax.set_ylabel(
        "Sample median nearest-HGCA-reference distance\n"
        "(lower = closer to adult HGCA)"
    )
    ax.set_title(
        "Within-study association (publication-adjusted)",
        fontweight="bold",
    )
    ax.tick_params(axis="y", labelleft=True)
    ax.text(
        0.98,
        0.98,
        f"n={int(within['n_samples'])} samples, "
        f"{int(within['n_publications'])} time-varying publications\n"
        f"β={within['coefficient']:.2f} "
        f"[{within['ci_low']:.2f}, {within['ci_high']:.2f}]\n"
        f"HC3 P={within['p_value']:.2g}; all 5 study slopes <0",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=4.7,
    )
    sns.despine(ax=ax)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=time_palette[time_class],
            markeredgecolor="white",
            label=time_class,
            markersize=4,
        )
        for time_class in ["≤14 d", "15–55 d", "≥56 d"]
    ]
    fig.legend(
        handles=legend_handles,
        title="Maturation/time",
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    fig.suptitle(
        "PSC-organoid maturation is associated with lower adult-reference distance",
        x=0.08,
        y=0.99,
        ha="left",
        fontsize=8,
        fontweight="bold",
    )
    fig.subplots_adjust(
        left=0.1, right=0.98, top=0.84, bottom=0.27
    )
    for extension in ("pdf", "svg", "png"):
        fig.savefig(
            C.OUT
            / f"fig5_h_f_maturation_reference_distance.{extension}",
            dpi=300,
            facecolor="white",
        )
    plt.close(fig)


def render_source_faceted_maturation_distance(
    frame: pd.DataFrame,
) -> None:
    configure_style()
    figure, axes = plt.subplots(
        1,
        4,
        figsize=(180 * MM, 66 * MM),
        gridspec_kw={
            "width_ratios": [2.15, 0.72, 1.0, 1.35],
            "wspace": 0.43,
        },
        sharey=True,
    )
    overall_order = [
        "Not reported",
        "Reported early",
        "Reported late",
        "PSC ≤14 d",
        "PSC 15–55 d",
        "PSC ≥56 d",
    ]
    palette = {
        "Not reported": TIME_COLORS["Not reported"],
        "Reported early": TIME_COLORS["Early"],
        "Reported late": TIME_COLORS["Late"],
        "PSC ≤14 d": TIME_COLORS["≤14 d"],
        "PSC 15–55 d": TIME_COLORS["15–55 d"],
        "PSC ≥56 d": TIME_COLORS["≥56 d"],
    }
    panels = [
        (
            "Overall",
            frame,
            overall_order,
            "Reported maturation category",
            "Descriptive only:\ncategory definitions differ by source",
        ),
        (
            SOURCE_LABELS["ASC"].replace(" (ASC)", "\n(ASC)"),
            frame[frame["source_standardized"].eq("ASC")],
            ["Not reported"],
            "Maturation",
            "0/29 samples report maturation time",
        ),
        (
            SOURCE_LABELS["FSC"].replace(" (FSC)", "\n(FSC)"),
            frame[frame["source_standardized"].eq("FSC")],
            ["Reported early", "Reported late"],
            "Reported maturation",
            "Early: 6 samples, 1 publication\nLate: 22 samples, 2 publications",
        ),
        (
            SOURCE_LABELS["PSC"].replace(" (PSC)", "\n(PSC)"),
            frame[frame["source_standardized"].eq("PSC")],
            ["PSC ≤14 d", "PSC 15–55 d", "PSC ≥56 d"],
            "Numeric maturation class",
            "41 samples, 9 publications",
        ),
    ]
    for panel_index, (
        title,
        subset,
        order,
        xlabel,
        note,
    ) in enumerate(panels):
        axis = axes[panel_index]
        present_order = [
            value
            for value in order
            if subset["maturation_display"].eq(value).any()
        ]
        sns.boxplot(
            data=subset,
            x="maturation_display",
            y="median_nn_distance",
            hue="maturation_display",
            order=present_order,
            hue_order=present_order,
            palette=palette,
            legend=False,
            width=0.58,
            linewidth=0.55,
            fliersize=0,
            saturation=0.72,
            ax=axis,
        )
        sns.stripplot(
            data=subset,
            x="maturation_display",
            y="median_nn_distance",
            order=present_order,
            color="#222222",
            alpha=0.62,
            size=2.2,
            jitter=0.18,
            ax=axis,
        )
        tick_labels = []
        for value in present_order:
            count = subset["maturation_display"].eq(value).sum()
            short = (
                value.replace("Reported ", "")
                .replace("PSC ", "")
                .replace("Not reported", "Missing")
            )
            tick_labels.append(f"{short}\nn={count}")
        axis.set_xticks(
            range(len(present_order)),
            tick_labels,
            rotation=38 if panel_index == 0 else 25,
            ha="right",
        )
        axis.set_xlabel(xlabel)
        axis.set_title(title, fontweight="bold")
        axis.text(
            0.5,
            0.98,
            note,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=4.2,
        )
        axis.grid(False)
        sns.despine(ax=axis)
        if panel_index:
            axis.set_ylabel("")
        else:
            axis.set_ylabel(
                "Sample median nearest-HGCA-reference distance\n"
                "(lower = closer to adult HGCA)"
            )
            axis.text(
                -0.17,
                1.05,
                "g",
                transform=axis.transAxes,
                fontsize=7,
                fontweight="bold",
            )
    axes[0].set_ylim(
        max(0, frame["median_nn_distance"].min() - 0.3),
        frame["median_nn_distance"].max() + 1.35,
    )
    figure.suptitle(
        "Maturation metadata and adult-reference distance by organoid source",
        x=0.08,
        y=0.99,
        ha="left",
        fontsize=8,
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.09, right=0.99, top=0.81, bottom=0.28
    )
    for extension in ("pdf", "svg", "png"):
        figure.savefig(
            C.OUT
            / f"fig5_supp10g_source_maturation_reference_distance.{extension}",
            dpi=300,
            facecolor="white",
        )
    plt.close(figure)


def render_main_figure_composite(
    maturation_samples: pd.DataFrame,
    maturation_models: pd.DataFrame,
) -> None:
    configure_style()
    figure = plt.figure(figsize=(180 * MM, 240 * MM))
    outer = figure.add_gridspec(
        4,
        1,
        height_ratios=[0.78, 0.88, 0.88, 0.82],
        hspace=0.4,
    )
    top = outer[0].subgridspec(
        1, 2, width_ratios=[1.62, 1], wspace=0.34
    )

    def label_panel(axis: plt.Axes, label: str) -> None:
        axis.text(
            -0.12,
            1.06,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )

    # A: reserved for the externally assembled ARBOL taxonomy.
    axis_a = figure.add_subplot(top[0, 0])
    axis_a.axis("off")
    label_panel(axis_a, "A")

    # B: sample-level mapping QC.
    axis_b = figure.add_subplot(top[0, 1])
    sample_qc = pd.read_csv(C.DATA / "supp_mapping_qc_by_sample.csv")
    sample_qc["region_broad"] = sample_qc["region_broad"].fillna(
        "Not reported"
    )
    region_palette = {
        **SEGMENT_COLORS,
        "Intestine": "#8C8C8C",
        "Not reported": "#D0D0D0",
    }
    source_order = ["ASC", "FSC", "PSC"]
    region_order = [
        value
        for value in [
            "Duodenum",
            "Jejunum",
            "Ileum",
            "Colon",
            "Intestine",
            "Not reported",
        ]
        if value in set(sample_qc["region_broad"])
    ]
    sns.boxplot(
        data=sample_qc,
        x="source_standardized",
        y="fraction_strict_mapping_pass",
        hue="region_broad",
        order=source_order,
        hue_order=region_order,
        palette=region_palette,
        linewidth=0.5,
        fliersize=0,
        ax=axis_b,
    )
    sns.stripplot(
        data=sample_qc,
        x="source_standardized",
        y="fraction_strict_mapping_pass",
        hue="region_broad",
        order=source_order,
        hue_order=region_order,
        palette=region_palette,
        dodge=True,
        jitter=0.1,
        size=2,
        edgecolor="white",
        linewidth=0.15,
        ax=axis_b,
    )
    handles, labels = axis_b.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis_b.legend(
        unique.values(),
        unique.keys(),
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=4.2,
        handletextpad=0.25,
        borderaxespad=0,
    )
    axis_b.set_xticks(
        range(3),
        [
            "Adult stem cell-\nderived (ASC)",
            "Fetal stem cell-\nderived (FSC)",
            "Pluripotent stem cell-\nderived (PSC)",
        ],
    )
    axis_b.tick_params(axis="x", labelsize=4.4)
    axis_b.set_xlabel("")
    axis_b.set_ylabel(
        "Cells passing confidence + distance QC"
    )
    axis_b.set_ylim(0, 1.02)
    axis_b.grid(False)
    sns.despine(ax=axis_b)
    label_panel(axis_b, "B")

    # C: compact origin-proximity matrix.
    axis_c = figure.add_subplot(outer[1])
    origin = pd.read_csv(C.DATA / "fig5b_origin_proximity_summary.csv")
    hierarchy = pd.read_csv(
        C.DATA / "supp_subtype_capability_table.csv", index_col=0
    )
    origin = origin[
        (origin["n_samples"] >= 3) & (origin["n_publications"] >= 2)
    ].copy()
    small_intestine = origin["origin_region"].isin(["Duodenum", "Ileum"])
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
        ("PSC", "Colon"),
    ]
    column_lookup = {
        value: index for index, value in enumerate(columns)
    }
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
        for subtype in hierarchy.index
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
                np.abs(origin["median_relative_origin_proximity"]),
                0.98,
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
        sizes=(6, 62),
        edgecolor="#333333",
        linewidth=0.25,
        legend=False,
        ax=axis_c,
    )
    axis_c.set_xticks(
        range(len(columns)),
        [
            region.replace("Duodenum", "Duod.")
            for _, region in columns
        ],
    )
    for tick, (_, region) in zip(axis_c.get_xticklabels(), columns):
        tick.set_color(region_palette[region])
    axis_c.set_yticks(
        range(len(subtype_order)),
        [short_identity_label(value) for value in subtype_order],
    )
    axis_c.set_ylim(len(subtype_order) - 0.5, -1.0)
    axis_c.set_xlim(-0.55, len(columns) - 0.45)
    axis_c.set_xlabel("Declared segment of origin")
    axis_c.set_ylabel("HGCA epithelial subtype")
    axis_c.tick_params(length=0, labelsize=4.2)
    for boundary in [2.5, 4.5]:
        axis_c.axvline(boundary, color="#BDBDBD", lw=0.45)
    source_headers = {
        "ASC": "Adult stem cell-\nderived (ASC)",
        "FSC": "Fetal stem cell-\nderived (FSC)",
        "PSC": "Pluripotent stem cell-\nderived (PSC)",
    }
    for center, source in [(1, "ASC"), (3.5, "FSC"), (5, "PSC")]:
        axis_c.text(
            center,
            -0.9,
            source_headers[source],
            ha="center",
            va="bottom",
            fontsize=3.7,
            fontweight="bold",
        )
    for spine in axis_c.spines.values():
        spine.set_visible(False)
    color_axis = axis_c.inset_axes([1.03, 0.44, 0.025, 0.38])
    colorbar = figure.colorbar(
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
        "Median relative origin proximity\n(positive = origin closer)",
        fontsize=4.2,
        labelpad=3,
    )
    colorbar.ax.tick_params(labelsize=3.8, length=2)
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="#333333",
            markeredgewidth=0.3,
            markersize=size,
            label=label,
        )
        for size, label in [(2.5, "25%"), (4.5, "50%"), (7, "100%")]
    ]
    axis_c.legend(
        handles=size_handles,
        title="Cells with declared\norigin closer",
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(1.01, 0.02),
        fontsize=4,
        title_fontsize=4.2,
        borderaxespad=0,
    )
    label_panel(axis_c, "C")

    # D: three shared CLR-PCA maps and all-PC partial R2.
    d_grid = outer[2].subgridspec(
        1, 4, wspace=0.64
    )
    coordinates = pd.read_csv(
        C.DATA / "fig5c_clr_pca_coordinates.csv", index_col=0
    )
    metadata = pd.read_csv(C.DATA / "sample_metadata.csv", index_col=0)
    pca_variance = pd.read_csv(C.DATA / "fig5c_clr_pca_variance.csv")
    partial_r2 = pd.read_csv(C.DATA / "fig5c_pca_partial_r2.csv")
    pca_frame = coordinates.join(metadata)
    pc1_percent = 100 * pca_variance.loc[
        pca_variance["axis"].eq("PC1"), "explained_variance_fraction"
    ].iloc[0]
    pc2_percent = 100 * pca_variance.loc[
        pca_variance["axis"].eq("PC2"), "explained_variance_fraction"
    ].iloc[0]
    source_palette = {
        "ASC": "#0072B2",
        "FSC": "#009E73",
        "PSC": "#D55E00",
    }
    embedding_specs = [
        (
            "time_class",
            {
                key: TIME_COLORS[key]
                for key in [
                    "Not reported",
                    "Early",
                    "Late",
                    "≤14 d",
                    "15–55 d",
                    "≥56 d",
                ]
            },
            "Maturation/time",
        ),
        (
            "region_broad",
            region_palette,
            "Declared region",
        ),
        (
            "source_standardized",
            source_palette,
            "Organoid source",
        ),
    ]
    d_axes = []
    for index, (field, palette, legend_title) in enumerate(
        embedding_specs
    ):
        cell_grid = d_grid[0, index].subgridspec(
            2,
            1,
            height_ratios=[1, 0.34],
            hspace=0.02,
        )
        axis = figure.add_subplot(cell_grid[0, 0])
        legend_axis = figure.add_subplot(cell_grid[1, 0])
        legend_axis.axis("off")
        d_axes.append(axis)
        plot_frame = pca_frame.copy()
        plot_frame[field] = plot_frame[field].fillna("Not reported")
        sns.scatterplot(
            data=plot_frame,
            x="PC1",
            y="PC2",
            hue=field,
            palette=palette,
            s=10,
            edgecolor="white",
            linewidth=0.15,
            legend=True,
            ax=axis,
        )
        axis.set_box_aspect(1)
        axis.set_xlabel(f"CLR PC1 ({pc1_percent:.1f}%)")
        axis.set_ylabel(f"CLR PC2 ({pc2_percent:.1f}%)")
        axis.tick_params(labelsize=4.2)
        handles, legend_labels = axis.get_legend_handles_labels()
        if axis.get_legend() is not None:
            axis.get_legend().remove()
        if field == "source_standardized":
            legend_labels = [
                source_headers.get(value, value)
                for value in legend_labels
            ]
        legend_axis.legend(
            handles,
            legend_labels,
            frameon=False,
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            ncol=2 if field != "source_standardized" else 1,
            fontsize=3.2,
            title=legend_title,
            title_fontsize=3.5,
            handletextpad=0.2,
            columnspacing=0.35,
            borderaxespad=0,
        )
        axis.grid(False)
        sns.despine(ax=axis)
    label_panel(d_axes[0], "D")

    partial_axis = figure.add_subplot(d_grid[0, 3])
    partial_plot = partial_r2.sort_values(
        "partial_r2_all_nonzero_pcs"
    )
    sns.barplot(
        data=partial_plot,
        y="display",
        x="partial_r2_all_nonzero_pcs",
        color="#0072B2",
        edgecolor="none",
        ax=partial_axis,
    )
    for patch, sample_count in zip(
        partial_axis.patches, partial_plot["n_samples"]
    ):
        partial_axis.text(
            patch.get_width() + 0.008,
            patch.get_y() + patch.get_height() / 2,
            f"n={int(sample_count)}",
            va="center",
            fontsize=3.7,
        )
    partial_axis.set_xlabel(
        "Covariate-adjusted partial R²\n"
        f"all {int(partial_r2['n_nonzero_pcs'].iloc[0])} CLR PCs"
    )
    partial_axis.set_ylabel("")
    partial_axis.tick_params(axis="y", labelsize=3.8)
    partial_axis.grid(False)
    sns.despine(ax=partial_axis)

    # E: across-study and publication-adjusted maturation associations.
    e_grid = outer[3].subgridspec(1, 2, wspace=0.34)
    tick_days = np.array([0, 3, 7, 14, 28, 56, 98], dtype=float)
    tick_positions = np.log1p(tick_days)
    time_palette = {
        key: TIME_COLORS[key]
        for key in ["≤14 d", "15–55 d", "≥56 d"]
    }
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
    across_samples = maturation_samples
    within_samples = maturation_samples[
        maturation_samples["model_included"]
    ].copy()
    for axis, plot_samples in zip(
        e_axes, [across_samples, within_samples]
    ):
        for time_class in ["≤14 d", "15–55 d", "≥56 d"]:
            subset = plot_samples[
                plot_samples["time_class"].eq(time_class)
            ]
            axis.scatter(
                subset["log1p_day"],
                subset["median_nn_distance"],
                s=14,
                color=time_palette[time_class],
                edgecolor="white",
                linewidth=0.25,
                zorder=3,
            )
        axis.set_xticks(
            tick_positions, [str(int(day)) for day in tick_days]
        )
        axis.set_xlabel("Maturation day (log1p scale)")
        axis.set_ylabel(
            "Sample median HGCA distance\n(lower = closer)"
        )
        axis.set_xlim(
            maturation_samples["log1p_day"].min() - 0.15,
            maturation_samples["log1p_day"].max() + 0.15,
        )
        axis.set_ylim(
            maturation_samples["median_nn_distance"].min() - 0.35,
            maturation_samples["median_nn_distance"].max() + 0.55,
        )
        axis.grid(False)
        sns.despine(ax=axis)
    across_grid = np.linspace(
        across_samples["log1p_day"].min(),
        across_samples["log1p_day"].max(),
        200,
    )
    across_fit = smf.ols(
        "median_nn_distance ~ log1p_day",
        data=across_samples,
    ).fit(cov_type="HC3")
    across_prediction = across_fit.get_prediction(
        pd.DataFrame({"log1p_day": across_grid})
    ).summary_frame(alpha=0.05)
    e_axes[0].plot(
        across_grid, across_prediction["mean"], color="#222222", lw=0.8
    )
    e_axes[0].fill_between(
        across_grid,
        across_prediction["mean_ci_lower"],
        across_prediction["mean_ci_upper"],
        color="#777777",
        alpha=0.16,
        linewidth=0,
    )
    e_axes[0].text(
        0.02,
        0.98,
        "Across studies",
        transform=e_axes[0].transAxes,
        va="top",
        fontweight="bold",
        fontsize=5.3,
    )
    e_axes[0].text(
        0.98,
        0.98,
        f"n={int(across['n_samples'])}; β={across['coefficient']:.2f} "
        f"[{across['ci_low']:.2f}, {across['ci_high']:.2f}]",
        transform=e_axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=4.2,
    )
    within_grid = np.linspace(
        maturation_samples["log1p_day"].min(),
        maturation_samples["log1p_day"].max(),
        200,
    )
    adjusted_fit = smf.ols(
        "median_nn_distance ~ log1p_day + publication_display",
        data=within_samples,
    ).fit(cov_type="HC3")
    parameter_names = list(adjusted_fit.params.index)
    publication_terms = [
        name
        for name in parameter_names
        if name.startswith("publication_display[T.")
    ]
    n_publications = within_samples["publication_display"].nunique()
    fitted_values = []
    fitted_se = []
    covariance = adjusted_fit.cov_params().to_numpy()
    for value in within_grid:
        contrast = np.zeros(len(parameter_names), dtype=float)
        contrast[parameter_names.index("Intercept")] = 1
        contrast[parameter_names.index("log1p_day")] = value
        for name in publication_terms:
            contrast[parameter_names.index(name)] = 1 / n_publications
        fitted_values.append(
            float(contrast @ adjusted_fit.params.to_numpy())
        )
        fitted_se.append(
            float(np.sqrt(contrast @ covariance @ contrast))
        )
    fitted_values = np.asarray(fitted_values)
    fitted_se = np.asarray(fitted_se)
    e_axes[1].plot(
        within_grid, fitted_values, color="#222222", lw=0.8
    )
    e_axes[1].fill_between(
        within_grid,
        fitted_values - 1.96 * fitted_se,
        fitted_values + 1.96 * fitted_se,
        color="#777777",
        alpha=0.16,
        linewidth=0,
    )
    e_axes[1].text(
        0.02,
        0.98,
        "Within studies (publication-adjusted)",
        transform=e_axes[1].transAxes,
        va="top",
        fontweight="bold",
        fontsize=5.3,
    )
    e_axes[1].text(
        0.98,
        0.98,
        f"n={int(within['n_samples'])}; β={within['coefficient']:.2f} "
        f"[{within['ci_low']:.2f}, {within['ci_high']:.2f}]",
        transform=e_axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=4.2,
    )
    label_panel(e_axes[0], "E")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=time_palette[key],
            markeredgecolor="white",
            markersize=4,
            label=key,
        )
        for key in ["≤14 d", "15–55 d", "≥56 d"]
    ]
    figure.legend(
        handles=legend_handles,
        title="Maturation/time",
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        fontsize=4.2,
        title_fontsize=4.5,
        handletextpad=0.25,
        columnspacing=0.7,
    )
    figure.subplots_adjust(
        left=0.12, right=0.91, top=0.98, bottom=0.07
    )
    for extension in ("pdf", "svg", "png"):
        figure.savefig(
            C.OUT / f"fig5_main_composite.{extension}",
            dpi=300,
            facecolor="white",
        )
    plt.close(figure)


def render_maturation_origin_supplement(
    coverage: pd.DataFrame,
) -> None:
    configure_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(180 * MM, 66 * MM),
        gridspec_kw={"wspace": 0.42, "width_ratios": [1.08, 0.92]},
    )
    tick_days = np.array([0, 3, 7, 14, 28, 56, 98], dtype=float)
    tick_positions = np.log1p(tick_days)
    region_order = ["Duodenum", "Jejunum", "Ileum", "Colon"]
    region_colors = {
        "Duodenum": SEGMENT_COLORS["Duodenum"],
        "Jejunum": SEGMENT_COLORS["Jejunum"],
        "Ileum": SEGMENT_COLORS["Ileum"],
        "Colon": SEGMENT_COLORS["Colon"],
    }

    ax = axes[0]
    day_offsets = {}
    for day, frame in coverage.groupby("day", observed=True):
        ordered_ids = sorted(frame["sample_id"].astype(str))
        offsets = np.linspace(-0.035, 0.035, len(ordered_ids))
        day_offsets.update(
            dict(zip(ordered_ids, offsets))
        )
    for region_index, region in enumerate(region_order):
        fractions = coverage[region].to_numpy(float)
        x_values = coverage["log1p_day"].to_numpy(float) + coverage[
            "sample_id"
        ].map(day_offsets).to_numpy(float)
        ax.scatter(
            x_values,
            np.full(
                len(coverage), len(region_order) - 1 - region_index
            ),
            s=4 + 52 * fractions,
            color=region_colors[region],
            edgecolor="white",
            linewidth=0.2,
            alpha=0.48,
        )
    ax.set_yticks(range(4), region_order[::-1])
    ax.set_xticks(tick_positions, [str(int(day)) for day in tick_days])
    ax.set_xlabel("Numeric maturation day (log1p scale)")
    ax.set_ylabel("Nearest HGCA segment")
    ax.set_title(
        "Nearest-segment profile\n(origin-independent)",
        fontweight="bold",
    )
    ax.text(
        0.03,
        0.97,
        'n=41 samples, 9 publications\n37 declared only as "Intestine"\n'
        "Bubble area = within-sample fraction (25–100%)",
        transform=ax.transAxes,
        va="top",
        fontsize=4.7,
    )
    sns.despine(ax=ax)
    ax.text(
        -0.16,
        1.05,
        "f",
        transform=ax.transAxes,
        fontsize=7,
        fontweight="bold",
    )

    ax = axes[1]
    available = coverage[
        coverage["origin_proximity_available"]
    ].copy()
    ax.scatter(
        available["log1p_day"],
        available["median_relative_origin_proximity"],
        s=19,
        color=TIME_COLORS["15–55 d"],
        edgecolor="white",
        linewidth=0.35,
    )
    ax.axhline(0, color="#777777", lw=0.55)
    labels = []
    for _, row in available.iterrows():
        labels.append(
            ax.text(
                row["log1p_day"],
                row["median_relative_origin_proximity"],
                f"{short_publication_label(row['publication_display'])}\n"
                f"{row['source_standardized']} · {row['region_broad']}",
                fontsize=4.1,
                color="#444444",
                ha="right",
            )
        )
    adjust_text(
        labels,
        ax=ax,
        force_text=(0.35, 0.6),
        expand=(1.05, 1.12),
        max_move=12,
        ensure_inside_axes=True,
    )
    ax.set_xticks(tick_positions, [str(int(day)) for day in tick_days])
    ax.set_xlabel("Numeric maturation day (log1p scale)")
    ax.set_ylabel(
        "Median relative origin proximity\n"
        "(positive = declared origin is closer)"
    )
    ax.set_title(
        "Segment-of-origin proximity",
        fontweight="bold",
    )
    ax.text(
        0.03,
        0.97,
        "Not estimable: n=4 samples, 2 publications.\n"
        "37/41 lack a specific declared segment.",
        transform=ax.transAxes,
        va="top",
        fontsize=4.7,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.9,
            "pad": 1.0,
        },
    )
    sns.despine(ax=ax)
    fig.suptitle(
        "Regional-reference coverage across numeric PSC maturation times",
        x=0.09,
        y=0.99,
        ha="left",
        fontsize=8,
        fontweight="bold",
    )
    fig.subplots_adjust(
        left=0.1, right=0.98, top=0.82, bottom=0.22
    )
    for extension in ("pdf", "svg", "png"):
        fig.savefig(
            C.OUT / f"fig5_supp10f_origin_coverage.{extension}",
            dpi=300,
            facecolor="white",
        )
    plt.close(fig)


def draw_ternary_axis(
    ax: plt.Axes, frame: pd.DataFrame, source: str
) -> None:
    height = np.sqrt(3) / 2
    triangle_x = [0, 1, 0.5, 0]
    triangle_y = [0, 0, height, 0]
    ax.plot(triangle_x, triangle_y, color="#555555", lw=0.6)
    for fraction in [0.25, 0.5, 0.75]:
        ax.plot(
            [fraction / 2, 1 - fraction / 2],
            [fraction * height, fraction * height],
            color="#E0E0E0",
            lw=0.3,
            zorder=0,
        )
    subset = frame[
        frame["source_standardized"].eq(source)
    ].copy()
    subset["_x"] = (
        subset["absorptive_fraction"]
        + 0.5 * subset["secretory_fraction"]
    )
    subset["_y"] = height * subset["secretory_fraction"]
    colors = subset["time_class"].fillna("Not reported").map(TIME_COLORS)
    ax.scatter(
        subset["_x"],
        subset["_y"],
        c=colors,
        s=12,
        edgecolors="white",
        linewidths=0.25,
        zorder=2,
    )
    ax.text(-0.02, -0.04, "Progenitor", ha="left", va="top", fontsize=5)
    ax.text(1.02, -0.04, "Absorptive", ha="right", va="top", fontsize=5)
    ax.text(0.5, height + 0.025, "Secretory", ha="center", fontsize=5)
    ax.set_title(SOURCE_LABELS[source], fontweight="bold")
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.08, height + 0.08)
    ax.set_aspect("equal")
    ax.axis("off")


def render_branch_differentiation_figure(
    branches: pd.DataFrame,
    condition_effects: pd.DataFrame,
    subtype_effects: pd.DataFrame,
    mapping_sample: pd.DataFrame,
    mapping_contrasts: pd.DataFrame,
) -> None:
    configure_style()
    fig = plt.figure(figsize=(180 * MM, 178 * MM))
    grid = fig.add_gridspec(
        3,
        6,
        height_ratios=[1.0, 1.25, 0.9],
        hspace=0.72,
        wspace=0.7,
    )
    ternary_grid = grid[0, :].subgridspec(1, 3, wspace=0.25)
    for index, source in enumerate(["ASC", "FSC", "PSC"]):
        ax = fig.add_subplot(ternary_grid[0, index])
        draw_ternary_axis(ax, branches, source)
        if index == 0:
            ax.text(
                -0.12,
                1.08,
                "a",
                transform=ax.transAxes,
                fontsize=7,
                fontweight="bold",
            )
    time_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=color,
            label=label,
            markersize=3.5,
        )
        for label, color in TIME_COLORS.items()
    ]
    fig.legend(
        handles=time_handles,
        frameon=False,
        ncol=6,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.69),
        title="Maturation/time",
    )

    ax = fig.add_subplot(grid[1, :2])
    forest = condition_effects.iloc[::-1].reset_index(drop=True)
    family_colors = {
        "Molecular": "#0072B2",
        "Co-culture": "#009E73",
        "Maturation": "#CC79A7",
    }
    forest_labels = {
        "capeling_enr_vs_egf": "Capeling ENR−EGF (d7/14)",
        "yu_nrg1_vs_egf_d40": "Yu NRG1−EGF (d40)",
        "yu_enr_vs_egf_d28": "Yu ENR−EGF (d28)",
        "kilik_nrg1_vs_egf": "Kilik NRG1−EGF",
        "he_il22_pos_vs_neg": "He IL-22+−IL-22−",
        "heoca_hioec_vs_control": "HEOCA HIO-EC−control",
        "capeling_day28_vs_early": "Capeling EGF d28−d7/14",
        "yu_week8_vs_week4": "Yu ENR wk8−wk4",
    }
    for y_index, row in forest.iterrows():
        color = family_colors[row["family"]]
        if np.isfinite(row["secretory_fraction_difference_ci_low"]):
            ax.plot(
                [
                    100 * row["secretory_fraction_difference_ci_low"],
                    100 * row["secretory_fraction_difference_ci_high"],
                ],
                [y_index, y_index],
                color=color,
                lw=0.8,
            )
        ax.scatter(
            100 * row["secretory_fraction_difference"],
            y_index,
            s=22,
            marker=(
                "o" if row["bootstrap_interval_available"] else "s"
            ),
            facecolor=(
                color
                if row["bootstrap_interval_available"]
                else "white"
            ),
            edgecolor=color,
            linewidth=0.7,
            zorder=3,
        )
    ax.axvline(0, color="#777777", lw=0.5)
    ax.set_yticks(
        range(len(forest)),
        [
            forest_labels.get(value, value)
            for value in forest["contrast_id"]
        ],
    )
    for tick, family in zip(ax.get_yticklabels(), forest["family"]):
        tick.set_color(family_colors[family])
    ax.set_xlabel(
        "Change in differentiated secretory cells\n"
        "(percentage points; test − comparator)"
    )
    ax.set_title(
        "Within-study condition contrasts\n"
        "(descriptive; no cross-study replication)",
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.02,
        "● resampling interval\n□ single-sample arm",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=4.2,
    )
    sns.despine(ax=ax)
    ax.text(
        -0.18,
        1.05,
        "b",
        transform=ax.transAxes,
        fontsize=7,
        fontweight="bold",
    )

    ax = fig.add_subplot(grid[1, 2:])
    contrast_order = condition_effects["contrast_id"].tolist()
    subtype_order = [
        subtype
        for subtype in SECRETORY_IDENTITY_ORDER
        if subtype in subtype_effects["hgca_celltype_v1"].unique()
    ]
    matrix = subtype_effects.copy()
    matrix["x"] = matrix["contrast_id"].map(
        {value: index for index, value in enumerate(contrast_order)}
    )
    matrix["y"] = matrix["hgca_celltype_v1"].map(
        {
            value: len(subtype_order) - 1 - index
            for index, value in enumerate(subtype_order)
        }
    )
    vmax = max(
        1.0,
        float(
            np.nanquantile(
                np.abs(matrix["log2_proportion_effect"]), 0.95
            )
        ),
    )
    sizes = 5 + 38 * np.sqrt(
        matrix["n_mapped_cells"]
        / max(1, matrix["n_mapped_cells"].max())
    )
    edge_values = matrix["median_mapping_confidence"].fillna(0.5)
    edge_colors = plt.cm.Greys(
        np.clip((edge_values.to_numpy() - 0.45) / 0.45, 0.2, 1)
    )
    scatter = ax.scatter(
        matrix["x"],
        matrix["y"],
        c=matrix["log2_proportion_effect"],
        cmap="vlag",
        vmin=-vmax,
        vmax=vmax,
        s=sizes,
        edgecolors=edge_colors,
        linewidths=0.8,
    )
    ax.set_xticks(
        range(len(contrast_order)),
        [
            condition_effects.set_index("contrast_id").loc[value, "display"]
            for value in contrast_order
        ],
        rotation=55,
        ha="right",
    )
    ax.set_yticks(
        range(len(subtype_order)),
        [short_identity_label(value) for value in subtype_order[::-1]],
    )
    ax.set_xlim(-0.6, len(contrast_order) - 0.4)
    ax.set_ylim(-0.6, len(subtype_order) - 0.4)
    ax.grid(color="#E8E8E8", lw=0.3)
    ax.set_axisbelow(True)
    ax.set_title(
        "Which secretory and BEST4 identities change?",
        fontweight="bold",
    )
    colorbar = fig.colorbar(
        scatter, ax=ax, fraction=0.035, pad=0.02
    )
    colorbar.set_label("Δ log₂ subtype proportion", fontsize=5)
    colorbar.ax.tick_params(labelsize=4)
    ax.text(
        0.0,
        -0.39,
        "Dot size: mapped cells; darker outline: higher median mapping confidence",
        transform=ax.transAxes,
        fontsize=4.5,
    )
    sns.despine(ax=ax, left=False, bottom=False)
    ax.text(
        -0.12,
        1.05,
        "c",
        transform=ax.transAxes,
        fontsize=7,
        fontweight="bold",
    )

    eligible = mapping_sample[
        mapping_sample["mapping_summary_eligible"]
    ].copy()
    metrics = [
        (
            "median_mapping_confidence",
            "Median mapping\nconfidence",
        ),
        (
            "median_reference_distance_z",
            "Reference-distance score\n(within mapped identity)",
        ),
        (
            "fraction_strict_mapping",
            "Fraction passing\nstrict mapping QC",
        ),
    ]
    order = ["Absorptive", "Secretory", "Progenitor"]
    bottom = grid[2, :].subgridspec(1, 3, wspace=0.42)
    for index, (metric, label) in enumerate(metrics):
        ax = fig.add_subplot(bottom[0, index])
        sns.violinplot(
            data=eligible,
            x="branch",
            y=metric,
            order=order,
            hue="branch",
            palette=BRANCH_COLORS,
            inner="quartile",
            cut=0,
            linewidth=0.45,
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=eligible,
            x="branch",
            y=metric,
            order=order,
            color="#333333",
            size=1.2,
            alpha=0.35,
            jitter=0.16,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.set_xticks(range(3), ["Abs.", "Sec.", "Prog."])
        for category_index, category in enumerate(
            ["Secretory", "Progenitor"]
        ):
            row = mapping_contrasts[
                (mapping_contrasts["metric"] == metric)
                & mapping_contrasts["contrast"].str.startswith(category)
            ]
            if row.empty:
                continue
            result = row.iloc[0]
            ax.text(
                0.02,
                0.98 - 0.11 * category_index,
                f"{category[:3]}−Abs: "
                f"{result['median_paired_difference']:+.2f} "
                f"[{result['publication_bootstrap_ci_low']:+.2f}, "
                f"{result['publication_bootstrap_ci_high']:+.2f}]",
                transform=ax.transAxes,
                va="top",
                fontsize=4.2,
            )
        sns.despine(ax=ax)
        if index == 0:
            ax.text(
                -0.2,
                1.05,
                "d",
                transform=ax.transAxes,
                fontsize=7,
                fontweight="bold",
            )
    fig.suptitle(
        "Organoid conditions allocate epithelial cells among progenitor, absorptive and secretory programs",
        x=0.07,
        y=0.992,
        ha="left",
        fontsize=8,
        fontweight="bold",
    )
    fig.subplots_adjust(
        left=0.15, right=0.98, top=0.94, bottom=0.08
    )
    for extension in ("pdf", "svg", "png"):
        fig.savefig(
            C.OUT / f"fig5_h_epithelial_branch_differentiation.{extension}",
            dpi=300,
            facecolor="white",
        )
    plt.close(fig)


def render_figure(
    point_data: pd.DataFrame,
    sample_data: pd.DataFrame,
    associations: pd.DataFrame,
    class_test: pd.DataFrame,
) -> None:
    configure_style()
    class_supported = bool(
        not class_test.empty
        and (
            class_test.iloc[0]["publication_bootstrap_ci_low"] > 0
            or class_test.iloc[0]["publication_bootstrap_ci_high"] < 0
        )
    )
    n_columns = 4 if class_supported else 3
    fig = plt.figure(figsize=(180 * MM, 82 * MM))
    grid = fig.add_gridspec(
        1,
        n_columns,
        width_ratios=[1, 1, 1, 0.8] if class_supported else [1, 1, 1],
        wspace=0.38,
    )
    x = "balanced_accuracy_above_permutation"
    y = "median_relative_origin_proximity"
    x_limits = (
        point_data[x].min() - 0.03,
        point_data[x].max() + 0.03,
    )
    y_limits = (
        point_data[y].min() - 0.04,
        point_data[y].max() + 0.04,
    )
    region_markers = {
        "Duodenum": "o",
        "Jejunum": "s",
        "Ileum": "^",
        "Colon": "D",
    }
    for panel, source in enumerate(["ASC", "FSC", "PSC"]):
        ax = fig.add_subplot(grid[0, panel])
        subset = point_data[
            point_data["source_standardized"] == source
        ].copy()
        sns.scatterplot(
            data=subset,
            x=x,
            y=y,
            hue="maturation_class",
            style="origin_region",
            size="n_samples",
            palette=MATURATION_COLORS,
            markers=region_markers,
            sizes=(14, 55),
            edgecolor="white",
            linewidth=0.35,
            alpha=0.9,
            legend=False,
            ax=ax,
        )
        ax.axhline(0, color="#BDBDBD", lw=0.4)
        ax.axvline(0, color="#BDBDBD", lw=0.4)
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_title(SOURCE_LABELS[source], fontweight="bold")
        ax.set_xlabel(
            "Healthy donor-held-out segment accuracy\n"
            "minus permuted accuracy"
        )
        ax.set_ylabel(
            "Median organoid relative-origin proximity"
            if panel == 0
            else ""
        )
        supported = subset[subset["well_supported"]].copy()
        if not supported.empty:
            supported["outlier_score"] = (
                np.abs(
                    supported[y]
                    - np.polyval(
                        np.polyfit(supported[x], supported[y], 1),
                        supported[x],
                    )
                )
                if len(supported) >= 3
                else np.abs(supported[y])
            )
            representatives = (
                supported.sort_values("n_samples")
                .drop_duplicates("hgca_celltype_v1", keep="last")
            )
            selected_identities = set(
                representatives.nlargest(
                    min(4, len(representatives)), "n_samples"
                )["hgca_celltype_v1"]
            )
            selected_identities.update(
                representatives.nlargest(
                    min(2, len(representatives)), "outlier_score"
                )["hgca_celltype_v1"]
            )
            labels = representatives[
                representatives["hgca_celltype_v1"].isin(
                    selected_identities
                )
            ]
            texts = [
                ax.text(
                    row[x],
                    row[y],
                    short_identity_label(row["hgca_celltype_v1"]),
                    fontsize=4.0,
                    color="#333333",
                )
                for _, row in labels.iterrows()
            ]
            adjust_text(
                texts,
                ax=ax,
                x=subset[x].to_numpy(),
                y=subset[y].to_numpy(),
                force_text=(0.4, 0.7),
                force_static=(0.2, 0.4),
                expand=(1.08, 1.15),
                max_move=12,
                ensure_inside_axes=True,
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#777777",
                    "lw": 0.3,
                },
            )
        result = associations[associations["scope"] == source]
        if not result.empty:
            row = result.iloc[0]
            ax.text(
                0.02,
                0.02,
                f"β={row['coefficient']:.2f}\n"
                f"95% pub-bootstrap CI "
                f"{row['publication_bootstrap_ci_low']:.2f}, "
                f"{row['publication_bootstrap_ci_high']:.2f}",
                transform=ax.transAxes,
                fontsize=4.5,
                va="bottom",
            )
        sns.despine(ax=ax)
        if panel == 0:
            ax.text(
                -0.2,
                1.04,
                "g",
                transform=ax.transAxes,
                fontsize=7,
                fontweight="bold",
            )

    if class_supported:
        ax = fig.add_subplot(grid[0, 3])
        comparison = sample_data[
            sample_data["maturation_class"].isin(
                ["Shared proliferative/progenitor", "Mature identity"]
            )
            & sample_data["identity_well_supported"]
        ].copy()
        order = ["Shared proliferative/progenitor", "Mature identity"]
        sns.boxplot(
            data=comparison,
            x="maturation_class",
            y=y,
            order=order,
            palette=MATURATION_COLORS,
            hue="maturation_class",
            linewidth=0.5,
            fliersize=0,
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=comparison,
            x="maturation_class",
            y=y,
            order=order,
            color="#333333",
            size=1.8,
            jitter=0.18,
            alpha=0.55,
            ax=ax,
        )
        ax.set_xticks([0, 1], ["Shared\nprogenitor", "Mature"])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim(y_limits)
        ax.set_title("Identity-class comparison", fontweight="bold")
        row = class_test.iloc[0]
        ax.text(
            0.03,
            0.98,
            f"Adjusted Δ={row['coefficient']:.2f}\n"
            f"95% pub-bootstrap CI "
            f"{row['publication_bootstrap_ci_low']:.2f}, "
            f"{row['publication_bootstrap_ci_high']:.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=4.5,
        )
        sns.despine(ax=ax)

    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=color,
            label=label,
            markersize=4,
        )
        for label, color in MATURATION_COLORS.items()
    ]
    region_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            color="#666666",
            label=region,
            markersize=4,
        )
        for region, marker in region_markers.items()
    ]
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="#666666",
            markersize=size,
            label=label,
        )
        for size, label in [(3, "3 samples"), (5, "10 samples"), (7, "20 samples")]
    ]
    fig.legend(
        handles=class_handles + region_handles + size_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
    )
    overall = associations[associations["scope"] == "Overall"]
    if not overall.empty:
        row = overall.iloc[0]
        fig.text(
            0.08,
            0.925,
            f"Overall adjusted β={row['coefficient']:.2f}; "
            f"95% publication-bootstrap CI "
            f"{row['publication_bootstrap_ci_low']:.2f}, "
            f"{row['publication_bootstrap_ci_high']:.2f}. "
            "Each identity has one healthy-reference x value repeated across "
            "organoid strata.",
            fontsize=5,
            color="#555555",
        )
    fig.suptitle(
        "Healthy segment separability does not predict stronger organoid origin proximity",
        x=0.08,
        y=0.99,
        ha="left",
        fontsize=8,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.29)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(
            C.OUT
            / f"fig5_g_healthy_segment_separability_correspondence.{extension}",
            dpi=300,
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    config = C.load_config()
    C.require_files(config)
    logger = C.setup_logging("05_segment_separability_correspondence")
    query = ad.read_h5ad(config["inputs"]["heoca_query"], backed="r")
    shared_labels = set(
        query.obs[config["columns"]["query_label_confident"]]
        .astype(str)
        .unique()
    ) - {config["filters"]["unknown_label"], "nan"}
    healthy_path = C.DATA / "fig5g_healthy_segment_separability.csv"
    repeats_path = (
        C.DATA / "fig5g_healthy_segment_separability_repeats.csv.gz"
    )
    healthy_exclusions_path = (
        C.DATA / "fig5g_healthy_cross_compartment_exclusions.csv"
    )
    cache_is_current = False
    if healthy_path.is_file() and repeats_path.is_file():
        cached_healthy = pd.read_csv(healthy_path)
        cache_is_current = (
            "cache_version" in cached_healthy.columns
            and cached_healthy["cache_version"]
            .eq(HEALTHY_CACHE_VERSION)
            .all()
            and healthy_exclusions_path.is_file()
        )
    if cache_is_current:
        healthy = cached_healthy
        repeats = pd.read_csv(repeats_path)
        healthy_exclusions = pd.read_csv(healthy_exclusions_path)
        logger.info("Using cached healthy segment-separability resamples")
    else:
        healthy, repeats, healthy_exclusions = healthy_segment_separability(
            config, shared_labels, logger
        )
        healthy.to_csv(healthy_path, index=False)
        repeats.to_csv(
            repeats_path,
            index=False,
            compression="gzip",
        )
        healthy_exclusions.to_csv(
            healthy_exclusions_path, index=False
        )
    organoid_sample, organoid_summary, organoid_exclusions = (
        organoid_origin_retention(config)
    )
    organoid_sample.to_csv(
        C.DATA / "fig5g_organoid_origin_proximity_by_sample.csv",
        index=False,
    )
    organoid_summary.to_csv(
        C.DATA / "fig5g_organoid_origin_proximity_summary.csv",
        index=False,
    )
    organoid_exclusions.to_csv(
        C.DATA / "fig5g_organoid_cross_compartment_exclusions.csv",
        index=False,
    )
    healthy_supported = healthy[healthy["supported"].fillna(False)].copy()
    point_data = organoid_summary.merge(
        healthy_supported,
        on="hgca_celltype_v1",
        how="inner",
        validate="many_to_one",
    )
    point_data["maturation_class"] = point_data[
        "hgca_celltype_v1"
    ].map(maturation_class)
    point_data["identity_well_supported"] = point_data["well_supported"]
    point_data.to_csv(
        C.DATA / "fig5g_joined_identity_source_region.csv", index=False
    )
    sample_data = organoid_sample.merge(
        healthy_supported,
        on="hgca_celltype_v1",
        how="inner",
        validate="many_to_one",
    )
    support = (
        sample_data.groupby("hgca_celltype_v1", observed=True)
        .agg(
            identity_n_samples=("sample_id", "nunique"),
            identity_n_publications=("publication_display", "nunique"),
        )
    )
    sample_data = sample_data.join(
        support, on="hgca_celltype_v1", how="left"
    )
    sample_data["identity_well_supported"] = (
        (sample_data["identity_n_samples"] >= 3)
        & (sample_data["identity_n_publications"] >= 2)
    )
    sample_data["maturation_class"] = sample_data[
        "hgca_celltype_v1"
    ].map(maturation_class)
    associations, class_test = association_models(
        sample_data, int(config["project"]["seed"])
    )
    associations.to_csv(
        C.DATA / "fig5g_association_models.csv", index=False
    )
    class_test.to_csv(
        C.DATA / "fig5g_identity_class_test.csv", index=False
    )
    render_figure(point_data, sample_data, associations, class_test)
    (
        branch_composition,
        branch_covariates,
        branch_definitions,
        filtered_counts,
    ) = build_branch_composition_tables()
    branch_composition.to_csv(
        C.DATA / "fig5h_sample_branch_composition.csv", index=False
    )
    branch_covariates.to_csv(
        C.DATA / "fig5h_branch_covariate_summary.csv", index=False
    )
    branch_definitions.to_csv(
        C.DATA / "fig5h_branch_identity_definitions.csv", index=False
    )
    (
        mapping_sample,
        mapping_summary,
        mapping_contrasts,
        confident_cells,
    ) = build_branch_mapping_quality(
        branch_composition, int(config["project"]["seed"])
    )
    mapping_sample.to_csv(
        C.DATA / "fig5h_branch_mapping_quality_by_sample.csv",
        index=False,
    )
    mapping_summary.to_csv(
        C.DATA / "fig5h_branch_mapping_quality_summary.csv",
        index=False,
    )
    mapping_contrasts.to_csv(
        C.DATA / "fig5h_branch_mapping_quality_contrasts.csv",
        index=False,
    )
    condition_effects, subtype_effects = build_condition_contrast_tables(
        branch_composition,
        filtered_counts,
        confident_cells,
        int(config["project"]["seed"]),
    )
    condition_effects.to_csv(
        C.DATA / "fig5h_secretory_condition_contrasts.csv",
        index=False,
    )
    subtype_effects.to_csv(
        C.DATA / "fig5h_secretory_subtype_contrasts.csv",
        index=False,
    )
    (
        psc_time_summary,
        psc_within_publication,
        psc_time_models,
    ) = build_psc_maturation_tables(
        branch_composition, filtered_counts
    )
    psc_time_summary.to_csv(
        C.DATA / "fig5h_psc_secretory_best4_time_summary.csv",
        index=False,
    )
    psc_within_publication.to_csv(
        C.DATA
        / "fig5h_psc_secretory_best4_within_publication.csv",
        index=False,
    )
    psc_time_models.to_csv(
        C.DATA / "fig5h_psc_secretory_best4_time_models.csv",
        index=False,
    )
    (
        best4_protocol_effects,
        best4_protocol_samples,
        best4_protocol_qc,
    ) = build_best4_protocol_response(
        int(config["project"]["seed"])
    )
    best4_protocol_effects.to_csv(
        C.DATA / "fig5h_e_best4_protocol_effects.csv",
        index=False,
    )
    best4_protocol_samples.to_csv(
        C.DATA / "fig5h_e_best4_protocol_sample_metrics.csv",
        index=False,
    )
    best4_protocol_qc.to_csv(
        C.DATA / "fig5h_e_best4_protocol_mapping_qc.csv",
        index=False,
    )
    render_branch_differentiation_figure(
        branch_composition,
        condition_effects,
        subtype_effects,
        mapping_sample,
        mapping_contrasts,
    )
    render_best4_protocol_response(
        best4_protocol_effects,
        best4_protocol_samples,
    )
    (
        maturation_distance_samples,
        maturation_distance_models,
        origin_time_coverage,
    ) = build_maturation_reference_distance_tables()
    maturation_distance_samples.to_csv(
        C.DATA / "fig5h_f_maturation_distance_samples.csv",
        index=False,
    )
    maturation_distance_models.to_csv(
        C.DATA / "fig5h_f_maturation_distance_models.csv",
        index=False,
    )
    origin_time_coverage.to_csv(
        C.DATA / "fig5h_f_origin_time_coverage.csv",
        index=False,
    )
    source_maturation_distance = build_source_maturation_distance_table()
    source_maturation_distance.to_csv(
        C.DATA / "fig5h_g_source_maturation_distance_samples.csv",
        index=False,
    )
    render_maturation_models_main(
        maturation_distance_samples,
        maturation_distance_models,
    )
    render_main_figure_composite(
        maturation_distance_samples,
        maturation_distance_models,
    )
    render_maturation_origin_supplement(
        origin_time_coverage,
    )
    render_source_faceted_maturation_distance(
        source_maturation_distance,
    )
    logger.info(
        "Wrote healthy separability correspondence for %s supported identities and epithelial branch panels",
        len(healthy_supported),
    )


if __name__ == "__main__":
    main()
