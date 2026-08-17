#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

import common as C


HEOCA_DOI = "10.1038/s41588-025-02182-6"
HEOCA_CITATION = "Xu et al., Nature Genetics (2025)"


def sample_metadata(obs: pd.DataFrame, workbook: pd.DataFrame, config: dict) -> pd.DataFrame:
    sample_key = config["columns"]["sample"]
    fields = [
        "publication",
        "sample_name",
        "derive",
        "time",
        "protocol",
        "molecular",
        "gel",
        "tissue",
        "tissue_ontology_term_id",
        "assay",
        "assay_ontology_term_id",
        "donor_id",
        "suspension_type",
    ]
    grouped = obs.groupby(sample_key, observed=True, sort=False)
    rows = []
    for sample, frame in grouped:
        row = {sample_key: str(sample)}
        for field in fields:
            row[field] = C.single_value(frame[field], str(sample), field)
        row["n_cells_total"] = len(frame)
        rows.append(row)
    metadata = pd.DataFrame(rows).set_index(sample_key)

    wb = workbook.copy()
    wb[sample_key] = wb[sample_key].astype(str)
    if wb[sample_key].duplicated().any():
        raise ValueError("HEOCA workbook contains duplicate sample_id values")
    wb = wb.set_index(sample_key)
    if set(wb.index) != set(metadata.index):
        raise ValueError(
            "HEOCA workbook and AnnData sample IDs differ: "
            f"workbook_only={sorted(set(wb.index) - set(metadata.index))}, "
            f"anndata_only={sorted(set(metadata.index) - set(wb.index))}"
        )
    metadata = metadata.join(
        wb[
            [
                "paper",
                "doi",
                "paper_link",
                "data_link",
                "detail_tissue",
                "tech",
                "derive",
            ]
        ].rename(columns={"derive": "derive_original"}),
        how="left",
    )
    metadata["publication_original"] = metadata["publication"]
    is_new = metadata["publication"].eq("Thispaper_Thispaper_2023")
    metadata["publication_display"] = metadata["publication"].replace(
        {"Thispaper_Thispaper_2023": "HEOCA newly generated"}
    )
    metadata["data_origin"] = np.where(
        is_new, "HEOCA newly generated", "Previously published"
    )
    metadata.loc[is_new, "paper"] = (
        "An integrated transcriptomic cell atlas of human endoderm-derived organoids"
    )
    metadata.loc[is_new, "doi"] = HEOCA_DOI
    metadata.loc[is_new, "paper_link"] = (
        "https://www.nature.com/articles/s41588-025-02182-6"
    )
    metadata.loc[is_new, "data_link"] = (
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE287233"
    )
    metadata["region_broad"] = C.broad_region(metadata["tissue"])
    metadata["region_detail"] = C.broad_region(metadata["detail_tissue"])
    metadata["time_class"] = C.time_class(metadata["time"])
    metadata["source_standardized"] = metadata["derive"].replace({"IPS": "PSC"})
    return metadata


def mapping_quality(obs: pd.DataFrame, metadata: pd.DataFrame, config: dict) -> pd.DataFrame:
    sample_key = config["columns"]["sample"]
    confident_key = config["columns"]["query_label_confident"]
    raw_key = config["columns"]["query_label_raw"]
    conf_key = config["columns"]["confidence"]
    entropy_key = config["columns"]["entropy"]
    unknown = config["filters"]["unknown_label"]
    epithelial = (
        obs[config["columns"]["original_level1"]]
        .astype(str)
        .str.lower()
        .eq(config["filters"]["query_epithelial_value"])
    )
    confident = obs[confident_key].astype(str).ne(unknown)
    grouped = obs.assign(_epithelial=epithelial, _confident=confident).groupby(
        sample_key, observed=True
    )
    rows = []
    for sample, frame in grouped:
        conf = frame["_confident"]
        rows.append(
            {
                sample_key: str(sample),
                "n_cells_total": len(frame),
                "n_epithelial_author": int(frame["_epithelial"].sum()),
                "n_non_epithelial_author": int((~frame["_epithelial"]).sum()),
                "n_confident_sysvi": int(conf.sum()),
                "n_low_confidence_sysvi": int((~conf).sum()),
                "fraction_confident_sysvi": float(conf.mean()),
                "median_sysvi_confidence": float(frame[conf_key].median()),
                "median_sysvi_entropy": float(frame[entropy_key].median()),
                "n_raw_hgca_subtypes": int(frame[raw_key].astype(str).nunique()),
                "n_confident_hgca_subtypes": int(
                    frame.loc[conf, confident_key].astype(str).nunique()
                ),
            }
        )
    result = pd.DataFrame(rows).set_index(sample_key)
    result = metadata.drop(columns=["n_cells_total"]).join(result)
    return result


def hierarchy_and_reference(config: dict, logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = ad.read_h5ad(config["inputs"]["hgca_epithelial"], backed="r")
    obs = reference.obs
    healthy = (
        obs[config["filters"]["reference_healthy_column"]]
        .astype(str)
        .eq(config["filters"]["reference_healthy_value"])
    )
    ref = obs.loc[healthy].copy()
    logger.info(
        "HGCA reference: %s cells total; %s healthy cells",
        f"{reference.n_obs:,}",
        f"{len(ref):,}",
    )
    label = config["columns"]["reference_label"]
    hierarchy_columns = [
        "hgca_celltype_level1",
        "hgca_celltype_level2",
        "hgca_celltype_level3",
        "hgca_celltype_level4",
        "hgca_celltype_level5",
    ]
    hierarchy = ref[[label] + hierarchy_columns].copy()
    for column in hierarchy.columns:
        hierarchy[column] = C.normalize_missing(hierarchy[column])
    hierarchy = hierarchy.drop_duplicates()
    duplicates = hierarchy.groupby(label, observed=True).size()
    if (duplicates > 1).any():
        raise ValueError(
            "HGCA v1 labels do not map uniquely to hierarchy: "
            f"{duplicates[duplicates > 1].to_dict()}"
        )
    hierarchy = hierarchy.set_index(label)

    grouped = ref.groupby(label, observed=True)
    summary = pd.DataFrame(
        {
            "healthy_hgca_cells": grouped.size(),
            "healthy_hgca_donors": grouped["donor_id"].nunique(),
            "healthy_hgca_datasets": grouped["dataset_id"].nunique(),
            "healthy_hgca_samples": grouped["sample_id"].nunique(),
        }
    )
    summary["healthy_hgca_prevalence"] = summary["healthy_hgca_cells"] / len(ref)
    summary = hierarchy.join(summary, how="outer")
    return hierarchy, summary


def rarefy_counts(
    counts: pd.DataFrame, depth: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sample, row in counts.iterrows():
        values = row.to_numpy(dtype=int)
        if values.sum() < depth:
            continue
        if hasattr(rng, "multivariate_hypergeometric"):
            draw = rng.multivariate_hypergeometric(values, depth)
        else:
            labels = np.repeat(np.arange(len(values)), values)
            selected = rng.choice(labels, depth, replace=False)
            draw = np.bincount(selected, minlength=len(values))
        rows.append(pd.Series(draw, index=counts.columns, name=sample))
    return pd.DataFrame(rows, columns=counts.columns).astype(int)


def composition_tables(
    obs: pd.DataFrame,
    metadata: pd.DataFrame,
    hierarchy: pd.DataFrame,
    reference_summary: pd.DataFrame,
    config: dict,
) -> None:
    sample_key = config["columns"]["sample"]
    confident_key = config["columns"]["query_label_confident"]
    raw_key = config["columns"]["query_label_raw"]
    conf_key = config["columns"]["confidence"]
    unknown = config["filters"]["unknown_label"]
    primary_depth = int(config["filters"]["primary_min_confident_cells"])
    seed = int(config["project"]["seed"])

    confident = obs[confident_key].astype(str).ne(unknown)
    confident_obs = obs.loc[confident].copy()
    confident_obs[confident_key] = confident_obs[confident_key].astype(str)
    counts = pd.crosstab(
        confident_obs[sample_key].astype(str), confident_obs[confident_key]
    )
    labels = sorted(set(hierarchy.index.astype(str)) | set(counts.columns))
    counts = counts.reindex(index=metadata.index, columns=labels, fill_value=0)
    counts.to_csv(C.DATA / "sample_subtype_counts_confident.csv")
    proportions = counts.div(counts.sum(axis=1), axis=0)
    proportions.to_csv(C.DATA / "sample_subtype_proportions_confident.csv")

    rarefied = rarefy_counts(counts, primary_depth, seed)
    rarefied.to_csv(C.DATA / f"sample_subtype_counts_rarefied_{primary_depth}.csv")
    rarefied.gt(0).astype(int).to_csv(
        C.DATA / f"sample_subtype_detection_rarefied_{primary_depth}.csv"
    )

    raw_counts = pd.crosstab(
        obs[sample_key].astype(str), obs[raw_key].astype(str)
    ).reindex(index=metadata.index, columns=labels, fill_value=0)
    raw_counts.to_csv(C.DATA / "sample_subtype_counts_unthresholded.csv")

    query_group = confident_obs.groupby(confident_key, observed=True)
    query_summary = pd.DataFrame(
        {
            "heoca_cells_confident": query_group.size(),
            "heoca_samples_detected": query_group[sample_key].nunique(),
            "heoca_publications_detected": query_group["publication"].nunique(),
            "median_mapping_confidence": query_group[conf_key].median(),
        }
    )
    query_summary["heoca_detection_rate_samples"] = (
        query_summary["heoca_samples_detected"] / len(metadata)
    )
    rarefied_detected = rarefied.gt(0).sum(axis=0)
    query_summary["heoca_samples_detected_rarefied"] = rarefied_detected
    query_summary["heoca_detection_rate_rarefied"] = (
        rarefied_detected / len(rarefied)
    )
    query_summary["rarefaction_depth"] = primary_depth
    query_summary["rarefied_sample_denominator"] = len(rarefied)
    med_when = proportions.replace(0, np.nan).median(axis=0, skipna=True)
    query_summary["median_abundance_when_detected"] = med_when
    query_summary["median_abundance_when_detected_rarefied"] = (
        rarefied.div(primary_depth)
        .replace(0, np.nan)
        .median(axis=0, skipna=True)
    )
    for field, output in [
        ("derive", "source_types"),
        ("tissue", "anatomical_regions"),
    ]:
        values = (
            confident_obs.groupby(confident_key, observed=True)[field]
            .agg(lambda x: "|".join(sorted(set(C.normalize_missing(x).dropna()))))
        )
        query_summary[output] = values
    pub_counts = pd.crosstab(
        confident_obs[confident_key].astype(str),
        confident_obs["publication"].astype(str),
    )
    query_summary["maximum_publication_fraction"] = (
        pub_counts.max(axis=1) / pub_counts.sum(axis=1)
    )
    query_summary["publication_dependent"] = (
        query_summary["maximum_publication_fraction"] >= 0.75
    )
    query_summary["representation_class"] = "absent"
    query_summary.loc[
        query_summary["heoca_samples_detected_rarefied"].between(1, 2),
        "representation_class",
    ] = "rare"
    query_summary.loc[
        query_summary["heoca_samples_detected_rarefied"].between(3, 24),
        "representation_class",
    ] = "recurrent"
    query_summary.loc[
        query_summary["heoca_samples_detected_rarefied"] >= 25,
        "representation_class",
    ] = "broadly represented"

    capability = reference_summary.join(query_summary, how="outer")
    capability.index.name = "hgca_celltype_v1"
    capability["representation_class"] = capability[
        "representation_class"
    ].fillna("absent")
    capability["heoca_cells_confident"] = capability[
        "heoca_cells_confident"
    ].fillna(0).astype(int)
    capability.to_csv(C.DATA / "supp_subtype_capability_table.csv")

    confidence_matrix = (
        confident_obs.groupby([sample_key, confident_key], observed=True)[conf_key]
        .median()
        .unstack(fill_value=np.nan)
        .reindex(index=rarefied.index, columns=rarefied.columns)
    )
    confidence_matrix.to_csv(C.DATA / "sample_subtype_median_confidence.csv")
    publication_counts = pd.crosstab(
        confident_obs[confident_key].astype(str),
        confident_obs["publication"].astype(str),
    ).reindex(index=labels, fill_value=0)
    publication_counts.to_csv(C.DATA / "publication_subtype_counts.csv")
    publication_confidence = (
        confident_obs.groupby([confident_key, "publication"], observed=True)[conf_key]
        .median()
        .unstack()
        .reindex(index=labels, columns=publication_counts.columns)
    )
    publication_confidence.to_csv(C.DATA / "publication_subtype_median_confidence.csv")

    capability_rows = []
    capability_fields = [
        ("source_standardized", "Organoid source"),
        ("region_broad", "Declared region"),
        ("time_class", "Maturation/time"),
        ("gel", "Matrix"),
        ("molecular", "Molecular condition"),
        ("protocol", "Protocol"),
    ]
    rarefied_proportions = rarefied / primary_depth
    for field, display in capability_fields:
        values = C.normalize_missing(metadata.loc[rarefied.index, field]).fillna(
            "Not reported"
        )
        for value, sample_ids in values.groupby(values, observed=True).groups.items():
            sample_ids = pd.Index(sample_ids)
            group_counts = rarefied.loc[sample_ids]
            group_props = rarefied_proportions.loc[sample_ids]
            for subtype_name in rarefied.columns:
                detected = group_counts[subtype_name] > 0
                capability_rows.append(
                    {
                        "field": field,
                        "field_display": display,
                        "value": value,
                        "hgca_celltype_v1": subtype_name,
                        "n_samples": len(sample_ids),
                        "n_publications": metadata.loc[
                            sample_ids, "publication_display"
                        ].nunique(),
                        "samples_detected": int(detected.sum()),
                        "detection_rate_rarefied": float(detected.mean()),
                        "median_abundance": float(
                            group_props.loc[detected, subtype_name].median()
                            if detected.any()
                            else 0
                        ),
                    }
                )
    pd.DataFrame(capability_rows).to_csv(
        C.DATA / "supp_protocol_subtype_capability_long.csv", index=False
    )


def extracted_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    patterns = {
        "TNF": r"\bTNF\b",
        "IL22": r"\bIL22\b",
        "EGF": r"\bEGF\b|(?<![A-Z])\d+EGF",
        "EREG": r"\bEREG\b",
        "NRG1": r"\bNRG1\b|(?<![A-Z])\d+NRG1",
        "WNT3A": r"\bWNT3A\b",
        "RSPO": r"\bRSPO\b",
        "transplanted_HIO": r"\btHIO\b",
        "kidney_capsule": r"kidneycapsule|kidney_capsule",
        "mesentery": r"\bmesentery\b",
        "alginate": r"\balg[ei]nate\b",
        "Matrigel": r"\bmatrigel\b",
        "suspension": r"\bsuspension\b",
        "EEC_enriched": r"\bEEC",
    }
    rows = []
    for sample, row in metadata.iterrows():
        source = str(row["sample_name"])
        for field, pattern in patterns.items():
            if re.search(pattern, source, flags=re.IGNORECASE):
                rows.append(
                    {
                        "sample_id": sample,
                        "candidate_field": field,
                        "extracted_value": field,
                        "source_string": source,
                        "confidence": "high"
                        if field in {"kidney_capsule", "mesentery", "alginate", "Matrigel"}
                        else "medium",
                        "verification_status": "unverified_sample_name_extraction",
                    }
                )
        for value in re.findall(r"(?:day|D)(\d+)|(\d+)d\b", source, re.I):
            day = next((item for item in value if item), None)
            if day:
                rows.append(
                    {
                        "sample_id": sample,
                        "candidate_field": "culture_day",
                        "extracted_value": day,
                        "source_string": source,
                        "confidence": "medium",
                        "verification_status": "unverified_sample_name_extraction",
                    }
                )
    return pd.DataFrame(rows)


def metadata_audit(metadata: pd.DataFrame) -> pd.DataFrame:
    candidates = {
        "sample_id": pd.Series(metadata.index, index=metadata.index),
        "publication": metadata["publication_display"],
        "organoid_source": metadata["source_standardized"],
        "region_broad": metadata["region_broad"],
        "region_detail": metadata["region_detail"],
        "time": metadata["time"],
        "time_class": metadata["time_class"],
        "matrix": metadata["gel"],
        "molecular_condition": metadata["molecular"],
        "protocol_or_transplant": metadata["protocol"],
        "sequencing_protocol_original": metadata["tech"],
        "sequencing_protocol_standardized": metadata["assay"],
        "donor_or_cell_line": metadata["donor_id"],
    }
    rows = []
    for field, values in candidates.items():
        values = C.normalize_missing(values)
        populated = int(values.notna().sum())
        publication_specific = False
        if field != "publication":
            cross = pd.DataFrame(
                {"value": values, "publication": metadata["publication"]}
            ).dropna()
            if not cross.empty:
                publication_specific = bool(
                    (cross.groupby("value", observed=True)["publication"].nunique() == 1).mean()
                    >= 0.75
                )
        standardized = field in {
            "sample_id",
            "publication",
            "organoid_source",
            "region_broad",
            "sequencing_protocol_original",
            "sequencing_protocol_standardized",
        }
        if field == "donor_or_cell_line":
            usability = "neither: identifiers are mostly sample-derived"
        elif populated == len(metadata) and values.nunique() >= 2:
            usability = "descriptive"
        elif populated >= 30 and values.nunique() >= 2:
            usability = "descriptive; within-study testing only"
        else:
            usability = "audit only"
        rows.append(
            {
                "field": field,
                "n_samples_populated": populated,
                "percent_samples_populated": 100 * populated / len(metadata),
                "n_unique_values": int(values.nunique(dropna=True)),
                "standardized": standardized,
                "publication_specific": publication_specific,
                "sample_name_contains_additional_information": field
                in {
                    "time",
                    "matrix",
                    "molecular_condition",
                    "protocol_or_transplant",
                },
                "recommended_use": usability,
            }
        )
    return pd.DataFrame(rows)


def segment_tables(
    obs: pd.DataFrame, metadata: pd.DataFrame, config: dict, logger
) -> None:
    path = config["inputs"]["sysvi_distances"]
    columns = [
        "cell_id",
        "d_nn1",
        "d_knn_mean",
        "d_origin",
        "d_best_other",
        "delta_best_other_minus_origin",
        "origin_tissue_label",
        "nearest_tissue_label",
        "best_other_tissue_label",
        "origin_rank_among_seen_tissues",
        "n_origin_neighbors_seen",
        "origin_seen_in_k_total",
        "sample_id",
    ]
    distance = pd.read_csv(path, usecols=columns)
    distance["query_obs_name"] = distance["cell_id"].astype(str).str.replace(
        r"-Organoid$", "", regex=True
    )
    if distance["query_obs_name"].duplicated().any():
        raise ValueError("Normalized sysVI distance cell IDs are not unique")
    query_columns = [
        config["columns"]["query_label_confident"],
        config["columns"]["query_label_raw"],
        config["columns"]["confidence"],
        config["columns"]["entropy"],
    ]
    query = obs[query_columns].copy()
    query.index = query.index.astype(str)
    joined = distance.join(
        query, on="query_obs_name", how="left", validate="one_to_one"
    )
    if joined[query_columns[0]].isna().any():
        raise ValueError("Per-cell sysVI distances do not align to query obs_names")
    joined["origin_region"] = C.broad_region(joined["origin_tissue_label"])
    joined["nearest_region"] = C.broad_region(joined["nearest_tissue_label"])
    joined["segment_evaluable"] = joined["origin_region"].isin(
        ["Duodenum", "Ileum", "Colon"]
    ) & joined["nearest_region"].isin(["Duodenum", "Ileum", "Colon"])
    joined["segment_match"] = (
        joined["origin_region"] == joined["nearest_region"]
    )
    unknown = config["filters"]["unknown_label"]
    label = config["columns"]["query_label_confident"]
    joined["confident"] = joined[label].astype(str).ne(unknown)
    joined = joined.join(
        metadata[
            [
                "source_standardized",
                "publication_display",
                "region_broad",
                "time_class",
                "gel",
                "molecular",
                "protocol",
            ]
        ],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )

    label_distance = joined.groupby(label, observed=True)["d_nn1"].agg(
        ["median", lambda values: np.median(np.abs(values - np.median(values)))]
    )
    label_distance.columns = ["label_distance_median", "label_distance_mad"]
    joined = joined.join(label_distance, on=label)
    denominator = 1.4826 * joined["label_distance_mad"].replace(0, np.nan)
    joined["distance_robust_z_within_label"] = (
        joined["d_nn1"] - joined["label_distance_median"]
    ) / denominator
    joined["distance_outlier"] = (
        joined["distance_robust_z_within_label"] > 3.5
    ).fillna(False)
    joined["strict_mapping_pass"] = joined["confident"] & ~joined["distance_outlier"]
    denominator = joined["d_best_other"] + joined["d_origin"]
    joined["relative_origin_proximity"] = (
        2
        * (joined["d_best_other"] - joined["d_origin"])
        / denominator.replace(0, np.nan)
    )
    joined[
        [
            "cell_id",
            "query_obs_name",
            "sample_id",
            label,
            config["columns"]["query_label_raw"],
            config["columns"]["confidence"],
            "d_nn1",
            "d_knn_mean",
            "distance_robust_z_within_label",
            "confident",
            "distance_outlier",
            "strict_mapping_pass",
            "d_origin",
            "d_best_other",
            "delta_best_other_minus_origin",
            "relative_origin_proximity",
            "origin_rank_among_seen_tissues",
            "n_origin_neighbors_seen",
            "origin_region",
            "nearest_region",
            "segment_evaluable",
            "segment_match",
        ]
    ].to_csv(C.DATA / "per_cell_mapping_qc_flags.csv.gz", index=False, compression="gzip")

    origin_specific = joined.loc[
        joined["strict_mapping_pass"]
        & joined["origin_region"].isin(["Duodenum", "Ileum", "Colon"])
        & joined["relative_origin_proximity"].notna()
    ].copy()
    cells_per_group = int(
        config["filters"]["origin_margin_cells_per_sample_subtype"]
    )
    rng = np.random.default_rng(int(config["project"]["seed"]))
    rarefied_groups = []
    for _, frame in origin_specific.groupby(
        ["sample_id", label], observed=True, sort=True
    ):
        if len(frame) < cells_per_group:
            continue
        rarefied_groups.append(
            frame.sample(
                n=cells_per_group,
                replace=False,
                random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
            )
        )
    origin_rarefied = pd.concat(rarefied_groups, ignore_index=True)
    origin_rarefied = origin_rarefied.rename(
        columns={label: "hgca_celltype_v1"}
    )
    origin_rarefied[
        [
            "cell_id",
            "sample_id",
            "publication_display",
            "source_standardized",
            "origin_region",
            "hgca_celltype_v1",
            "d_origin",
            "d_best_other",
            "delta_best_other_minus_origin",
            "relative_origin_proximity",
            "origin_rank_among_seen_tissues",
            "n_origin_neighbors_seen",
        ]
    ].to_csv(
        C.DATA / "fig5b_origin_proximity_rarefied_cells.csv.gz",
        index=False,
        compression="gzip",
    )
    origin_summary = (
        origin_rarefied.groupby(
            [
                "source_standardized",
                "origin_region",
                "hgca_celltype_v1",
            ],
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
            q10_relative_origin_proximity=(
                "relative_origin_proximity",
                lambda values: values.quantile(0.1),
            ),
            q25_relative_origin_proximity=(
                "relative_origin_proximity",
                lambda values: values.quantile(0.25),
            ),
            q75_relative_origin_proximity=(
                "relative_origin_proximity",
                lambda values: values.quantile(0.75),
            ),
            q90_relative_origin_proximity=(
                "relative_origin_proximity",
                lambda values: values.quantile(0.9),
            ),
            fraction_cells_origin_closer=(
                "relative_origin_proximity",
                lambda values: (values > 0).mean(),
            ),
            median_origin_rank=("origin_rank_among_seen_tissues", "median"),
        )
        .reset_index()
    )
    origin_summary.to_csv(
        C.DATA / "fig5b_origin_proximity_summary.csv", index=False
    )

    evaluable = joined.loc[joined["segment_evaluable"] & joined["confident"]].copy()
    logger.info(
        "Segment identity evaluable for %s confident cells across %s samples",
        f"{len(evaluable):,}",
        evaluable["sample_id"].nunique(),
    )
    subtype = (
        evaluable.groupby([label, "origin_region"], observed=True)
        .agg(
            n_cells=("segment_match", "size"),
            segment_match_fraction=("segment_match", "mean"),
            median_nn_distance=("d_nn1", "median"),
            median_knn_distance=("d_knn_mean", "median"),
        )
        .reset_index()
        .rename(columns={label: "hgca_celltype_v1"})
    )
    p = subtype["segment_match_fraction"]
    n = subtype["n_cells"]
    subtype["segment_match_se"] = np.sqrt(p * (1 - p) / n)
    subtype.to_csv(C.DATA / "fig5b_subtype_segment_identity.csv", index=False)

    subtype_stratified = (
        evaluable.groupby(
            ["source_standardized", "origin_region", label], observed=True
        )
        .agg(
            n_cells=("segment_match", "size"),
            n_samples=("sample_id", "nunique"),
            n_publications=("publication_display", "nunique"),
            segment_match_fraction=("segment_match", "mean"),
            median_mapping_confidence=(
                config["columns"]["confidence"],
                "median",
            ),
            median_nn_distance=("d_nn1", "median"),
        )
        .reset_index()
        .rename(columns={label: "hgca_celltype_v1"})
    )
    subtype_stratified.to_csv(
        C.DATA / "fig5b_source_stratified_subtype_segment_identity.csv",
        index=False,
    )

    nearest_region = joined.loc[joined["confident"]].copy()
    nearest_region["nearest_region_plot"] = nearest_region["nearest_region"].where(
        nearest_region["nearest_region"].isin(
            ["Duodenum", "Jejunum", "Ileum", "Colon"]
        ),
        "Other",
    )
    nearest_counts = pd.crosstab(
        nearest_region["sample_id"], nearest_region["nearest_region_plot"]
    ).reindex(
        index=metadata.index,
        columns=["Duodenum", "Jejunum", "Ileum", "Colon", "Other"],
        fill_value=0,
    )
    nearest_proportions = nearest_counts.div(
        nearest_counts.sum(axis=1).replace(0, np.nan), axis=0
    )
    nearest_counts.to_csv(C.DATA / "fig5b_sample_nearest_region_counts.csv")
    nearest_proportions.to_csv(
        C.DATA / "fig5b_sample_nearest_region_proportions.csv"
    )

    sample_subtype = (
        evaluable.groupby(
            ["sample_id", "origin_region", label], observed=True
        )
        .agg(
            n_cells=("segment_match", "size"),
            segment_match_fraction=("segment_match", "mean"),
        )
        .reset_index()
        .rename(columns={label: "hgca_celltype_v1"})
    )
    sample_subtype = sample_subtype.join(
        metadata[["publication_display", "source_standardized"]],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    sample_subtype.to_csv(
        C.DATA / "fig5b_sample_subtype_segment_identity.csv", index=False
    )
    sample_equal_subtype = (
        sample_subtype.groupby(
            ["origin_region", "hgca_celltype_v1"], observed=True
        )
        .agg(
            n_samples=("sample_id", "nunique"),
            n_publications=("publication_display", "nunique"),
            median_sample_concordance=("segment_match_fraction", "median"),
            q25_sample_concordance=(
                "segment_match_fraction",
                lambda values: values.quantile(0.25),
            ),
            q75_sample_concordance=(
                "segment_match_fraction",
                lambda values: values.quantile(0.75),
            ),
        )
        .reset_index()
    )
    sample_equal_subtype.to_csv(
        C.DATA / "fig5b_sample_equal_subtype_segment_identity.csv", index=False
    )

    per_cell_summary = (
        joined.groupby("sample_id", observed=True)
        .agg(
            n_cells=("cell_id", "size"),
            fraction_confident=("confident", "mean"),
            fraction_distance_outlier=("distance_outlier", "mean"),
            fraction_strict_mapping_pass=("strict_mapping_pass", "mean"),
            median_mapping_confidence=(
                config["columns"]["confidence"],
                "median",
            ),
            median_mapping_entropy=(
                config["columns"]["entropy"],
                "median",
            ),
            median_nn_distance=("d_nn1", "median"),
            median_knn_distance=("d_knn_mean", "median"),
        )
        .join(metadata, how="left")
    )
    per_cell_summary.to_csv(C.DATA / "supp_mapping_qc_by_sample.csv")
    strict_counts = pd.crosstab(
        joined.loc[joined["strict_mapping_pass"], "sample_id"].astype(str),
        joined.loc[joined["strict_mapping_pass"], label].astype(str),
    )
    strict_counts.to_csv(C.DATA / "sample_subtype_counts_strict_mapping.csv")

    sample = (
        evaluable.groupby("sample_id", observed=True)
        .agg(
            n_segment_evaluable_cells=("segment_match", "size"),
            segment_identity_fraction=("segment_match", "mean"),
            median_nn_distance=("d_nn1", "median"),
            median_knn_distance=("d_knn_mean", "median"),
        )
        .join(metadata, how="left")
    )
    sample.to_csv(C.DATA / "fig5b_sample_segment_identity.csv")

    condition_rows = []
    for field in [
        "source_standardized",
        "region_broad",
        "time_class",
        "molecular",
        "gel",
        "protocol",
        "publication_display",
    ]:
        available = sample.dropna(subset=[field, "segment_identity_fraction"])
        for value, frame in available.groupby(field, observed=True):
            if len(frame) < 2:
                continue
            condition_rows.append(
                {
                    "field": field,
                    "value": value,
                    "n_samples": len(frame),
                    "n_publications": frame["publication"].nunique(),
                    "median_segment_identity": frame[
                        "segment_identity_fraction"
                    ].median(),
                    "mean_segment_identity": frame[
                        "segment_identity_fraction"
                    ].mean(),
                }
            )
    pd.DataFrame(condition_rows).to_csv(
        C.DATA / "fig5b_condition_segment_identity.csv", index=False
    )


def write_hca_metadata_recommendations() -> None:
    rows = [
        ("donor_or_cell_line", "Current donor_id is mostly sample-derived", "Stable donor, cell-line and clone identifiers"),
        ("tissue_source", "38/98 samples have only nonspecific intestine", "Tissue source plus standardized anatomical region"),
        ("stem_cell_source", "ASC/FSC/PSC is complete but IPS was recoded to PSC", "Original and harmonized stem-cell source"),
        ("basal_medium", "Not represented", "Basal medium formulation and lot"),
        ("growth_factors", "Only 33/98 molecular fields are populated", "Factor names, concentrations and vendors"),
        ("media_schedule", "Time is populated for 69/98 but media switches are absent", "Media-switch schedule and collection time"),
        ("matrix", "Matrix is populated for 37/98", "Matrix type, concentration, lot and geometry"),
        ("culture_geometry", "Suspension appears inconsistently in gel/sample names", "2D/3D, suspension, chip, flow and scaffold geometry"),
        ("differentiation_duration", "Mixed early/late labels and day values", "Numeric duration with units and developmental stage"),
        ("passage", "Mostly present only in selected sample names", "Passage at perturbation and collection"),
        ("perturbations", "TNF, IL22, EGF, EREG, NRG1, WNT3A and RSPO appear in names", "Structured perturbation, dose, timing and control"),
        ("co_culture", "HIO-EC/EEC enrichment is incompletely encoded", "Co-culture cell type, ratio and introduction time"),
        ("transplantation", "Protocol is populated for only 10/98", "Host, anatomical site, duration and recovery"),
        ("environment", "Oxygen and mechanical conditions are absent", "Oxygen, flow, stretch and pressure"),
        ("replicate_structure", "True donor/experiment pairing is unavailable", "Biological donor, experiment, well and technical replicate IDs"),
        ("sequencing", "One CEL-seq2/Seq-Well S3 conflict exists", "Library, sequencing and dissociation protocols with ontology IDs"),
    ]
    pd.DataFrame(
        rows, columns=["category", "observed_heoca_deficiency", "recommended_hca_field"]
    ).to_csv(C.DATA / "hca_organoid_metadata_recommendations.csv", index=False)


def main() -> None:
    config = C.load_config()
    C.require_files(config)
    C.set_seed(int(config["project"]["seed"]))
    logger = C.setup_logging("01_build_tables")
    C.ensure_session_info()

    query = ad.read_h5ad(config["inputs"]["heoca_query"], backed="r")
    obs = query.obs.copy()
    workbook = pd.read_excel(
        config["inputs"]["heoca_metadata"], sheet_name="Supplementary_Table_1"
    )
    logger.info("HEOCA query: %s cells x %s genes", f"{query.n_obs:,}", f"{query.n_vars:,}")
    metadata = sample_metadata(obs, workbook, config)
    quality = mapping_quality(obs, metadata, config)
    hierarchy, reference_summary = hierarchy_and_reference(config, logger)

    metadata.to_csv(C.DATA / "sample_metadata.csv")
    quality.to_csv(C.DATA / "sample_mapping_qc.csv")
    hierarchy.to_csv(C.DATA / "hgca_epithelial_hierarchy.csv")
    reference_summary.to_csv(C.DATA / "hgca_healthy_subtype_summary.csv")
    metadata_audit(metadata).to_csv(C.DATA / "metadata_completeness_audit.csv", index=False)
    extracted_metadata(metadata).to_csv(
        C.DATA / "metadata_extracted_from_sample_names.csv", index=False
    )
    write_hca_metadata_recommendations()
    composition_tables(obs, metadata, hierarchy, reference_summary, config)
    segment_tables(obs, metadata, config, logger)

    warnings = [
        "Frozen sysVI predictions are treated as primary; the sysVI checkpoint and mapping code are unavailable.",
        "The sysVI reference used an older HGCA export; current healthy-reference summaries use sampled_site_condition == healthy.",
        "The AnnData normalizes workbook IPS to PSC; both values are retained.",
        "donor_id is largely sample-derived and is not treated as an independent biological replicate identifier.",
        "Protocol, matrix, molecular condition and time are incomplete and publication-confounded.",
        "Sample-name metadata extraction is unverified and excluded from confirmatory tests.",
        "CEL-seq2 in the workbook is represented as Seq-Well S3 for one AnnData sample and requires source-level resolution.",
    ]
    (C.LOGS / "warnings_and_unresolved_issues.txt").write_text(
        "\n".join(f"- {warning}" for warning in warnings) + "\n"
    )
    logger.info("Wrote table-building outputs to %s", C.DATA)


if __name__ == "__main__":
    main()
