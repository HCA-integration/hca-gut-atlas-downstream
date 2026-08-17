"""Recompute CLR composition + Mann-Whitney tables for the sampling-depth story
straight from the current integrated atlas obs.

Why this exists: the previously stored CLR volcano CSVs were generated on an
older cell-type set that included a `BEST4 CHE` subtype (since rejected by
collaborators). Reading `hgca_celltype_v1` off the live objects gives the
current label set (BEST4 Enterocytes as its own full label, no CHE) and lets us
add the full-thickness (EPI_LP_MUSC) contrast the story now needs.

Design (matches vignettes/Composition_patpy_variance.ipynb):
  - CLR is computed WITHIN each lineage (per-sample crosstab over that lineage's
    cell types, pseudocount 0.5, centred log-ratio), then concatenated. This
    avoids the between-lineage sorting/enrichment artifact.
  - Every cell type carries its lineage (from its source object) so figures can
    colour by lineage, and a `is_follicle_tlo` flag for the follicle/TLO niche.
  - Mann-Whitney U per cell type; BH-FDR across all tested cell types per contrast.

Contrasts written to ../data/:
  clr_wilcoxon_collection.csv            biopsy (A) vs surgical resection (B)
  clr_wilcoxon_radial_epi_lp.csv         EPI (A) vs LP (B), pure layers only
  clr_wilcoxon_full_thickness.csv        rest (A) vs EPI_LP_MUSC full-thickness (B)
  by_tissue/<t>/clr_wilcoxon_{collection,full_thickness}_<t>.csv
  clr_long.csv                           per sample x cell type CLR + metadata (for splines)
  celltype_lineage_map.csv               cell type -> lineage, is_follicle_tlo
"""
from __future__ import annotations

import argparse
import os
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parents[3]
_OBJECTS = Path(os.environ["HGCA_OBJECTS"]) if os.environ.get("HGCA_OBJECTS") else None
LINEAGES = (
    {
        "epithelial": _OBJECTS / "epithelial.h5ad",
        "lymphoid": _OBJECTS / "lymphoid.h5ad",
        "myeloid": _OBJECTS / "myeloid.h5ad",
        "stroma": _OBJECTS / "stroma.h5ad",
    }
    if _OBJECTS is not None
    else {}
)
OUT = Path(__file__).resolve().parent.parent / "data"
DEMO_H5AD = REPO_ROOT / "data" / "demo" / "hgca_all_lineages_v1_demo.h5ad"
EXPECTED_CLR = REPO_ROOT / "data" / "demo" / "expected" / "clr"

CT_COL = "hgca_celltype_v1"
META_COLS = [
    "sample_id", "donor_id", "dataset_id", "tissue_level_1",
    "sampled_site_condition", "radial_tissue_term",
    "sample_preservation_method", "sex_ontology_term", "age_range", "assay",
    "sample_collection_method", "sequenced_fragment", "gene_annotation_version",
]
AUDIT_COVARIATES = [
    "sampled_site_condition", "radial_tissue_term",
    "sample_preservation_method", "sex_ontology_term", "age_range", "assay",
    "sample_collection_method", "sequenced_fragment", "gene_annotation_version",
]
CANONICAL_SEGMENTS = ["duodenum", "jejunum", "ileum", "colon"]
UNKNOWN_VALUES = {"", "unknown", "nan", "none", "n/a", "na", "not applicable"}

PSEUDOCOUNT = 0.5
MIN_SAMPLES_PER_GROUP = 5
MIN_PRESENT_FRAC = 0.5
FDR_ALPHA = 0.05

FOLLICLE_TLO = {
    "GC B Light Zone (GC B LZ)", "GC B Dark Zone (GC B DZ)",
    "Follicular Dendritic Cells (fDC)", "Fibroblastic Reticular Cells (FRC)",
    "Marginal Reticular Cells (MRC)",
    "Mesenchymal Lymphoid Tissue Organizer Cells (mLTo Cells)",
    "Follicle Associated Resident Macrophages", "Perivascular Resident Macrophages",
    "CD4 Tfh", "CD4 Tfr", "Lymphatic Endothelial", "Medullary Sinus Endothelial",
}

AGE_ORDER = {
    "0-9": 5, "10-19": 15, "20-29": 25, "30-39": 35, "40-49": 45,
    "50-59": 55, "60-69": 65, "70-79": 75, "80-89": 85,
}
LEVEL1_TO_LINEAGE = {
    "epithelial": "epithelial",
    "lymphoid": "lymphoid",
    "myeloid": "myeloid",
    "stroma": "stroma",
    "stromal": "stroma",
}


def _clean(s: str) -> str:
    return " ".join(str(s).split())  # collapse embedded newlines / double spaces


def read_obs(path: Path, cols) -> pd.DataFrame:
    """Fast obs read via h5py for the requested (categorical or plain) columns."""
    out = {}
    with h5py.File(path, "r") as f:
        obs = f["obs"]
        n = None
        for c in cols:
            if c not in obs:
                continue
            node = obs[c]
            if isinstance(node, h5py.Group) and "categories" in node and "codes" in node:
                cats = [x.decode() if isinstance(x, bytes) else x for x in node["categories"][:]]
                codes = node["codes"][:]
                vals = pd.Categorical.from_codes(codes, categories=cats)
                out[c] = pd.Series(vals).astype(object)
            else:
                arr = node[:]
                arr = [x.decode() if isinstance(x, bytes) else x for x in arr]
                out[c] = pd.Series(arr)
            n = len(out[c])
    df = pd.DataFrame(out)
    return df


def compute_clr(counts: pd.DataFrame) -> pd.DataFrame:
    x = counts.astype(float) + PSEUDOCOUNT
    prop = x.div(x.sum(axis=1), axis=0)
    logx = np.log(prop)
    clr = logx.sub(logx.mean(axis=1), axis=0)
    return clr


def mode_or_nan(s: pd.Series):
    s = s.dropna()
    if s.empty:
        return np.nan
    m = s.mode()
    return m.iloc[0] if len(m) else np.nan


def clean_category(value):
    if pd.isna(value):
        return np.nan
    value = _clean(value)
    return np.nan if value.lower() in UNKNOWN_VALUES else value


def one_way_omega_squared(values: pd.Series, groups: pd.Series) -> float:
    """One-way omega squared for one cell type and one categorical covariate."""
    frame = pd.DataFrame({"value": values, "group": groups}).dropna()
    if frame.empty or frame["group"].nunique() < 2:
        return np.nan
    grand = frame["value"].mean()
    grouped = frame.groupby("group", observed=True)["value"]
    ss_between = sum(len(x) * (x.mean() - grand) ** 2 for _, x in grouped)
    ss_total = ((frame["value"] - grand) ** 2).sum()
    k = frame["group"].nunique()
    n = len(frame)
    if n <= k or ss_total <= 0:
        return 0.0
    ss_within = max(ss_total - ss_between, 0.0)
    ms_within = ss_within / (n - k)
    omega = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)
    return float(max(0.0, min(1.0, omega)))


def add_group_fdr(
    frame: pd.DataFrame, p_col: str, group_cols: list[str], out_col: str
) -> pd.DataFrame:
    frame[out_col] = np.nan
    if frame.empty:
        return frame
    for _, idx in frame.groupby(group_cols, dropna=False).groups.items():
        idx = list(idx)
        valid = frame.loc[idx, p_col].notna()
        valid_idx = list(frame.loc[idx].index[valid])
        if not valid_idx:
            continue
        frame.loc[valid_idx, out_col] = multipletests(
            frame.loc[valid_idx, p_col], method="fdr_bh"
        )[1]
    return frame


def _lineage_groups(all_cells: Path | None):
    """Yield (lineage, obs) from four lineage objects or one all-cells h5ad."""
    if all_cells is None:
        for ln, path in LINEAGES.items():
            print(f"[{ln}] reading obs …")
            obs = read_obs(path, [CT_COL] + META_COLS)
            yield ln, obs
        return
    print(f"[all-cells] reading obs from {all_cells}")
    obs = read_obs(all_cells, [CT_COL, "hgca_celltype_level1"] + META_COLS)
    if "hgca_celltype_level1" not in obs:
        raise SystemExit(f"{all_cells} is missing hgca_celltype_level1")
    obs["_lineage"] = (
        obs["hgca_celltype_level1"].map(_clean).str.lower().map(LEVEL1_TO_LINEAGE)
    )
    n_unmapped = int(obs["_lineage"].isna().sum())
    if n_unmapped:
        print(f"  warning: {n_unmapped} cells with unmapped hgca_celltype_level1")
    for ln, sub in obs.dropna(subset=["_lineage"]).groupby("_lineage", sort=True):
        print(f"[{ln}] {len(sub):,} cells from all-cells object")
        yield ln, sub.drop(columns=["_lineage", "hgca_celltype_level1"], errors="ignore")


def build(all_cells: Path | None = None):
    clr_parts = []
    count_parts = []
    percentage_parts = []
    lineage_of = {}
    meta_rows = []

    for ln, obs in _lineage_groups(all_cells):
        obs[CT_COL] = obs[CT_COL].map(_clean)
        obs = obs.dropna(subset=["sample_id", CT_COL])

        counts = pd.crosstab(obs["sample_id"], obs[CT_COL])
        clr = compute_clr(counts)
        clr.columns = [_clean(c) for c in clr.columns]
        percentage = counts.div(counts.sum(axis=1), axis=0) * 100
        percentage.columns = [_clean(c) for c in percentage.columns]
        for c in clr.columns:
            lineage_of[c] = ln
        clr_parts.append(clr)
        count_parts.append(counts)
        percentage_parts.append(percentage)

        # per-sample metadata (mode over that lineage's cells)
        m = obs.groupby("sample_id", observed=True)[META_COLS[1:]].agg(mode_or_nan)
        m["_lineage"] = ln
        meta_rows.append(m.reset_index())
        print(f"    {counts.shape[0]} samples x {counts.shape[1]} cell types")

    clr_wide = pd.concat(clr_parts, axis=1, sort=False)
    clr_wide = clr_wide.loc[:, ~clr_wide.columns.duplicated()]
    counts_wide = pd.concat(count_parts, axis=1, sort=False)
    counts_wide = counts_wide.loc[:, ~counts_wide.columns.duplicated()]
    percentage_wide = pd.concat(percentage_parts, axis=1, sort=False)
    percentage_wide = percentage_wide.loc[:, ~percentage_wide.columns.duplicated()]

    # one metadata row per sample (mode across lineages)
    meta = pd.concat(meta_rows, ignore_index=True)
    meta_agg = meta.groupby("sample_id", observed=True)[META_COLS[1:]].agg(mode_or_nan).reset_index()
    meta_agg["age_order"] = meta_agg["age_range"].map(AGE_ORDER)

    ct_map = pd.DataFrame(
        {"celltype": list(lineage_of.keys())}
    )
    ct_map["lineage"] = ct_map["celltype"].map(lineage_of)
    ct_map["is_follicle_tlo"] = ct_map["celltype"].isin(FOLLICLE_TLO)
    ct_map = ct_map.sort_values(["lineage", "celltype"]).reset_index(drop=True)
    ct_map.to_csv(OUT / "celltype_lineage_map.csv", index=False)

    # long CLR + metadata for splines
    long = (
        clr_wide.reset_index()
        .melt(id_vars="sample_id", var_name="celltype", value_name="clr")
        .dropna(subset=["clr"])
    )
    long["lineage"] = long["celltype"].map(lineage_of)
    percentage_long = (
        percentage_wide.reset_index()
        .melt(id_vars="sample_id", var_name="celltype",
              value_name="within_lineage_percentage")
        .dropna(subset=["within_lineage_percentage"])
    )
    long = long.merge(
        percentage_long, on=["sample_id", "celltype"], how="left"
    )
    count_long = (
        counts_wide.reset_index()
        .melt(id_vars="sample_id", var_name="celltype", value_name="n_cells")
        .dropna(subset=["n_cells"])
    )
    long = long.merge(count_long, on=["sample_id", "celltype"], how="left")
    long = long.merge(meta_agg, on="sample_id", how="left")
    long.to_csv(OUT / "clr_long.csv", index=False)
    print(f"clr_long.csv: {long.shape[0]:,} rows")

    return clr_wide, counts_wide, meta_agg, lineage_of, ct_map, long


def build_covariate_audit_tables(
    long: pd.DataFrame, meta_agg: pd.DataFrame
) -> None:
    """Write segment-stratified directional tables for every Figure 3 covariate."""
    audit_dir = OUT / "covariate_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    data = long.copy()
    data["segment"] = data["tissue_level_1"].map(
        lambda x: clean_category(x).lower() if pd.notna(clean_category(x)) else np.nan
    )
    sample_meta = meta_agg.copy()
    sample_meta["segment"] = sample_meta["tissue_level_1"].map(
        lambda x: clean_category(x).lower() if pd.notna(clean_category(x)) else np.nan
    )

    design_rows = []
    summary_frames = []
    omnibus_rows = []
    pairwise_rows = []

    for segment in CANONICAL_SEGMENTS:
        seg_data = data.loc[data["segment"] == segment].copy()
        seg_meta = sample_meta.loc[sample_meta["segment"] == segment].copy()
        if seg_data.empty:
            continue

        for covariate in AUDIT_COVARIATES:
            if covariate not in seg_data or covariate not in seg_meta:
                continue
            seg_data["_category"] = seg_data[covariate].map(clean_category)
            seg_meta["_category"] = seg_meta[covariate].map(clean_category)

            # Design and identifiability: retain unknown as a descriptive category,
            # but exclude it from inference.
            descriptive_meta = seg_meta.copy()
            descriptive_meta["_category_display"] = descriptive_meta["_category"].fillna("unknown")
            valid_meta = seg_meta.dropna(subset=["_category"])
            dataset_category_counts = (
                valid_meta.groupby(["dataset_id", "_category"], observed=True)
                .size()
                .rename("n")
                .reset_index()
            )
            datasets_with_two_levels = 0
            datasets_with_two_powered_levels = 0
            if not dataset_category_counts.empty:
                datasets_with_two_levels = int(
                    (dataset_category_counts.groupby("dataset_id")["_category"].nunique() >= 2).sum()
                )
                powered = dataset_category_counts.loc[
                    dataset_category_counts["n"] >= MIN_SAMPLES_PER_GROUP
                ]
                datasets_with_two_powered_levels = int(
                    (powered.groupby("dataset_id")["_category"].nunique() >= 2).sum()
                )

            for category, group in descriptive_meta.groupby(
                "_category_display", observed=True, dropna=False
            ):
                design_rows.append({
                    "segment": segment,
                    "covariate": covariate,
                    "category": category,
                    "n_samples": int(group["sample_id"].nunique()),
                    "n_donors": int(group["donor_id"].nunique()),
                    "n_datasets": int(group["dataset_id"].nunique()),
                    "n_categories_excluding_unknown": int(valid_meta["_category"].nunique()),
                    "n_datasets_with_multiple_categories": datasets_with_two_levels,
                    "n_datasets_with_two_categories_each_n_ge_5":
                        datasets_with_two_powered_levels,
                    "within_study_contrast_supported":
                        datasets_with_two_powered_levels > 0,
                })

            # Directional descriptive table: one row per segment, covariate
            # category and cell type.
            descriptive_data = seg_data.copy()
            descriptive_data["_category_display"] = descriptive_data["_category"].fillna("unknown")
            summary = (
                descriptive_data.groupby(
                    ["_category_display", "lineage", "celltype"],
                    observed=True, dropna=False
                )
                .agg(
                    n_samples=("sample_id", "nunique"),
                    n_donors=("donor_id", "nunique"),
                    n_datasets=("dataset_id", "nunique"),
                    total_cells=("n_cells", "sum"),
                    n_samples_present=("n_cells", lambda x: int((x > 0).sum())),
                    mean_cells_per_sample=("n_cells", "mean"),
                    median_cells_per_sample=("n_cells", "median"),
                    mean_clr=("clr", "mean"),
                    median_clr=("clr", "median"),
                    q1_clr=("clr", lambda x: x.quantile(0.25)),
                    q3_clr=("clr", lambda x: x.quantile(0.75)),
                    mean_within_lineage_pct=("within_lineage_percentage", "mean"),
                    median_within_lineage_pct=("within_lineage_percentage", "median"),
                    q1_within_lineage_pct=(
                        "within_lineage_percentage", lambda x: x.quantile(0.25)
                    ),
                    q3_within_lineage_pct=(
                        "within_lineage_percentage", lambda x: x.quantile(0.75)
                    ),
                )
                .reset_index()
                .rename(columns={"_category_display": "category"})
            )
            summary["present_sample_fraction"] = (
                summary["n_samples_present"] / summary["n_samples"]
            )
            summary.insert(0, "covariate", covariate)
            summary.insert(0, "segment", segment)
            summary_frames.append(summary)

            # Omnibus and pairwise tests use non-missing categories only.
            infer = seg_data.dropna(subset=["_category"]).copy()
            for (lineage, celltype), ct_data in infer.groupby(
                ["lineage", "celltype"], observed=True
            ):
                group_sizes = ct_data.groupby("_category", observed=True).size()
                eligible = group_sizes[group_sizes >= MIN_SAMPLES_PER_GROUP].index.tolist()
                descriptive = group_sizes[group_sizes >= 2].index.tolist()
                test_categories = eligible if len(eligible) >= 2 else descriptive
                if len(test_categories) < 2:
                    continue
                tested = ct_data.loc[ct_data["_category"].isin(test_categories)].copy()
                groups = [
                    g["clr"].dropna().values
                    for _, g in tested.groupby("_category", observed=True)
                ]
                inferential = len(eligible) >= 2
                p_value = np.nan
                if inferential:
                    try:
                        _, p_value = kruskal(*groups)
                    except ValueError:
                        p_value = np.nan
                medians = tested.groupby("_category", observed=True)["within_lineage_percentage"].median()
                row = {
                    "segment": segment,
                    "covariate": covariate,
                    "lineage": lineage,
                    "celltype": celltype,
                    "analysis_type": "inferential" if inferential else "descriptive",
                    "n_categories_tested": len(test_categories),
                    "categories_tested": " | ".join(map(str, test_categories)),
                    "n_samples": int(tested["sample_id"].nunique()),
                    "n_donors": int(tested["donor_id"].nunique()),
                    "n_datasets": int(tested["dataset_id"].nunique()),
                    "omega_squared": one_way_omega_squared(
                        tested["clr"], tested["_category"]
                    ),
                    "kruskal_p_value": p_value,
                    "highest_median_pct_category": str(medians.idxmax()),
                    "highest_median_within_lineage_pct": float(medians.max()),
                    "lowest_median_pct_category": str(medians.idxmin()),
                    "lowest_median_within_lineage_pct": float(medians.min()),
                    "n_datasets_with_multiple_categories": datasets_with_two_levels,
                    "n_datasets_with_two_categories_each_n_ge_5":
                        datasets_with_two_powered_levels,
                    "within_study_contrast_supported":
                        datasets_with_two_powered_levels > 0,
                    "spearman_age_rho": np.nan,
                    "spearman_age_p_value": np.nan,
                }
                if covariate == "age_range":
                    age = tested["age_range"].map(AGE_ORDER)
                    mask = age.notna() & tested["clr"].notna()
                    if mask.sum() >= 5 and age.loc[mask].nunique() >= 2:
                        rho, p_age = spearmanr(age.loc[mask], tested.loc[mask, "clr"])
                        row["spearman_age_rho"] = float(rho)
                        row["spearman_age_p_value"] = float(p_age)
                omnibus_rows.append(row)

                ordered_categories = list(test_categories)
                if covariate == "age_range":
                    ordered_categories = sorted(
                        test_categories, key=lambda x: AGE_ORDER.get(str(x), np.inf)
                    )
                else:
                    ordered_categories = sorted(map(str, test_categories))
                for category_a, category_b in combinations(ordered_categories, 2):
                    a = tested.loc[tested["_category"].astype(str) == str(category_a)]
                    b = tested.loc[tested["_category"].astype(str) == str(category_b)]
                    if len(a) < 2 or len(b) < 2:
                        continue
                    pair_inferential = (
                        len(a) >= MIN_SAMPLES_PER_GROUP
                        and len(b) >= MIN_SAMPLES_PER_GROUP
                    )
                    pair_p = np.nan
                    if pair_inferential:
                        try:
                            _, pair_p = mannwhitneyu(
                                a["clr"], b["clr"], alternative="two-sided"
                            )
                        except ValueError:
                            pair_p = np.nan
                    pairwise_rows.append({
                        "segment": segment,
                        "covariate": covariate,
                        "category_A": category_a,
                        "category_B": category_b,
                        "contrast": f"{category_b} minus {category_a}",
                        "lineage": lineage,
                        "celltype": celltype,
                        "analysis_type":
                            "inferential" if pair_inferential else "descriptive",
                        "n_samples_A": int(a["sample_id"].nunique()),
                        "n_samples_B": int(b["sample_id"].nunique()),
                        "n_donors_A": int(a["donor_id"].nunique()),
                        "n_donors_B": int(b["donor_id"].nunique()),
                        "n_datasets_A": int(a["dataset_id"].nunique()),
                        "n_datasets_B": int(b["dataset_id"].nunique()),
                        "mean_clr_A": float(a["clr"].mean()),
                        "mean_clr_B": float(b["clr"].mean()),
                        "delta_clr_B_minus_A": float(b["clr"].mean() - a["clr"].mean()),
                        "median_within_lineage_pct_A":
                            float(a["within_lineage_percentage"].median()),
                        "median_within_lineage_pct_B":
                            float(b["within_lineage_percentage"].median()),
                        "delta_median_pct_B_minus_A": float(
                            b["within_lineage_percentage"].median()
                            - a["within_lineage_percentage"].median()
                        ),
                        "mannwhitney_p_value": pair_p,
                        "n_datasets_with_multiple_categories":
                            datasets_with_two_levels,
                        "n_datasets_with_two_categories_each_n_ge_5":
                            datasets_with_two_powered_levels,
                        "within_study_contrast_supported":
                            datasets_with_two_powered_levels > 0,
                    })

    design = pd.DataFrame(design_rows)
    summary = pd.concat(summary_frames, ignore_index=True)
    omnibus = pd.DataFrame(omnibus_rows)
    pairwise = pd.DataFrame(pairwise_rows)

    omnibus = add_group_fdr(
        omnibus, "kruskal_p_value", ["segment", "covariate"],
        "kruskal_fdr_within_segment_covariate"
    )
    omnibus = add_group_fdr(
        omnibus, "spearman_age_p_value", ["segment", "covariate"],
        "spearman_age_fdr_within_segment_covariate"
    )
    pairwise = add_group_fdr(
        pairwise, "mannwhitney_p_value",
        ["segment", "covariate", "category_A", "category_B"],
        "mannwhitney_fdr_within_segment_covariate_contrast"
    )

    design.to_csv(audit_dir / "covariate_segment_design.csv", index=False)
    summary.to_csv(audit_dir / "covariate_segment_composition_summary.csv", index=False)
    omnibus.to_csv(audit_dir / "covariate_segment_omnibus_effects.csv", index=False)
    pairwise.to_csv(audit_dir / "covariate_segment_pairwise_contrasts.csv", index=False)
    print(
        "covariate audit:",
        f"{len(design):,} design rows,",
        f"{len(summary):,} summary rows,",
        f"{len(omnibus):,} omnibus rows,",
        f"{len(pairwise):,} pairwise rows",
    )


def wilcoxon_contrast(clr_wide, meta_agg, lineage_of, ct_map,
                      group_series: pd.Series, label_a: str, label_b: str):
    """group_series indexed by sample_id with values in {A,B,other/nan}."""
    data = clr_wide.join(group_series.rename("_g"), how="inner")
    rows = []
    for ct in clr_wide.columns:
        y = pd.to_numeric(data[ct], errors="coerce")
        ya = y[data["_g"] == label_a].dropna()
        yb = y[data["_g"] == label_b].dropna()
        na, nb = len(ya), len(yb)
        if na < MIN_SAMPLES_PER_GROUP or nb < MIN_SAMPLES_PER_GROUP:
            continue
        try:
            _, p = mannwhitneyu(ya.values, yb.values, alternative="two-sided")
        except ValueError:
            continue
        rows.append({
            "celltype": ct,
            "lineage": lineage_of.get(ct, "unknown"),
            "is_follicle_tlo": ct in FOLLICLE_TLO,
            "n_A": na, "n_B": nb,
            "mean_CLR_A": float(np.nanmean(ya)),
            "mean_CLR_B": float(np.nanmean(yb)),
            "delta_CLR_B_minus_A": float(np.nanmean(yb)) - float(np.nanmean(ya)),
            "p_value": float(p),
        })
    if not rows:
        return pd.DataFrame()
    res = pd.DataFrame(rows)
    _, res["p_adj"], _, _ = multipletests(res["p_value"], method="fdr_bh", alpha=FDR_ALPHA)
    res["neglog10_p_adj"] = -np.log10(np.clip(res["p_adj"], 1e-300, None))
    return res.sort_values("p_value").reset_index(drop=True)


def descriptive_contrast(clr_wide, lineage_of, group_series,
                         label_a: str, label_b: str, min_samples: int = 2):
    """Effect-size-only contrast for strata too small for inference."""
    data = clr_wide.join(group_series.rename("_g"), how="inner")
    rows = []
    for ct in clr_wide.columns:
        y = pd.to_numeric(data[ct], errors="coerce")
        ya = y[data["_g"] == label_a].dropna()
        yb = y[data["_g"] == label_b].dropna()
        na, nb = len(ya), len(yb)
        if na < min_samples or nb < min_samples:
            continue
        rows.append({
            "celltype": ct,
            "lineage": lineage_of.get(ct, "unknown"),
            "is_follicle_tlo": ct in FOLLICLE_TLO,
            "n_A": na, "n_B": nb,
            "mean_CLR_A": float(np.nanmean(ya)),
            "mean_CLR_B": float(np.nanmean(yb)),
            "delta_CLR_B_minus_A": float(np.nanmean(yb)) - float(np.nanmean(ya)),
            "p_value": np.nan,
            "p_adj": np.nan,
            "neglog10_p_adj": np.nan,
            "analysis_type": "descriptive",
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        "delta_CLR_B_minus_A", ascending=False
    ).reset_index(drop=True)


def norm(s):
    return s.astype(str).str.strip().str.lower()


def _n_sig(res: pd.DataFrame) -> int:
    if res.empty or "p_adj" not in res:
        return 0
    return int((res["p_adj"] < 0.05).sum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-cells",
        type=Path,
        default=None,
        help="Single all-cells h5ad (demo or full). Splits CLR by hgca_celltype_level1.",
    )
    parser.add_argument("--outdir", type=Path, default=OUT)
    return parser.parse_args()


def main():
    global OUT
    args = parse_args()
    all_cells = args.all_cells
    if all_cells is None:
        env = os.environ.get("HGCA_H5AD")
        if env:
            all_cells = Path(env)
        elif not LINEAGES:
            all_cells = DEMO_H5AD
    outdir = args.outdir
    default_out = Path(__file__).resolve().parent.parent / "data"
    if all_cells is not None and "demo" in all_cells.name and outdir == default_out:
        outdir = EXPECTED_CLR
        print(f"Demo input: writing to {outdir}")
    OUT = outdir
    OUT.mkdir(parents=True, exist_ok=True)
    if all_cells is not None and "demo" in all_cells.name:
        print("DEMO MODE: results are for software checking, not manuscript figures.")

    clr_wide, counts_wide, meta_agg, lineage_of, ct_map, long = build(all_cells)
    g = meta_agg.set_index("sample_id")
    build_covariate_audit_tables(long, meta_agg)

    # ---- contrast 1: collection method ----
    coll = norm(g["sample_collection_method"]).map(
        {"biopsy": "biopsy", "surgical resection": "resection"}
    )
    res = wilcoxon_contrast(clr_wide, meta_agg, lineage_of, ct_map, coll, "biopsy", "resection")
    res.to_csv(OUT / "clr_wilcoxon_collection.csv", index=False)
    print(f"collection: {len(res)} cell types, {_n_sig(res)} FDR<0.05")

    # ---- contrast 2: radial EPI vs LP (pure layers) ----
    rad = norm(g["radial_tissue_term"])
    rad_epi_lp = rad.where(rad.isin(["epi", "lp"])).map({"epi": "EPI", "lp": "LP"})
    res_r = wilcoxon_contrast(clr_wide, meta_agg, lineage_of, ct_map, rad_epi_lp, "EPI", "LP")
    res_r.to_csv(OUT / "clr_wilcoxon_radial_epi_lp.csv", index=False)
    print(f"radial EPI vs LP: {len(res_r)} cell types, {_n_sig(res_r)} FDR<0.05")

    # ---- contrast 3: full thickness (EPI_LP_MUSC) vs all other radial ----
    ft = rad.map(lambda r: "full_thickness" if r == "epi_lp_musc"
                 else ("rest" if r in {"epi", "epi_lp", "lp", "wm"} else np.nan))
    res_ft = wilcoxon_contrast(clr_wide, meta_agg, lineage_of, ct_map, ft, "rest", "full_thickness")
    res_ft.to_csv(OUT / "clr_wilcoxon_full_thickness.csv", index=False)
    print(f"full-thickness vs rest: {len(res_ft)} cell types, {_n_sig(res_ft)} FDR<0.05")

    # ---- full-thickness within each canonical gut segment ----
    # Colon and ileum meet the pre-specified >=5 samples/arm threshold and are
    # inferential. Duodenum and jejunum are effect-size-only because several
    # lineage strata contain just 2-4 samples per arm.
    for t in ("duodenum", "jejunum", "ileum", "colon"):
        samp_t = meta_agg.loc[norm(meta_agg["tissue_level_1"]) == t, "sample_id"]
        sub = clr_wide.loc[clr_wide.index.isin(samp_t)]
        d = OUT / "by_tissue" / t
        d.mkdir(parents=True, exist_ok=True)
        ft_t = ft.loc[ft.index.isin(samp_t)]
        if t in {"colon", "ileum"}:
            r2 = wilcoxon_contrast(
                sub, meta_agg, lineage_of, ct_map,
                ft_t, "rest", "full_thickness"
            )
            if not r2.empty:
                r2["analysis_type"] = "inferential"
        else:
            r2 = descriptive_contrast(
                sub, lineage_of, ft_t,
                "rest", "full_thickness", min_samples=2
            )
        if not r2.empty:
            r2.to_csv(d / f"clr_wilcoxon_full_thickness_{t}.csv", index=False)
        print(f"  [{t}] full_thickness={len(r2)} "
              f"({'inferential' if t in {'colon', 'ileum'} else 'descriptive'})")

    print("done.")


if __name__ == "__main__":
    main()
