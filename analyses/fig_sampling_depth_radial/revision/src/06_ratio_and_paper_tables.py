"""Task 7 ratio CIs + paper-facing identifiability / Results-edit tables."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
DATA = ROOT.parent / "data"
SEED = 20260804
N_BOOT = 500

PRETTY = {
    "sampled_site_condition": "Sample condition",
    "radial_tissue_term": "Radial layer",
    "sample_preservation_method": "Preservation",
    "sex_ontology_term": "Sex",
    "age_range": "Age",
    "dataset_id": "Study / batch",
    "assay": "Assay",
    "sample_collection_method": "Biopsy vs resection",
    "sequenced_fragment": "Sequenced fragment",
    "gene_annotation_version": "Gene annotation",
    "tissue_level_1": "Gut segment",
}


def main():
    rng = np.random.default_rng(SEED)
    comp_pcr = pd.read_csv(TABLES / "composition_pcr_lineage.csv")
    boot_c = pd.read_csv(TABLES / "composition_pcr_study_bootstrap.csv")
    expr_lin = pd.read_csv(TABLES / "expression_pcr_lineage.csv")
    # expression study bootstrap: resample lineages' celltype rows by study is hard
    # without raw draws; approximate CI via lineage jackknife of ratio
    comp_ct = pd.read_csv(TABLES / "composition_celltype_estimates.csv")
    expr_ct = pd.read_csv(TABLES / "expression_pcr_celltype.csv")
    id_sum = pd.read_csv(TABLES / "identifiability_summary.csv")
    lmg = pd.read_csv(TABLES / "composition_lmg_celltype.csv")

    # ---- Task 7: anatomy/study ratio with bootstrap CIs ----
    # Recompute composition ratio CI from study-bootstrap table by pairing
    # tissue_level_1 and dataset_id draws via their percentile envelopes is
    # insufficient; use lineage-level PCR point estimates and bootstrap the
    # ratio of lineage-means by resampling lineages (composition) and, for
    # study-level, use the available study-bootstrap means per lineage.
    ratio_rows = []
    for modality, src, col in [
        ("composition", comp_pcr, "omega2_trunc"),
        ("expression", expr_lin, "pcr_weighted"),
    ]:
        wide = (
            src[src["covariate"].isin(["dataset_id", "tissue_level_1"])]
            .pivot_table(index="lineage", columns="covariate", values=col)
            .reset_index()
        )
        wide["ratio"] = wide["tissue_level_1"] / wide["dataset_id"]
        wide["modality"] = modality
        # bootstrap lineages
        ratios = []
        lin = wide["lineage"].to_numpy()
        for _ in range(N_BOOT):
            draw = rng.choice(len(lin), size=len(lin), replace=True)
            sub = wide.iloc[draw]
            ratios.append(sub["tissue_level_1"].mean() / sub["dataset_id"].mean())
        ratios = np.asarray(ratios, dtype=float)
        pooled = wide["tissue_level_1"].mean() / wide["dataset_id"].mean()
        ratio_rows.append(
            dict(
                modality=modality,
                pooled_ratio=pooled,
                boot_lo=float(np.percentile(ratios, 2.5)),
                boot_hi=float(np.percentile(ratios, 97.5)),
                lineage_min=float(wide["ratio"].min()),
                lineage_max=float(wide["ratio"].max()),
                epithelial=float(wide.loc[wide.lineage == "epithelial", "ratio"].iloc[0]),
                lymphoid=float(wide.loc[wide.lineage == "lymphoid", "ratio"].iloc[0]),
                myeloid=float(wide.loc[wide.lineage == "myeloid", "ratio"].iloc[0]),
                stroma=float(wide.loc[wide.lineage == "stroma", "ratio"].iloc[0]),
            )
        )
        wide.to_csv(TABLES / f"task7_ratio_lineage_{modality}.csv", index=False)

    # study-bootstrap based composition ratio (resample studies already done per
    # lineage for each covariate separately — use ratio of bootstrap means as
    # point and percentile of boot_mean_tissue/boot_mean_dataset across lineages
    # as a conservative CI using paired lineage boot means)
    bwide = boot_c.pivot_table(
        index="lineage", columns="covariate", values=["boot_mean", "boot_lo", "boot_hi"]
    )
    # flatten
    if ("boot_mean", "dataset_id") in bwide.columns and ("boot_mean", "tissue_level_1") in bwide.columns:
        r = (
            bwide[("boot_mean", "tissue_level_1")]
            / bwide[("boot_mean", "dataset_id")]
        )
        # CI via delta on lo/hi bounds (worst-case)
        r_lo = bwide[("boot_lo", "tissue_level_1")] / bwide[("boot_hi", "dataset_id")]
        r_hi = bwide[("boot_hi", "tissue_level_1")] / bwide[("boot_lo", "dataset_id")]
        ratio_rows.append(
            dict(
                modality="composition_studyboot_bounds",
                pooled_ratio=float(r.mean()),
                boot_lo=float(r_lo.mean()),
                boot_hi=float(r_hi.mean()),
                lineage_min=float(r.min()),
                lineage_max=float(r.max()),
                epithelial=float(r.get("epithelial", np.nan)),
                lymphoid=float(r.get("lymphoid", np.nan)),
                myeloid=float(r.get("myeloid", np.nan)),
                stroma=float(r.get("stroma", np.nan)),
            )
        )

    # Prefer mixed-model bootstrap if present
    for path, mod in [
        (TABLES / "composition_mixed_study_bootstrap.csv", "composition_mixed"),
        (TABLES / "expression_mixed_study_bootstrap.csv", "expression_mixed"),
    ]:
        if not path.exists():
            continue
        m = pd.read_csv(path)
        # dataset_id may be absent as fixed effect; skip if so
        if not set(["dataset_id", "tissue_level_1"]).issubset(set(m["covariate"])):
            # try using OLS PCR for dataset and mixed for tissue — skip
            continue
        wide = m.pivot_table(
            index="lineage", columns="covariate",
            values=["fixed_frac", "boot_lo", "boot_hi"]
        )
        if ("fixed_frac", "dataset_id") not in wide.columns:
            continue
        r = wide[("fixed_frac", "tissue_level_1")] / wide[("fixed_frac", "dataset_id")]
        r_lo = wide[("boot_lo", "tissue_level_1")] / wide[("boot_hi", "dataset_id")]
        r_hi = wide[("boot_hi", "tissue_level_1")] / wide[("boot_lo", "dataset_id")]
        ratio_rows.append(
            dict(
                modality=mod,
                pooled_ratio=float(r.mean()),
                boot_lo=float(r_lo.mean()),
                boot_hi=float(r_hi.mean()),
                lineage_min=float(r.min()),
                lineage_max=float(r.max()),
                epithelial=float(r.get("epithelial", np.nan)),
                lymphoid=float(r.get("lymphoid", np.nan)),
                myeloid=float(r.get("myeloid", np.nan)),
                stroma=float(r.get("stroma", np.nan)),
            )
        )

    ratio_df = pd.DataFrame(ratio_rows)
    ratio_df.to_csv(TABLES / "task7_anatomy_study_ratio_ci.csv", index=False)
    print(ratio_df.to_string(index=False))

    # Does lineage ordering survive CIs?
    # published composition ranges 0.21–0.59 — check overlap of lineage ratios
    for modality in ["composition", "expression"]:
        sub = ratio_df[ratio_df["modality"] == modality]
        if sub.empty:
            continue
        r = sub.iloc[0]
        # lineage CI overlap with pooled: if lineage_min and lineage_max span
        # and boot CI includes 1 or crosses substantially
        survives = r["boot_hi"] < 1.0  # all anatomy << study
        print(
            f"{modality}: pooled={r['pooled_ratio']:.3f} "
            f"CI=[{r['boot_lo']:.3f},{r['boot_hi']:.3f}] "
            f"lineage range=[{r['lineage_min']:.3f},{r['lineage_max']:.3f}] "
            f"anatomy<study_survives={survives}"
        )

    # ---- Paper identifiability table ----
    pooled = (
        comp_ct.groupby("covariate")["omega2_trunc"].mean().rename("pooled_omega2")
    )
    within = (
        comp_ct.groupby("covariate")["omega2_within_study"].mean().rename("within_omega2")
    )
    lmg_m = lmg.groupby("covariate")["point"].mean().rename("lmg_r2")
    expr_m = (
        expr_ct.groupby("covariate")["omega2_trunc"].mean().rename("expr_pcr_omega2")
    )
    paper = id_sum.merge(pooled, left_on="covariate", right_index=True, how="left")
    paper = paper.merge(within, left_on="covariate", right_index=True, how="left")
    paper = paper.merge(lmg_m, left_on="covariate", right_index=True, how="left")
    paper = paper.merge(expr_m, left_on="covariate", right_index=True, how="left")
    paper = paper.rename(
        columns={
            "identifiable_within_study": "identifiable_within_study",
            "n_datasets_with_support": "n_datasets_with_support",
            "pooled_omega2": "pooled_estimate_composition_omega2",
            "within_omega2": "within_study_estimate_composition",
        }
    )
    paper.to_csv(TABLES / "paper_identifiability_table.csv", index=False)

    # ---- Results sentences that must change ----
    # Key published numbers from heatmap (partial R2 labeled as omega2)
    keys = [
        ("Submucosal Fibroblasts (S3)", "radial_tissue_term", 0.562),
        ("Submucosal Fibroblasts (S3)", "sample_collection_method", 0.578),
        ("Post Arteriole Capillary Endothelial (PAC)", "radial_tissue_term", 0.45),
        ("Gamma Delta T Cells", "age_range", 0.36),
        ("Colonocyte Progenitors", "sample_preservation_method", 0.285),
    ]
    edits = []
    for ct, cov, old in keys:
        r = comp_ct[(comp_ct.celltype == ct) & (comp_ct.covariate == cov)].iloc[0]
        edits.append(
            dict(
                celltype=ct,
                covariate=cov,
                published_as_omega2_was_partial_r2=old,
                recomputed_partial_r2=float(r.partial_r2),
                recomputed_omega2=float(r.omega2_trunc),
                empirical_p=float(r.empirical_p),
                fdr_q=float(r.fdr_q),
                within_study_omega2=float(r.omega2_within_study)
                if pd.notna(r.omega2_within_study)
                else np.nan,
                n_datasets_with_support=int(r.n_datasets_with_support),
                note=(
                    "Published figure number is partial R2, not omega2; "
                    "report omega2 + permutation q; flag if not within-study identifiable"
                ),
            )
        )
    # published PCR ranges
    pub_long = pd.read_csv(DATA / "composition_vs_expression_pcr_long.csv")
    for modality in ["composition", "expression"]:
        sub = pub_long[pub_long.modality == modality]
        for cov in ["dataset_id", "radial_tissue_term", "tissue_level_1"]:
            old = float(sub.loc[sub.covariate == cov, "pcr"].mean())
            if modality == "composition":
                new = float(comp_pcr.loc[comp_pcr.covariate == cov, "omega2_trunc"].mean())
            else:
                new = float(expr_lin.loc[expr_lin.covariate == cov, "pcr_weighted"].mean())
            edits.append(
                dict(
                    celltype="(lineage-mean PCR)",
                    covariate=cov,
                    published_as_omega2_was_partial_r2=old,
                    recomputed_partial_r2=np.nan,
                    recomputed_omega2=new,
                    empirical_p=np.nan,
                    fdr_q=np.nan,
                    within_study_omega2=np.nan,
                    n_datasets_with_support=np.nan,
                    note=f"PCR {modality} lineage-mean",
                )
            )
    pd.DataFrame(edits).to_csv(TABLES / "results_sentences_to_edit.csv", index=False)

    # complete-case
    cc = pd.read_csv(TABLES / "task8_complete_case_ranking.csv")
    print("\nComplete-case rank stability:")
    print(cc[["covariate", "rank_primary", "rank_complete"]].to_string(index=False))

    # weighting alternative already in expression_pcr_lineage
    print("\nExpression weighted vs unweighted ranking:")
    w = (
        expr_lin.groupby("covariate")[["pcr_weighted", "pcr_unweighted"]]
        .mean()
        .sort_values("pcr_weighted", ascending=False)
    )
    print(w.to_string())
    print(
        "Spearman weighted vs unweighted:",
        w["pcr_weighted"].rank().corr(w["pcr_unweighted"].rank(), method="spearman"),
    )


if __name__ == "__main__":
    main()
