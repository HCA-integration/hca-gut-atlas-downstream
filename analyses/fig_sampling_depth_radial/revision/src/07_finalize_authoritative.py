"""Merge all estimators into covariate_variance_authoritative.csv + supp full table."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"


def main():
    comp = pd.read_csv(TABLES / "composition_celltype_estimates.csv")
    expr = pd.read_csv(TABLES / "expression_pcr_celltype.csv")
    lmg = pd.read_csv(TABLES / "composition_lmg_celltype.csv")
    mixed = (
        pd.read_csv(TABLES / "composition_mixed_varfrac.csv")
        if (TABLES / "composition_mixed_varfrac.csv").exists()
        else pd.DataFrame()
    )
    mixed_boot = (
        pd.read_csv(TABLES / "composition_mixed_study_bootstrap.csv")
        if (TABLES / "composition_mixed_study_bootstrap.csv").exists()
        else pd.DataFrame()
    )
    expr_adj = (
        pd.read_csv(TABLES / "expression_adjusted_pcr.csv")
        if (TABLES / "expression_adjusted_pcr.csv").exists()
        else pd.DataFrame()
    )
    expr_boot = (
        pd.read_csv(TABLES / "expression_mixed_study_bootstrap.csv")
        if (TABLES / "expression_mixed_study_bootstrap.csv").exists()
        else pd.DataFrame()
    )

    rows = []
    for _, r in comp.iterrows():
        blo = bhi = np.nan
        if len(mixed_boot):
            hit = mixed_boot[
                (mixed_boot.lineage == r.lineage) & (mixed_boot.covariate == r.covariate)
            ]
            if len(hit):
                blo, bhi = float(hit.boot_lo.iloc[0]), float(hit.boot_hi.iloc[0])
        mix = np.nan
        if len(mixed):
            h = mixed[(mixed.celltype == r.celltype) & (mixed.covariate == r.covariate)]
            if len(h):
                mix = float(h.fixed_frac.iloc[0])
        rows.append(
            dict(
                celltype=r.celltype,
                lineage=r.lineage,
                covariate=r.covariate,
                estimator="oneway_omega2",
                n=r.n,
                point_estimate=r.omega2_trunc,
                point_estimate_raw=r.omega2_raw,
                partial_r2=r.partial_r2,
                mixed_fixed_frac=mix,
                bootstrap_lo=blo,
                bootstrap_hi=bhi,
                null_mean=r.null_mean,
                null_p50=r.null_p50,
                null_p95=r.null_p95,
                null_p99=r.null_p99,
                empirical_p=r.empirical_p,
                fdr_q=r.fdr_q,
                null_z=r.null_z,
                identifiable_within_study=r.identifiable_within_study,
                n_datasets_with_support=r.n_datasets_with_support,
                within_study_estimate=r.omega2_within_study,
                modality="composition",
            )
        )
    for _, r in expr.iterrows():
        blo = bhi = np.nan
        if len(expr_boot):
            hit = expr_boot[
                (expr_boot.lineage == r.lineage) & (expr_boot.covariate == r.covariate)
            ]
            if len(hit):
                blo, bhi = float(hit.boot_lo.iloc[0]), float(hit.boot_hi.iloc[0])
        adj = np.nan
        if len(expr_adj):
            h = expr_adj[(expr_adj.celltype == r.celltype) & (expr_adj.covariate == r.covariate)]
            if len(h):
                adj = float(h.omega2_study_donor_adjusted.iloc[0])
        rows.append(
            dict(
                celltype=r.celltype,
                lineage=r.lineage,
                covariate=r.covariate,
                estimator="pcr_omega2",
                n=r.n,
                point_estimate=r.omega2_trunc,
                point_estimate_raw=r.omega2_raw,
                partial_r2=np.nan,
                mixed_fixed_frac=adj,
                bootstrap_lo=blo,
                bootstrap_hi=bhi,
                null_mean=r.null_mean,
                null_p50=r.null_p50,
                null_p95=r.null_p95,
                null_p99=r.null_p99,
                empirical_p=r.empirical_p,
                fdr_q=r.fdr_q,
                null_z=r.null_z,
                identifiable_within_study=r.identifiable_within_study,
                n_datasets_with_support=r.n_datasets_with_support,
                within_study_estimate=r.omega2_within_study,
                modality="expression",
            )
        )
    if len(lmg):
        for _, r in lmg.iterrows():
            rows.append(
                dict(
                    celltype=r.celltype,
                    lineage=r.lineage,
                    covariate=r.covariate,
                    estimator="lmg_r2",
                    n=r.n,
                    point_estimate=r.point,
                    point_estimate_raw=r.point,
                    partial_r2=np.nan,
                    mixed_fixed_frac=np.nan,
                    bootstrap_lo=np.nan,
                    bootstrap_hi=np.nan,
                    null_mean=np.nan,
                    null_p50=np.nan,
                    null_p95=np.nan,
                    null_p99=np.nan,
                    empirical_p=np.nan,
                    fdr_q=np.nan,
                    null_z=np.nan,
                    identifiable_within_study=np.nan,
                    n_datasets_with_support=np.nan,
                    within_study_estimate=np.nan,
                    modality="composition",
                )
            )

    auth = pd.DataFrame(rows)
    auth.to_csv(TABLES / "covariate_variance_authoritative.csv", index=False)

    # Supplementary: all 94 cell types × covariates with FDR (composition)
    supp = comp[
        [
            "celltype",
            "lineage",
            "covariate",
            "n",
            "partial_r2",
            "omega2_raw",
            "omega2_trunc",
            "null_mean",
            "null_p95",
            "null_z",
            "empirical_p",
            "fdr_q",
            "identifiable_within_study",
            "n_datasets_with_support",
            "omega2_within_study",
        ]
    ].copy()
    supp.to_csv(TABLES / "supp_all_celltypes_composition_omega2_fdr.csv", index=False)

    # Expression full table
    expr[
        [
            "celltype",
            "lineage",
            "covariate",
            "n",
            "omega2_raw",
            "omega2_trunc",
            "null_mean",
            "null_p95",
            "null_z",
            "empirical_p",
            "fdr_q",
            "identifiable_within_study",
            "n_datasets_with_support",
            "omega2_within_study",
        ]
    ].to_csv(TABLES / "supp_all_celltypes_expression_pcr_fdr.csv", index=False)

    print(f"authoritative rows={len(auth)}")
    print(f"supp composition rows={len(supp)}")


if __name__ == "__main__":
    main()
