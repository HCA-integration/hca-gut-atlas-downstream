#!/usr/bin/env python
"""Build sample × cell-type × gene expression tables for the follicle L–R analysis.

Uses disease==normal cells from hgca_all_lineages_v1, joined to the predefined
follicle+/− sample classification. Expression is mean log1p(CP10k) and fraction
detected within each sample×cell-type, matching the LIANA preprocessing scale.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from config import (  # noqa: E402
    ATLAS,
    AUDIT_CSV,
    CACHE,
    CURATED,
    ECOSYSTEM,
    GROUP_KEY,
    LIANA_COMBINED,
    MIN_CT_CELLS,
    OUT,
    POWERED_SEGMENTS,
)


def subunits(complex_name: str) -> list[str]:
    return [p for p in str(complex_name).split("_") if p]


def genes_from_resource_and_curated() -> set[str]:
    genes: set[str] = set()
    for lig, rec in CURATED:
        genes.update(subunits(lig))
        genes.update(subunits(rec))
    # FAE / MHC-II / early-M module genes
    genes.update([
        "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1",
        "CD74", "CIITA", "SPIB", "SOX8", "GP2", "TNFAIP2", "TNFRSF11A",
        "MARCKSL1", "ANXA5", "ICAM2", "ACKR4", "ACKR2", "ACKR1",
        "NECTIN2", "NECTIN3", "TIGIT", "ALCAM", "CD6", "LGALS3", "LAG3",
        "CD24", "SIGLEC10", "CXCL13", "CXCR5", "CCL19", "CCL21", "CCR7",
        "TNFSF11", "TNFRSF11B", "TNF", "TNFRSF1A", "TNFRSF1B",
        "TNFSF13B", "TNFRSF13C", "TNFRSF17", "TNFSF13", "TNFRSF13B",
        "LTB", "LTA", "LTBR", "ICAM1", "VCAM1", "ITGAL", "ITGB2",
        "ITGA4", "ITGB1", "ITGB7", "CCL25", "CCR9", "C1QA", "CR1", "C3",
        "CR2", "GAS6", "PROS1", "MERTK", "FCER2", "CD40LG", "CD40",
    ])
    # Top LIANA ecosystem edges (magnitude consensus) to expand discovery set
    if LIANA_COMBINED.exists():
        usecols = [
            "tissue_level_1", "source", "target",
            "ligand_complex", "receptor_complex", "magnitude_rank",
        ]
        keep = []
        for chunk in pd.read_csv(LIANA_COMBINED, usecols=usecols, chunksize=500_000):
            m = chunk[
                chunk["source"].isin(ECOSYSTEM)
                & chunk["target"].isin(ECOSYSTEM)
                & (chunk["magnitude_rank"] <= 0.05)
                & chunk["tissue_level_1"].str.lower().isin(POWERED_SEGMENTS)
            ]
            if len(m):
                keep.append(m)
        if keep:
            top = pd.concat(keep, ignore_index=True)
            # unique LR pairs, keep best ranks
            top = (
                top.groupby(["ligand_complex", "receptor_complex"], as_index=False)[
                    "magnitude_rank"
                ]
                .min()
                .sort_values("magnitude_rank")
                .head(400)
            )
            top.to_csv(CACHE / "liana_priority_lr_pairs.csv", index=False)
            for _, r in top.iterrows():
                genes.update(subunits(r["ligand_complex"]))
                genes.update(subunits(r["receptor_complex"]))
    return genes


def load_audit() -> pd.DataFrame:
    a = pd.read_csv(AUDIT_CSV)
    a["segment"] = a["segment"].astype(str).str.lower()
    a["follicle_pos"] = (a["follicle_status_primary"] == "positive").astype(int)
    return a


def main() -> None:
    t0 = time.time()
    audit = load_audit()
    sample_ids = set(audit["sample_id"].astype(str))
    genes_wanted = sorted(genes_from_resource_and_curated())
    print(f"Wanted genes: {len(genes_wanted)}")
    print(f"Samples in audit: {len(sample_ids)}")

    print("Opening AnnData backed…")
    a = ad.read_h5ad(ATLAS, backed="r")
    obs = a.obs
    if "gene_symbol" in a.var.columns:
        symbols = a.var["gene_symbol"].astype(str).values
    else:
        symbols = a.var_names.astype(str)
    gene_to_ix = {g: i for i, g in enumerate(symbols)}
    gene_ix = [gene_to_ix[g] for g in genes_wanted if g in gene_to_ix]
    genes_found = [g for g in genes_wanted if g in gene_to_ix]
    missing = sorted(set(genes_wanted) - set(genes_found))
    print(f"Found {len(genes_found)} genes; missing {len(missing)}")
    if missing[:20]:
        print("  e.g. missing:", ", ".join(missing[:20]))
    pd.Series(missing, name="gene").to_csv(CACHE / "genes_missing.csv", index=False)

    # cell mask
    disease = obs["disease"].astype(str)
    mask = (
        disease.eq("normal")
        & obs["sample_id"].astype(str).isin(sample_ids)
        & obs[GROUP_KEY].astype(str).isin(ECOSYSTEM)
    )
    ix_all = np.where(np.asarray(mask))[0]
    print(f"Cells after filters: {len(ix_all):,}")

    # pull sparse matrix for selected cells × genes
    print("Materializing gene subset…")
    sub = a[ix_all, gene_ix].to_memory()
    if "gene_symbol" in sub.var.columns:
        sub.var_names = sub.var["gene_symbol"].astype(str).values
    # CP10k + log1p on this gene subset (approx; total counts from full would be
    # better — use obs n_counts if present, else row sums of full X for these cells)
    if "n_counts" in sub.obs.columns:
        lib = sub.obs["n_counts"].to_numpy(dtype=np.float64)
    elif "total_counts" in sub.obs.columns:
        lib = sub.obs["total_counts"].to_numpy(dtype=np.float64)
    else:
        # fallback: sum over all genes for these cells (expensive but accurate)
        print("Computing library sizes from full matrix…")
        lib = np.asarray(a[ix_all, :].X.sum(axis=1)).ravel().astype(np.float64)
    lib = np.maximum(lib, 1.0)
    X = sub.X
    if sparse.issparse(X):
        X = X.tocsr().astype(np.float64).copy()
        # scale rows then log1p; convert back to CSR after multiply (coo not row-indexable)
        X = X.multiply(1e4 / lib[:, None]).tocsr()
        X.data = np.log1p(X.data)
    else:
        X = np.asarray(X, dtype=np.float64)
        X = np.log1p(X * (1e4 / lib[:, None]))

    ct = sub.obs[GROUP_KEY].astype(str).to_numpy()
    sid = sub.obs["sample_id"].astype(str).to_numpy()
    genes = np.asarray(sub.var_names)

    print("Aggregating sample × cell-type × gene…")
    # group keys
    key = pd.DataFrame({"sample_id": sid, "celltype": ct})
    key["_row"] = np.arange(len(key))
    rows = []
    for (s, c), g in key.groupby(["sample_id", "celltype"], sort=False):
        rows_ix = g["_row"].to_numpy()
        n = len(rows_ix)
        if n < 1:
            continue
        if sparse.issparse(X):
            block = X[rows_ix].toarray()
        else:
            block = X[rows_ix]
        mean = block.mean(axis=0)
        frac = (block > 0).mean(axis=0)
        rows.append(
            pd.DataFrame(
                {
                    "sample_id": s,
                    "celltype": c,
                    "gene": genes,
                    "n_cells": n,
                    "mean_log1p": mean,
                    "frac_expr": frac,
                }
            )
        )
    expr = pd.concat(rows, ignore_index=True)
    # attach metadata
    meta = audit[
        [
            "sample_id",
            "donor_id",
            "dataset_id",
            "segment",
            "collection",
            "follicle_status_primary",
            "follicle_pos",
            "n_gc_b",
            "gc_abundance_frac",
            "borderline_gc",
        ]
    ].drop_duplicates("sample_id")
    expr = expr.merge(meta, on="sample_id", how="left")
    # usable flag
    expr["usable"] = expr["n_cells"] >= MIN_CT_CELLS

    out_parquet = CACHE / "sample_celltype_gene_expr.parquet"
    expr.to_parquet(out_parquet, index=False)
    # wide n_cells table
    ntab = (
        expr[["sample_id", "celltype", "n_cells", "usable"]]
        .drop_duplicates()
        .pivot_table(index="sample_id", columns="celltype", values="n_cells", fill_value=0)
    )
    ntab.to_csv(CACHE / "sample_celltype_n_cells.csv")
    print(f"Wrote {out_parquet}  ({len(expr):,} rows) in {(time.time()-t0)/60:.1f} min")
    print(f"Usable sample×CT (n≥{MIN_CT_CELLS}): {expr['usable'].sum():,}")


if __name__ == "__main__":
    main()
