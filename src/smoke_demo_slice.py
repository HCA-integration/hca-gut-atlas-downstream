#!/usr/bin/env python3
"""Expected-output smoke test for the HGCA v1 demo slice."""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo" / "hgca_all_lineages_v1_demo.h5ad"
OUT = DEMO.parent / "expected"
GC = {"GC B Light Zone (GC B LZ)", "GC B Dark Zone (GC B DZ)"}


def main() -> None:
    print("DEMO MODE: results are for software checking, not manuscript figures.")
    adata = ad.read_h5ad(DEMO)
    obs = adata.obs.copy()
    obs["ct"] = obs["hgca_celltype_v1"].astype(str)
    obs["lin"] = obs["hgca_celltype_level1"].astype(str)
    obs["tissue"] = obs["tissue_level_1"].astype(str)
    obs["coll"] = obs["sample_collection_method"].astype(str)

    rows = []
    for lin, sub in obs.groupby("lin", observed=True):
        ct = pd.crosstab(sub["sample_id"], sub["ct"])
        x = np.log(ct.astype(float) + 0.5)
        x = x.sub(x.mean(axis=1), axis=0)
        long = x.stack().rename("clr").reset_index()
        long.columns = ["sample_id", "ct", "clr"]
        long["lin"] = lin
        rows.append(long)
    clr = pd.concat(rows, ignore_index=True)
    meta = obs.drop_duplicates("sample_id").set_index("sample_id")
    clr = clr.join(meta[["tissue", "coll"]], on="sample_id")

    tests = []
    ile = clr[clr["tissue"] == "ileum"]
    for typ, sub in ile.groupby("ct", observed=True):
        a = sub.loc[sub["coll"] == "biopsy", "clr"]
        b = sub.loc[sub["coll"] == "surgical resection", "clr"]
        if len(a) >= 5 and len(b) >= 5:
            tests.append((typ, float(mannwhitneyu(a, b, alternative="two-sided").pvalue)))
    gcb = (
        obs.loc[obs["ct"].isin(GC)]
        .groupby("sample_id", observed=True)
        .size()
        .rename("n_gcb")
    )
    OUT.mkdir(parents=True, exist_ok=True)
    clr.head(20).to_csv(OUT / "clr_long.head.csv", index=False)
    pd.DataFrame(tests, columns=["celltype", "mwu_p"]).to_csv(OUT / "ileum_collection_mwu.csv", index=False)
    gcb.reset_index().to_csv(OUT / "follicle_gcb_counts.csv", index=False)
    print(
        f"cells={adata.n_obs} types={obs['ct'].nunique()} samples={obs['sample_id'].nunique()} "
        f"mwu_tests={len(tests)} follicle_k3={int((gcb >= 3).sum())} "
        f"follicle_zero={obs['sample_id'].nunique() - gcb.shape[0]}"
    )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
