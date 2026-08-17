#!/usr/bin/env python3
"""Stdlib checks that the bundled demo slice is the object the README describes."""
from __future__ import annotations

from pathlib import Path

import anndata as ad

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "data" / "demo" / "hgca_all_lineages_v1_demo.h5ad"
GC = {"GC B Light Zone (GC B LZ)", "GC B Dark Zone (GC B DZ)"}


def main() -> int:
    adata = ad.read_h5ad(DEMO)
    obs = adata.obs
    n_types = obs["hgca_celltype_v1"].astype(str).nunique()
    gcb = (
        obs.loc[obs["hgca_celltype_v1"].astype(str).isin(GC)]
        .groupby("sample_id", observed=True)
        .size()
    )
    n_k3 = int((gcb >= 3).sum())
    assert adata.n_obs == 3185, adata.n_obs
    assert n_types == 94, n_types
    assert n_k3 == 17, n_k3
    print(f"ok: cells={adata.n_obs} types={n_types} follicle_k3={n_k3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
