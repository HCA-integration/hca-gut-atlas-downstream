"""Cache per-cell-type pseudobulk PCA embeddings for expression PCR revision.

Reuses the same thresholds / HVG / PCA logic as
`../../src/depth_expression_de.py` and
`hca-gut-atlas-downstream/scripts/composition_vs_expression_pcr.py`.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT.parent
CACHE = ROOT / "cache"

_OBJECTS = Path(os.environ["HGCA_OBJECTS"]) if os.environ.get("HGCA_OBJECTS") else None
LINEAGE_PATHS = (
    {
        "epithelial": str(_OBJECTS / "epithelial.h5ad"),
        "lymphoid": str(_OBJECTS / "lymphoid.h5ad"),
        "myeloid": str(_OBJECTS / "myeloid.h5ad"),
        "stroma": str(_OBJECTS / "stroma.h5ad"),
    }
    if _OBJECTS is not None
    else {}
)
SAMPLE_KEY = "sample_id"
CT_COL = "hgca_celltype_v1"
MIN_CELLS = 10
MIN_SAMPLES = 25
N_HVG = 2000
N_PCS = 50
META_COLS = [
    "donor_id", "dataset_id", "tissue_level_1", "sampled_site_condition",
    "radial_tissue_term", "sample_preservation_method", "sex_ontology_term",
    "age_range", "assay", "sample_collection_method", "sequenced_fragment",
    "gene_annotation_version",
]


def _mode(x):
    m = x.mode()
    return m.iloc[0] if len(m) else np.nan


def embed(X, n_pcs):
    X = np.asarray(X, dtype=float)
    sd = X.std(axis=0)
    X = X[:, sd > 1e-12]
    if X.shape[1] == 0:
        return None, None
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    n_comp = int(min(n_pcs, X.shape[0] - 1, X.shape[1]))
    if n_comp < 1:
        return None, None
    p = PCA(n_components=n_comp, svd_solver="full")
    return p.fit_transform(X), p.explained_variance_


def main():
    if not LINEAGE_PATHS:
        raise SystemExit(
            "Set HGCA_OBJECTS to the directory with epithelial/lymphoid/myeloid/stroma.h5ad."
        )
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    for lineage, path in LINEAGE_PATHS.items():
        print(f"=== {lineage} ===", flush=True)
        adata = sc.read_h5ad(path)
        adata.obs[SAMPLE_KEY] = adata.obs[SAMPLE_KEY].astype(str)
        cols = [c for c in META_COLS if c in adata.obs.columns]
        meta = adata.obs.groupby(SAMPLE_KEY).agg({c: _mode for c in cols})
        meta.index = meta.index.astype(str)

        cts = adata.obs[CT_COL].astype(str).value_counts()
        for ct, _ in cts.items():
            mask = (adata.obs[CT_COL].astype(str) == ct).values
            if mask.sum() == 0:
                continue
            X = adata.X[mask]
            if not sparse.issparse(X):
                X = sparse.csr_matrix(X)
            samples = adata.obs.loc[mask, SAMPLE_KEY].astype(str).values
            uniq, inv = np.unique(samples, return_inverse=True)
            ind = sparse.csr_matrix(
                (np.ones(len(inv)), (inv, np.arange(len(inv)))),
                shape=(len(uniq), X.shape[0]),
            )
            pb = np.asarray((ind @ X).todense(), dtype=float)
            cps = np.bincount(inv, minlength=len(uniq))
            keep = cps >= MIN_CELLS
            pb, uniq = pb[keep], uniq[keep]
            if pb.shape[0] < MIN_SAMPLES:
                continue
            lib = pb.sum(axis=1, keepdims=True)
            lib[lib == 0] = 1.0
            logcpm = np.log1p(pb / lib * 1e6)
            var = logcpm.var(axis=0)
            top = np.argsort(var)[::-1][:N_HVG]
            scores, ev = embed(logcpm[:, top], N_PCS)
            if scores is None:
                continue
            stem = (
                ct.replace("/", "_").replace(" ", "_").replace("(", "")
                .replace(")", "").replace(",", "")
            )
            out = CACHE / f"expr_{lineage}_{stem}.npz"
            np.savez_compressed(
                out,
                scores=scores.astype(np.float32),
                var_weights=ev.astype(np.float32),
                samples=uniq.astype(str),
            )
            msub = meta.reindex(uniq)
            msub.to_parquet(CACHE / f"expr_{lineage}_{stem}_meta.parquet")
            rows.append(
                dict(
                    lineage=lineage,
                    celltype=ct,
                    n_samples=int(len(uniq)),
                    n_pcs=int(scores.shape[1]),
                    path=str(out.name),
                )
            )
            print(f"  {ct}: n={len(uniq)} pcs={scores.shape[1]}", flush=True)
        del adata

    idx = pd.DataFrame(rows)
    idx.to_csv(CACHE / "expression_embedding_index.csv", index=False)
    with open(CACHE / "expression_embedding_index.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {len(rows)} embeddings to {CACHE}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
