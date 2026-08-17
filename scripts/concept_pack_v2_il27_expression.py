#!/usr/bin/env python
"""Quick expression check for IL27 complex subunits (why LIANA may drop them)."""
from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

AD_PATH = os.environ.get(
    "LIANA_AD_PATH",
    "/Users/kylekimler/Projects/GCA/meta_datasets/integrated-objects/hgca_all_lineages_v1.h5ad",
)
OUT = Path(os.environ.get(
    "CCC_OUTPUT_DIR",
    "/Users/kylekimler/Projects/GCA/github_vignette_output/LIANA_rank_aggregate",
)) / "concept_pack_v2"
OUT.mkdir(parents=True, exist_ok=True)

GENES = ["IL27", "EBI3", "IL12A", "IL27RA", "IL6ST", "IL12RB2"]
GROUP = "hgca_celltype_v1"
CD4_PAT = "CD4 "

print("Loading", AD_PATH)
a = ad.read_h5ad(AD_PATH, backed="r")
obs = a.obs
mask = obs["disease"].astype(str).eq("normal") if "disease" in obs else np.ones(a.n_obs, bool)
# gene symbols
if "gene_symbol" in a.var.columns:
    symbols = a.var["gene_symbol"].astype(str).values
else:
    symbols = a.var_names.astype(str)
idx = {g: int(np.where(symbols == g)[0][0]) for g in GENES if g in set(symbols)}
missing = [g for g in GENES if g not in idx]
print("found", sorted(idx), "missing", missing)

# subsample for speed: up to 80k normal cells
rng = np.random.default_rng(0)
ix = np.where(np.asarray(mask))[0]
if len(ix) > 80000:
    ix = rng.choice(ix, size=80000, replace=False)
ix = np.sort(ix)

sub = a[ix, list(idx.values())].to_memory()
if "gene_symbol" in sub.var.columns:
    sub.var_names = sub.var["gene_symbol"].astype(str).values
X = sub.X
if hasattr(X, "toarray"):
    X = X.toarray()
X = np.asarray(X)
# presence
pres = (X > 0).astype(np.float64)
ct = sub.obs[GROUP].astype(str).values
rows = []
for g, j in enumerate(sub.var_names):
    for c in np.unique(ct):
        m = ct == c
        if m.sum() < 10:
            continue
        rows.append({
            "gene": j,
            "cell_state": c,
            "n_cells": int(m.sum()),
            "frac_expr": float(pres[m, g].mean()),
            "mean_counts": float(X[m, g].mean()),
        })
df = pd.DataFrame(rows)
df.to_csv(OUT / "IL27_expression_by_celltype.csv", index=False)

# focus: CD4 receivers + top potential senders
cd4 = df[df["cell_state"].str.startswith(CD4_PAT)].copy()
cd4_sum = cd4.pivot_table(index="cell_state", columns="gene",
                          values="frac_expr", aggfunc="mean")
cd4_sum.to_csv(OUT / "IL27_CD4_frac_expr.csv")

# top expressers of ligand subunits
lig = df[df["gene"].isin(["IL27", "EBI3", "IL12A"])].copy()
top = (lig.sort_values("frac_expr", ascending=False)
       .groupby("gene").head(15))
top.to_csv(OUT / "IL27_top_ligand_expressers.csv", index=False)

lines = [
    "IL27 expression note (atlas subsample, normal cells)",
    f"Genes found: {sorted(idx)}",
    f"Genes missing from object: {missing}",
    "",
    "LIANA consensus uses complexes EBI3_IL27 and IL12A_IL27;",
    "both subunits must clear expr_prop=0.1 in the same sender.",
    "",
    "CD4 receptor-side frac_expr (IL27RA / IL6ST / IL12RB2):",
]
if len(cd4_sum):
    lines.append(cd4_sum.round(3).to_string())
lines.append("")
lines.append("Top ligand-subunit expressers (frac_expr):")
lines.append(top.round(3).to_string(index=False))
(OUT / "IL27_expression_note.txt").write_text("\n".join(lines))
print("\n".join(lines[-40:]))
print("Wrote under", OUT)
