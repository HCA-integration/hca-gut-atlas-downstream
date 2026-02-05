import numpy as np
import pandas as pd
import scanpy as sc

tissue_col = "tissue_ontology_term"
ct_col = "Prelim annotation"

def segment_from_tissue(x):
    s = str(x).lower()
    if "duod" in s:
        return "duodenum"
    if "jejun" in s:
        return "jejunum"
    if "ile" in s:
        return "ileum"
    if "colon" in s or "large intestine" in s:
        return "colon"
    return "other"

ad = adata.copy()
ad.obs["segment_simple"] = ad.obs[tissue_col].map(segment_from_tissue)
ad = ad[ad.obs["segment_simple"].isin(["duodenum", "jejunum", "ileum", "colon"])].copy()

counts = pd.crosstab(ad.obs[ct_col].astype(str), ad.obs["segment_simple"].astype(str))
cts = counts[(counts >= 50).all(axis=1)].index.tolist()
if len(cts) == 0:
    cts = counts.sum(axis=1).sort_values(ascending=False).head(5).index.tolist()
target_ct = cts[0]

sub = ad[ad.obs[ct_col].astype(str) == str(target_ct)].copy()

iron_duo = ["CYBRD1", "SLC11A2", "SLC40A1", "HEPH"]
bulk_jej = ["SLC15A1", "ALPI", "SI", "MGAM", "APOB", "MTTP"]
bile_ile = ["SLC10A2", "FABP6", "SLC51A", "SLC51B", "FGF19"]
colon_barrier = ["MUC2", "TFF3", "CA4", "SLC5A8"]

sets = {
    "Iron absorption": iron_duo,
    "Bulk absorption": bulk_jej,
    "Bile acid reclaim": bile_ile,
    "Colon barrier": colon_barrier,
}

for name, genes in sets.items():
    genes_present = [g for g in genes if g in sub.var_names]
    if len(genes_present) >= 2:
        sc.tl.score_genes(sub, gene_list=genes_present, score_name="score_" + name.replace(" ", "_"), use_raw=None)

score_cols = [c for c in sub.obs.columns if c.startswith("score_")]
if len(score_cols) == 0:
    raise ValueError("No module genes were found in var_names. If you use gene symbols in adata.var['gene_symbol'], tell me and I will adapt this block.")

sc.pl.violin(
    sub,
    keys=score_cols,
    groupby="segment_simple",
    stripplot=False,
    jitter=False,
    rotation=30,
    multi_panel=True,
)

print("Used cell type:", target_ct)
