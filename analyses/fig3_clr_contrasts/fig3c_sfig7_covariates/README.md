# Figure 3c and Supplementary Figure 7 — covariates

Theil’s U heatmap (Supplementary Figure 7) and the heavier
composition-versus-expression revision (Figure 3c).

Input for Theil’s U: all-cells h5ad obs. Missing covariate columns are
skipped. Expression PCR needs `scanpy` and `HGCA_OBJECTS` (four lineage
h5ads).

Script: `src/08_theils_u_heatmap.py`

Output: `theils_u_confounding_matrix.csv` and a heatmap.

```bash
python analyses/fig3_clr_contrasts/fig3c_sfig7_covariates/src/08_theils_u_heatmap.py \
  --h5ad /path/to/atlas.h5ad --outdir /tmp/theils_u
```

Full expression revision: set `HGCA_OBJECTS` and run
`01_cache_expression_embeddings.py` through
`07_finalize_authoritative.py`.
