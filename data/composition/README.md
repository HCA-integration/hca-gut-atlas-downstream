# Full-atlas composition tables

`clr_long.csv` is the within-lineage CLR table from
`hgca_all_lineages_v1` (33,432 sample × type rows; 502 samples and 94
types overall). A sample appears in a lineage block only when that
lineage is present. Rebuilt with
`analyses/fig3_clr_contrasts/src/recompute_clr_tables.py` and
pseudocount 1.

Use this for Figure 3 / Supplementary Figures 7, 11, and 12 instead of
the bundled 3,185-cell subset. It is still not a substitute for the
h5ad when an analysis needs expression.

```bash
python analyses/fig3_clr_contrasts/src/recompute_clr_tables.py \
  --all-cells /path/to/hgca_all_lineages_v1.h5ad \
  --outdir data/composition
```

Wilcoxon contrast tables and the covariate-audit folder were written
in the same run.
