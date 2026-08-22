# Figure 2 / 3a compositional enrichment

`celltype_compositional_enrichment_long.csv` is written by
`analyses/fig2_label_set/src/build_fig2_atlas_evidence.py`
(`build_compositional_enrichment`). It is the input for the Figure 3a
heatmap, not `data/composition/clr_long.csv`.

Columns:

- `mean_clr` — category-mean CLR. Not row-scaled.
- `row_z` — per cell type, z-score of those category means across
  annotation levels (ddof=0). Display only; the heatmap paints this.

CLR here is **global**: one composition per sample over all taxonomy
`hgca_celltype_v1` labels (pseudocount 1).
`data/composition/clr_long.csv` is **within-lineage** CLR for the
Mann–Whitney / spline panels. Same formula, different universe.

Rebuild from the all-cells h5ad:

```bash
python analyses/fig2_label_set/src/build_fig2_atlas_evidence.py \
  --metadata /path/to/hgca_all_lineages_v1.h5ad \
  --taxonomy data/demo/GCA_taxonomy_2026_CAP.csv \
  --figure-dir /tmp/fig2
```
