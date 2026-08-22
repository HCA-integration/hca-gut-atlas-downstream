# Figure 3 — CLR compositions and radial contrasts

Two CLR tables, same formula (log(count + 1) − row mean), different
universes:

| Table | Universe | Used for |
|---|---|---|
| [`data/fig2/celltype_compositional_enrichment_long.csv`](../../data/fig2/README.md) | One global composition per sample (all taxonomy v1 labels) | **Figure 3a heatmap** |
| [`data/composition/clr_long.csv`](../../data/composition/README.md) | Within-lineage composition | Fig. 3 contrasts, S7, S8, S11, S12 |

`mean_clr` in the enrichment table is not row-scaled. The heatmap paints
`row_z` (per cell type, z-score of category-mean CLRs across tissue or
radial levels). Do not rebuild Figure 3a from `clr_long.csv`.

```bash
Rscript analyses/fig3_clr_contrasts/src/render_fig3a_anatomical_clr_heatmap.R \
  --enrichment data/fig2/celltype_compositional_enrichment_long.csv
```

Related code in this directory:

- [`fig3c_sfig7_covariates/`](fig3c_sfig7_covariates/) — Figure 3c and Supplementary Figure 7
- [`fig3i_sfig12_follicle/`](fig3i_sfig12_follicle/) — Figure 3i and Supplementary Figure 12
- `src/render_sfig8_segment_examples.R` — Supplementary Figure 8 gut-axis splines

Input for the within-lineage tables: all-cells h5ad (obs only), or four
lineage objects via `HGCA_OBJECTS`. Contrasts use `tissue_level_1`.
Covariate audit tables also use `sampled_site_condition`,
`radial_tissue_term`, `sample_preservation_method`, `sex_ontology_term`,
`age_range`, `assay`, `sample_collection_method`, `sequenced_fragment`,
`gene_annotation_version`.

```bash
python analyses/fig3_clr_contrasts/src/recompute_clr_tables.py \
  --all-cells /path/to/atlas.h5ad --outdir /tmp/clr
```

`--all-cells` splits on `hgca_celltype_level1`. Mann–Whitney U requires
at least five samples per group; BH-FDR is per contrast.

```bash
Rscript analyses/fig3_clr_contrasts/src/render_sfig8_segment_examples.R \
  --clr-long data/composition/clr_long.csv
```

R and ggplot2 also re-render volcanoes and age/collection splines.
