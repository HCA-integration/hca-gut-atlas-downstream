# Figure 3 — CLR compositions and radial contrasts

Within-lineage CLR compositions and Mann–Whitney contrasts for biopsy
versus resection and full-thickness versus other radial layers
(Figure 3a–f).

Related code in this directory:

- [`fig3c_sfig7_covariates/`](fig3c_sfig7_covariates/) — Figure 3c and Supplementary Figure 7
- [`fig3i_sfig12_follicle/`](fig3i_sfig12_follicle/) — Figure 3i and Supplementary Figure 12

Input: all-cells h5ad (obs only; expression is unused), or four lineage
objects via `HGCA_OBJECTS`. Contrasts use `tissue_level_1`. Covariate
audit tables also use `sampled_site_condition`, `radial_tissue_term`,
`sample_preservation_method`, `sex_ontology_term`, `age_range`, `assay`,
`sample_collection_method`, `sequenced_fragment`,
`gene_annotation_version`.

Script: `src/recompute_clr_tables.py`

Output: `clr_long.csv` and contrast tables. The full-atlas run
(pseudocount 1) is checked in at `data/composition/`.

```bash
python analyses/fig3_clr_contrasts/src/recompute_clr_tables.py \
  --all-cells /path/to/atlas.h5ad --outdir /tmp/clr
```

`--all-cells` splits on `hgca_celltype_level1`. CLR is computed within
each lineage (crosstab, pseudocount 1, centred log-ratio).
Mann–Whitney U requires at least five samples per group; BH-FDR is per
contrast.

The Figure 3a heatmap renderer still lives in the local paper tree
(`publication2026/fig3_anatomical_enrichment`). R and ggplot2 are used
here to re-render volcanoes and age/collection splines.
