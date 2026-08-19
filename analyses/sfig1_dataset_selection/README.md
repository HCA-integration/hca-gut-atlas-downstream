# Supplementary Figure 1a — dataset selection

Study-level inclusion flow (60 candidates → 28 nominated → 26 after
modality review → 24 retained / 27 `dataset_id` values).

The script is the self-contained July 2026 audit block from
`byTheNumbers/Plotting_samples_cells.R`. Counts are hardcoded from the
working-group inventory, not recomputed from the atlas.

```bash
Rscript analyses/sfig1_dataset_selection/src/render_sfig1a_dataset_selection.R
```

Optional argument: output stem (default `out/sfig1a_dataset_selection_flow`).
Requires ggplot2 and svglite.

Supplementary Figure 1b–c (QC and the integration UMAP) are not in this
directory.


# Supplementary Figure 1b — QC metrics by lineage

Python script 

analyses/sfig1_dataset_selection/src/S1_QC.py

Derived code based on the full adata object.

Ensure to enter the right path to the test object or the full HGCA object to reproduce the QC plots.



