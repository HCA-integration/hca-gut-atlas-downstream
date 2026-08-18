# Figure 2 — atlas-building evidence

Author-label crosswalk versus HGCA v0, plus cell-type and dataset support
tables.

Input: all-cells h5ad and the taxonomy CSV. Required obs columns include
`author_cell_type`, `closest_GCA_celltype`, `hgca_celltype_v0`,
`hgca_celltype_v1`, `hgca_celltype_level1`, `dataset_id`, `sample_id`,
`donor_id`, `tissue_level_1`, `tissue_level_2`, `radial_tissue_term`,
`sample_collection_method`, `disease`, `sampled_site_condition`.

Script: `src/build_fig2_atlas_evidence.py`

Output: sidecar CSVs under the figure directory (`data/` and `out/`).
The compositional enrichment table (`mean_clr` plus display-only
`row_z`) is also checked in at
[`data/fig2/celltype_compositional_enrichment_long.csv`](../../data/fig2/README.md)
and is the input for the Figure 3a heatmap.

```bash
python analyses/fig2_label_set/src/build_fig2_atlas_evidence.py \
  --metadata /path/to/atlas.h5ad \
  --taxonomy /path/to/GCA_taxonomy_2026_CAP.csv \
  --figure-dir /tmp/fig2
```

`--demo` skips LODO F1, CAP votes, and PanGI prediction caches. CAP
fixtures for a non-demo run are in `data/cap/` (see
[`sfig4_cap_votes`](../sfig4_cap_votes/)). Paper LODO numbers need
`HGCA_BENCHMARK_RESULTS`.

Figure 2b trees are built with
[ARBOL](https://github.com/jo-m-lab/ARBOL). Install that package to
rebuild the dendrograms. This repository vendors `src/arbol.R` plus
frozen Post-CAP SVGs and the v1 graph under `data/arbol/` so the sidecar
overlays can be re-rendered without rerunning ARBOL
(`ARBOL_ROOT`, `ARBOL_GRAPH`, `ARBOL_R` override).
