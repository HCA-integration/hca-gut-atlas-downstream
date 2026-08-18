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
R and cairosvg are used only for the optional ARBOL sidecar renderer.
