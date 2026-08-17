# Figure 2 — atlas-building evidence

Author-label crosswalk vs HGCA v0, plus cell-type / dataset support tables.

**DEMO MODE: results from the bundled slice are for software checking,
not manuscript figures.**

## System requirements

- OS tested: macOS 15
- Python 3.12 with `anndata`, `pandas`, `numpy`, `matplotlib`
- R + cairosvg only for the ARBOL sidecar renderer (not required for the demo)
- Non-standard hardware: none for the demo

## Installation

From the repository root: `python -m pip install -r requirements.txt`
(under 5 minutes).

## Demo

```bash
python analyses/fig2_label_set/src/build_fig2_atlas_evidence.py --demo
```

| | |
|---|---|
| Input | `data/demo/hgca_all_lineages_v1_demo.h5ad` and `GCA_taxonomy_2026_CAP.csv` |
| Expected output | `data/demo/expected/fig2/data/` sidecar CSVs |
| Runtime | about a minute |

`--demo` skips LODO F1, CAP votes, and PanGI prediction caches.

Hard-coded metadata columns: `author_cell_type`, `closest_GCA_celltype`,
`hgca_celltype_v0`, `hgca_celltype_v1`, `hgca_celltype_level1`,
`dataset_id`, `sample_id`, `donor_id`, `tissue_level_1`, `tissue_level_2`,
`radial_tissue_term`, `sample_collection_method`, `disease`,
`sampled_site_condition`.

## Instructions for use

```bash
python analyses/fig2_label_set/src/build_fig2_atlas_evidence.py \
  --metadata /path/to/atlas.h5ad \
  --taxonomy /path/to/GCA_taxonomy_2026_CAP.csv \
  --figure-dir /tmp/fig2
```

Paper reproduction additionally needs `HGCA_BENCHMARK_RESULTS` and CAP
vote CSVs. That path is optional for review.
