# Figure 3 — sampling depth and radial composition

Within-lineage CLR compositions and Mann–Whitney contrasts for biopsy vs
resection and full-thickness vs other radial layers.

**DEMO MODE: results from the bundled slice are for software checking,
not manuscript figures.**

## System requirements

- OS tested: macOS 15
- Python 3.12 with `h5py`, `pandas`, `numpy`, `scipy`, `statsmodels`
- R + ggplot2 to re-render volcanoes (not required for the demo)
- Non-standard hardware: none for the demo

## Installation

From the repository root: `python -m pip install -r requirements.txt`
(under 5 minutes).

## Demo

```bash
python analyses/fig_sampling_depth_radial/src/recompute_clr_tables.py \
  --all-cells data/demo/hgca_all_lineages_v1_demo.h5ad \
  --outdir data/demo/expected/clr
```

| | |
|---|---|
| Input | demo all-cells h5ad (obs only; expression is unused) |
| Expected output | `clr_long.csv` and contrast tables under `data/demo/expected/clr/` |
| Runtime | about a minute |

CLR is computed **within each lineage** (crosstab + pseudocount **0.5** +
centred log-ratio), then concatenated. Mann–Whitney U requires ≥5
samples per group; BH-FDR is per contrast.

Covariates used in the audit tables: `sampled_site_condition`,
`radial_tissue_term`, `sample_preservation_method`, `sex_ontology_term`,
`age_range`, `assay`, `sample_collection_method`, `sequenced_fragment`,
`gene_annotation_version`. Contrasts also use `tissue_level_1`.

Note: Methods currently say CLR pseudocount 1; this script uses 0.5.
Reconcile before the accepted version.

## Instructions for use

```bash
python analyses/fig_sampling_depth_radial/src/recompute_clr_tables.py \
  --all-cells /path/to/atlas.h5ad --outdir /tmp/clr
```

`--all-cells` splits on `hgca_celltype_level1`. To rebuild from the four
lineage objects, set `HGCA_OBJECTS` to that directory.

Follicle / rare-cell tables: see `rare_cell_defensibility/README.md`.
