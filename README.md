# Human Gut Cell Atlas v1 — downstream analyses

Code for downstream analyses in *The Human Gut Cell Atlas v1.0*, plus a
laptop-scale demo that a reviewer can run without the 15 GB atlas.

This repository was rebuilt for publication. The previous working-group
vignette tree is preserved on read-only archive branches (see
[docs/archive.md](docs/archive.md)).

**DEMO MODE: numbers from the bundled slice are for software checking,
not manuscript figures.**

## System requirements

- OS tested: macOS 15 (Darwin 24.3)
- Python 3.12.11 with the packages in `requirements.txt`
  (tested: anndata 0.12.3, pandas 2.3.3, numpy 2.3.4, scipy 1.16.2,
  statsmodels 0.14.6, matplotlib 3.10.7, h5py 3.15.1)
- R 4.x + ggplot2 only if you re-render paper figures (not required for the demo)
- Non-standard hardware: none for the demo. Full atlas rebuilds, scVI,
  cell2location, and Xenium need a GPU and live in other repositories
  ([docs/related-repositories.md](docs/related-repositories.md)).

Typical laptop: 8 GB RAM is enough for the 1.4 MB demo object.

## Installation

```bash
git clone https://github.com/HCA-integration/hca-gut-atlas-downstream.git
cd hca-gut-atlas-downstream
python -m pip install -r requirements.txt
```

Typical install time on a laptop: under 5 minutes if Python is present.

## Demo

From the repository root:

```bash
python src/run_demo.py
```

| | |
|---|---|
| Input | `data/demo/hgca_all_lineages_v1_demo.h5ad` (3,185 cells, 94 types) |
| Expected output | Local `data/demo/expected/` (gitignored) plus the smoke-test line: 94 ileum MWU tests and 17 samples with ≥3 GC B cells |
| Runtime | about 30 seconds on a laptop |

The smoke test `python src/smoke_demo_slice.py` is a shorter check: it
should report 94 Mann–Whitney tests on ileum biopsy vs resection and 17
samples with ≥3 GC B cells.

Do not treat effect sizes or FDRs as paper results.

## Instructions for use

Point any wired script at your own h5ad. Column names are hard-coded
and must match the atlas:

```bash
export HGCA_H5AD=/path/to/your_atlas.h5ad
python analyses/fig1_donor_age/src/build_donor_age_counts.py --h5ad "$HGCA_H5AD" --outdir /tmp/hgca_out
python analyses/fig_sampling_depth_radial/src/recompute_clr_tables.py --all-cells "$HGCA_H5AD" --outdir /tmp/hgca_clr
```

Required obs columns include `hgca_celltype_v1`, `hgca_celltype_level1`,
`sample_id`, `donor_id`, `dataset_id`, `tissue_level_1`,
`sample_collection_method`, `radial_tissue_term`, and the other
covariates listed in each analysis README.

Full paper numbers need the 15 GB all-cells object (or the four lineage
objects) and, for Figure 2, the LODO / CAP / PanGI caches. That is
optional for review.

| Variable | Purpose |
|---|---|
| `HGCA_H5AD` | All-cells `.h5ad` (demo is the default) |
| `HGCA_OBJECTS` | Directory with `epithelial.h5ad`, `lymphoid.h5ad`, `myeloid.h5ad`, `stroma.h5ad` |
| `HGCA_TAXONOMY` | Taxonomy CSV (demo copy is bundled) |

## Repository layout

```
data/demo/                      bundled slice + companion CSVs
src/run_demo.py                 one-command laptop check
analyses/fig1_donor_age         Figure 1 donor-age counts
analyses/fig2_label_set         Figure 2 metadata / author-crosswalk tables
analyses/fig_sampling_depth_radial
                                Figure 3 CLR, contrasts, follicle, Theil’s U
analyses/supp5compositionalCorrelations
                                Supp. Fig. 11 correlations
analyses/supp_table_datasets    ED Table 1 live counts
analyses/reference_uncertainty  Frozen SCANVI recipes + taxonomy-path smoke
analyses/visium_cell2location   Supp. Fig. 9 notebooks (not a laptop demo)
analyses/fig4_hgca_taurus_refinement
                                Figure 4a–b (needs TAURUS obs)
analyses/fig5_organoid_benchmark
                                Figure 5 (needs frozen HEOCA files)
analyses/s7_prepub_contributions
                                Supp. Fig. 6 (needs lineage objects)
```

See [docs/CHECKLIST.md](docs/CHECKLIST.md) for package status and
[docs/related-repositories.md](docs/related-repositories.md) for
collaborator code that is still outside this tree.

## License

MIT. See [LICENSE](LICENSE).
