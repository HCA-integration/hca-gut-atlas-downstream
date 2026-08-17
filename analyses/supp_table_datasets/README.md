# P13 — Dataset inventory (ED Table 1)

Live cell / sample / donor counts per `dataset_id`.

**DEMO MODE: counts from the bundled slice will not match the paper.**

## System requirements

- OS tested: macOS 15
- Python 3.12 with `anndata`, `pandas`, `openpyxl`
- Non-standard hardware: none

## Installation

From the repository root: `python -m pip install -r requirements.txt`.

## Demo

```bash
python analyses/supp_table_datasets/src/build_supp_table_datasets.py \
  --h5ad data/demo/hgca_all_lineages_v1_demo.h5ad \
  --outdir data/demo/expected/tables
```

Expected: one row per dataset in the slice (23), plus a column dictionary.
Runtime: seconds. DOI / PI enrichment is skipped unless local tier-1 files
are supplied (they are not in this repo).

## Instructions for use

```bash
python analyses/supp_table_datasets/src/build_supp_table_datasets.py \
  --h5ad /path/to/hgca_all_lineages_v1.h5ad --outdir /tmp/ed1
```

The bundled taxonomy CSV is `data/demo/GCA_taxonomy_2026_CAP.csv` (ED Table 2
source). The GC-module gene list is `data/demo/follicle_gsva_gc_b_gene_list.csv`
(ED Table 5).
