# Extended Data Table 1 — dataset inventory

Live cell, sample, and donor counts per `dataset_id`.

Input: all-cells h5ad. DOI and PI enrichment need local tier-1 metadata
(not in this repository).

Script: `src/build_supp_table_datasets.py`

Output: `SuppTable_HGCA_v1_datasets.csv` / `.xlsx` and a column
dictionary. Requires `openpyxl`.

```bash
python analyses/ed_table1_datasets/src/build_supp_table_datasets.py \
  --h5ad /path/to/hgca_all_lineages_v1.h5ad --outdir /tmp/ed1
```

Counts from the bundled subset will not match the paper. The taxonomy
CSV (Extended Data Table 2) and GC-module gene list (Extended Data
Table 5) are bundled under `data/demo/`.
