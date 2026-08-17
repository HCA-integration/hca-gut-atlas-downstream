# HGCA v1 demo slice

`hgca_all_lineages_v1_demo.h5ad` is a 1.4 MB real slice of the 15 GB
all-cells object. It is for software checking, not manuscript figures.

## Why this size

| Venue | Limit | This file |
|---|---|---|
| GitHub | Warn 50 MB / block 100 MB per file; repos ideally <1 GB | 1.4 MB |
| Code Ocean | 5 GB capsule workspace; local `/data` should be a small example | Fits easily |
| Zenodo | 50 GB / record | Not the constraint |

## What is in it

- 3,185 cells, 1,668 genes (symbols as `var_names`, Ensembl in `gene_id`)
- All 94 `hgca_celltype_v1` labels
- 131 samples, 23 datasets
- ≥5 samples in ileum/colon × biopsy/resection
- 17 samples with ≥3 GC B LZ/DZ cells and 109 with zero (follicle k=3)
- Hard-coded covariates left as original strings

Run the laptop demo (does not need the 15 GB atlas):

```bash
python src/run_demo.py
```

Rebuild the slice from the full atlas (authors only):

```bash
export HGCA_H5AD=/path/to/hgca_all_lineages_v1.h5ad
python src/build_hgca_v1_demo_slice.py --source "$HGCA_H5AD"
```
