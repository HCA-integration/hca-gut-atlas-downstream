# Bundled subset

`hgca_all_lineages_v1_demo.h5ad` is a 1.4 MB slice of the all-cells
object, included so representative scripts can be executed without the
full atlas. Results from this subset are not manuscript values. The
full-atlas CLR table is in [`data/composition/`](../composition/).

- 3,185 cells, 1,668 genes (symbols as `var_names`; Ensembl in `gene_id`)
- All 94 `hgca_celltype_v1` labels
- 131 samples, 23 datasets
- At least five samples in ileum/colon × biopsy/resection
- 17 samples with at least three GC B LZ/DZ cells and 109 with zero

```bash
python src/run_demo.py
```

To rebuild the subset from the full atlas:

```bash
export HGCA_H5AD=/path/to/hgca_all_lineages_v1.h5ad
python src/build_hgca_v1_demo_slice.py --source "$HGCA_H5AD"
```
