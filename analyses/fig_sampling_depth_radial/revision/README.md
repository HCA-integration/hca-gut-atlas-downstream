# P04 revision — covariate confounding and variance

Theil’s U heatmap (paper Supp. Fig. 7) plus the heavier composition-vs-expression
revision pipeline.

**DEMO MODE: Theil’s U on the slice is a software check, not the paper matrix.**

## System requirements

- OS tested: macOS 15
- Python 3.12 with `h5py`, `pandas`, `numpy`, `scipy`, `matplotlib`
- Full expression PCR needs `scanpy` and `HGCA_OBJECTS` (four lineage h5ads)

## Demo

```bash
python analyses/fig_sampling_depth_radial/revision/src/08_theils_u_heatmap.py \
  --h5ad data/demo/hgca_all_lineages_v1_demo.h5ad \
  --outdir data/demo/expected/theils_u
```

Expected: `theils_u_confounding_matrix.csv` and a heatmap. Missing obs
columns are skipped. Runtime: seconds.

## Full reproduction

Set `HGCA_OBJECTS` to the directory with the four lineage objects and run
`01_cache_expression_embeddings.py` through `07_finalize_authoritative.py`.
That path is optional for review.
