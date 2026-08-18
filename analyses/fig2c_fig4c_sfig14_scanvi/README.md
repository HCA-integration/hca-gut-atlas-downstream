# Figure 2c, 4c, Supplementary Figure 14 — mapping recipes

Frozen SCANVI recipes and the taxonomy path-distance metric used for
jackknife naming depth. This directory does not retrain SCANVI.

Input for the path-distance check: bundled taxonomy and the all-cells
subset (or any h5ad with `hgca_celltype_v1`). Full LODO / jackknife
training needs a GPU, scvi-tools, and `reference_mapping_benchmark`.

Script: `src/smoke_taxonomy_paths.py`

```bash
python analyses/fig2c_fig4c_sfig14_scanvi/src/smoke_taxonomy_paths.py \
  --taxonomy data/demo/GCA_taxonomy_2026_CAP.csv \
  --h5ad data/demo/hgca_all_lineages_v1_demo.h5ad \
  --outdir /tmp/mapping
```

The check confirms that every subset label is in the taxonomy and that
path-distance is zero for a label versus itself, small for GC B LZ
versus DZ, and larger for LZ versus FARM. It is not the paper F1 or
stable-resolution table.

| File | Role |
|---|---|
| `configs/*_scanvi_recipe_frozen.json` | n_latent=30, 4000 HVGs, 10 epochs, gene_likelihood=nb |
| `manifests/*_shared_studies_exact_name.csv` | Studies omitted in the jackknife |
| `manifests/lodo_prespecified_jackknife_prediction.json` | Locked before jackknife results |

HGCA uses SCANVI with scVI pretraining; the matched PanGI full-reference
run used direct SCANVI. A frozen prediction CSV for an F1 check has not
been added.
