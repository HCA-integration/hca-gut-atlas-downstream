# Figure 4c and Supplementary Figure 14 — atlas mapping stability

Shared-study omission: retrain HGCA and PanGI SCANVI after dropping one
shared study, map the same TAURUS cells, then score stable naming depth
(primary τ = 0.90) and composition displacement. The paper calls this
Atlas Mapping Stability / maximum stable mapping resolution.

This is not a demo. The lymphoid screen (3 full seeds + one seed per
omit, TAURUS mapping, then metrics `03_`–`11_`) took about **24 hours**
on a GPU node. Checkpoints and prediction matrices are not in git.

```bash
export HGCA_OBJECTS=/path/to/lineage-h5ads
export PANGI_H5AD=/path/to/pangi_healthy_full.h5ad
export TAURUS_H5AD=/path/to/taurus.h5ad
export HGCA_ANNOT_PARQUET=/path/to/annotation_table.parquet
python analyses/fig2c_fig4c_sfig14_scanvi/mapping_stability/src/train_lineage_reference.py \
  --lineage stroma --atlas HGCA --omit full --seed 0
```

Omit lists are the CSVs in `../manifests/*_shared_studies_exact_name.csv`.
`../manifests/lodo_prespecified_jackknife_prediction.json` is the locked
LODO-to-stability prediction (filename kept; the analysis is mapping
stability). Frozen SCANVI recipes are in `../configs/`.

Queue helpers launch the paper screen without deleting checkpoints.
Metrics scripts `03_`–`11_` read `MAPPING_STABILITY_PREDICTIONS`.

[ArchMap](https://www.archmap.bio/) will host the four full-reference
lineage scANVI models for query mapping. It does not rerun this
omit-one-study screen.
