# Figure 5 — organoid benchmark

Downstream tables on frozen sysVI label-transfer outputs. Mapping is
not retrained here.

Input (environment variables): `HGCA_OBJECTS`, `HEOCA_QUERY`,
`HEOCA_METADATA`, `HEOCA_DISTANCES`. Requires `pyyaml` in addition to
the root environment.

Script: `src/01_build_tables.py` (see `config.yaml`)

```bash
export HGCA_OBJECTS=/path/to/lineage-h5ads
export HEOCA_QUERY=/path/to/organoid_query_predictions.h5ad
export HEOCA_METADATA=/path/to/HCA_Organoid_Atlas.xlsx
export HEOCA_DISTANCES=/path/to/organoid_per_cell_distances.csv
python analyses/fig5_organoid_benchmark/src/01_build_tables.py
```

Confidence threshold 0.5; k=30 in the frozen mapping. Proximity and
HC3 OLS are described in Methods. A small organoid subset for a
verification run has not been added.
