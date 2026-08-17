# P12 — Organoid benchmark (Fig. 5)

Frozen sysVI label-transfer outputs. Mapping itself is not retrained here.

**No laptop demo yet.** Inputs are collaborator HEOCA files, not the healthy
atlas slice.

## System requirements

- Python 3.12 with `pandas`, `numpy`, `pyyaml`
- Frozen query h5ad, metadata workbook, and per-cell distance CSV

## Instructions for use

```bash
export HGCA_OBJECTS=/path/to/lineage-h5ads
export HEOCA_QUERY=/path/to/organoid_query_predictions.h5ad
export HEOCA_METADATA=/path/to/HCA_Organoid_Atlas.xlsx
export HEOCA_DISTANCES=/path/to/organoid_per_cell_distances.csv
python analyses/fig5_organoid_benchmark/src/01_build_tables.py
```

Confidence threshold 0.5; k=30 in the frozen mapping; proximity formula and
HC3 OLS are in the paper Methods. A tiny 2–3 sample demo table is still to
be added.
