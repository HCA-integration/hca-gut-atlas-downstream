# Supplementary Figure 13 — LIANA cell–cell communication

LIANA `rank_aggregate` per `tissue_level_1` segment, ensemble ranking,
centrality along the gut axis, and the curated Supplementary Figure 13
panels. The bundled subset is too small for a meaningful LIANA run.

The paper tables were produced with `rank_aggregate` (CellPhoneDB,
Connectome, log2FC, NATMI, SingleCellSignalR), `n_perms=1000`,
`expr_prop=0.1`, at most 12,000 cells per segment, consensus resource,
`disease == normal`, and mesentery / appendix dropped.

## Scripts

| Script | Role |
|---|---|
| `src/run_liana_per_tissue.py` | Per-segment LIANA runner |
| `src/ccc_ensemble_synthesis.R` | `ensemble_rank = sqrt(magnitude_rank * specificity_rank)` |
| `src/ccc_centrality_gut_axis.R` | Node centrality and bump charts (panel a) |
| `src/focus_three_groups_liana.R` | BEST4 / macrophage / lymphatic focus |
| `src/00_extract_curated_tables.py` | Curated edges for panels b–e |
| `src/01_render_supp_fig12.R` | Panel renderer (filename is historical) |

## Rebuild the paper run

```bash
export HGCA_H5AD=/path/to/hgca_all_lineages_v1.h5ad
export LIANA_OUTPUT_DIR=/path/to/LIANA_rank_aggregate
export LIANA_METHOD=rank_aggregate
export LIANA_N_PERMS=1000
export LIANA_MAX_CELLS=12000
export LIANA_HIGHRES=0
python analyses/sfig13_liana/src/run_liana_per_tissue.py
```

Then, with `CCC_EDGE_CSV` pointing at `combined_lr_per_tissue_level_1.csv`
and `CCC_OUTPUT_DIR` at the same output root:

```bash
Rscript analyses/sfig13_liana/src/ccc_ensemble_synthesis.R
Rscript analyses/sfig13_liana/src/ccc_centrality_gut_axis.R
bash analyses/sfig13_liana/run_all.sh
```

Requires LIANA 1.7.x, scanpy, and the R packages used by the plotting
scripts. Runtime on the full atlas is hours, not a laptop check.
