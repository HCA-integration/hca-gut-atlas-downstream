# Figure 2c, 4c, Supplementary Figure 14 — SCANVI LODO and mapping stability

These are the paper’s leave-one-dataset-out and shared-study-omission
SCANVI screens. They live in this repository so a reader can see the
exact configs, omit lists, metrics, and plotters. They are **not** in
`python src/run_demo.py`, they are not a laptop check, and trained
models are not stored here.

| Subdirectory | Paper | Role |
|---|---|---|
| [`lodo/`](lodo/) | Fig. 2c | Leave-one-dataset-out SCANVI (HGCA v0 / v1 / PanGI) |
| [`mapping_stability/`](mapping_stability/) | Fig. 4c, S14 | Shared-study omission; stable naming depth |
| `src/smoke_taxonomy_paths.py` | support | Taxonomy path-distance check on the subset |

## Runtime

A full **mapping-stability** screen for lymphoid (three full-reference
seeds plus one seed per omitted shared study, then TAURUS mapping and
the `03_`–`11_` metrics) took about **24 hours** on a GPU node. The
other lineages are in the same range. **LODO** (every dataset held out,
v0 / v1 / PanGI) is as long or longer. State that in Code Availability
as a compute exception; the desktop demo does not stand in for these
runs.

## What ArchMap covers — and what it does not

Query-to-reference mapping of a new dataset onto the four lineage
scANVI models (the Fig. 4b-style transfer) will be available at
[ArchMap](https://www.archmap.bio/) after those models are uploaded.
Put the four ArchMap atlas / model IDs in this README when they exist.

ArchMap does **not** replace this folder. It will not rerun Fig. 2c
LODO, will not rerun Fig. 4c / S14 mapping stability, and does not host
SampleCLR or patpy.

## Inputs

Full retrain needs the four lineage objects (`HGCA_OBJECTS`), PanGI and
TAURUS h5ads, `HGCA_ANNOT_PARQUET` for stability, a GPU, and
scvi-tools. Frozen SCANVI recipes are in `configs/`. Shared-study omit
lists are in `manifests/`.

```bash
python analyses/fig2c_fig4c_sfig14_scanvi/src/smoke_taxonomy_paths.py \
  --taxonomy data/demo/GCA_taxonomy_2026_CAP.csv \
  --h5ad data/demo/hgca_all_lineages_v1_demo.h5ad \
  --outdir /tmp/mapping
```
