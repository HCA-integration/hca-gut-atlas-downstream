# P10 — Reference mapping metrics (Fig. 2c, 4c, S14)

Frozen SCANVI recipes and the taxonomy path-distance metric used for
jackknife naming depth. **This package does not retrain SCANVI.**

**DEMO MODE: the laptop smoke is a software check, not the paper F1 /
stable-resolution tables.**

## System requirements

- OS tested: macOS 15
- Python 3.12 with `anndata`, `pandas` for the smoke
- Full jackknife / LODO training: GPU, scvi-tools, and the sibling
  `reference_mapping_benchmark` repository

## Installation

From the repository root: `python -m pip install -r requirements.txt`.

## Demo

```bash
python analyses/reference_uncertainty/src/smoke_taxonomy_paths.py \
  --taxonomy data/demo/GCA_taxonomy_2026_CAP.csv \
  --h5ad data/demo/hgca_all_lineages_v1_demo.h5ad \
  --outdir data/demo/expected/mapping
```

Expected: every demo `hgca_celltype_v1` label is in the taxonomy; self
path-distance is 0; GC B LZ vs DZ is a small positive distance; LZ vs
FARM is larger. Runtime: seconds.

Or `python src/run_demo.py`.

## Frozen recipe (paper run, not the demo)

| File | Role |
|---|---|
| `configs/*_scanvi_recipe_frozen.json` | n_latent=30, 4000 HVGs, 10 epochs, gene_likelihood=nb |
| `manifests/*_shared_studies_exact_name.csv` | Studies omitted in the jackknife |
| `manifests/lodo_prespecified_jackknife_prediction.json` | Locked before jackknife results |

HGCA uses SCANVI with scVI pretraining; the matched PanGI full-reference
run used direct SCANVI. Do not “fix” that difference.

## Instructions for use

Point a trained prediction table at the same path-distance helper, or
retrain only from `reference_mapping_benchmark` with the frozen JSON.
A 5-minute CPU F1 smoke on a tiny frozen prediction CSV is still to be
added; do not treat a laptop as able to reproduce Fig. 2c / 4c.
