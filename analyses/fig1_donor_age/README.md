# Figure 1 — donor-age counts

Unique donors by `age_range` for ileum and colon.

**DEMO MODE: results from the bundled slice are for software checking,
not manuscript figures.**

## System requirements

- OS tested: macOS 15
- Python 3.12 with `anndata`, `pandas`
- Non-standard hardware: none

## Installation

From the repository root: `python -m pip install -r requirements.txt`
(under 5 minutes).

## Demo

```bash
python analyses/fig1_donor_age/src/build_donor_age_counts.py \
  --h5ad data/demo/hgca_all_lineages_v1_demo.h5ad \
  --outdir data/demo/expected/fig1
```

Expected output: `donor_age_by_tissue.csv`. Runtime: seconds.

Or run `python src/run_demo.py`.

## Instructions for use

```bash
python analyses/fig1_donor_age/src/build_donor_age_counts.py \
  --h5ad /path/to/atlas.h5ad --outdir /tmp/fig1
```

Required obs columns: `donor_id`, `age_range`, `tissue_level_1`.
`age_range` must use the atlas bins (`0-9` … `80-89`, `unknown`).
