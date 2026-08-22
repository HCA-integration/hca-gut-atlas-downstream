# Figure 1 — donor-age counts

Unique donors by `age_range` for ileum and colon.

Input: all-cells h5ad with `donor_id`, `age_range`, `tissue_level_1`.
`age_range` uses the atlas bins (`0-9` … `80-89`, `unknown`).

Script: `src/build_donor_age_counts.py`

Output: `donor_age_by_tissue.csv`

```bash
python analyses/fig1_donor_age/src/build_donor_age_counts.py \
  --h5ad /path/to/atlas.h5ad --outdir /tmp/fig1
```

The bundled subset can be used to check that the script runs; counts
will not match the paper.
