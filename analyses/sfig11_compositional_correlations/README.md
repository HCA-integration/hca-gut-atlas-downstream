# Supplementary Figure 11 — compositional correlations

Spearman correlations on joint CLR abundances, with support gates
(at least 30 samples and 20 detections per member of a pair).

Input: `clr_long.csv` and `celltype_lineage_map.csv` from
[`data/composition/`](../../data/composition/) (full atlas) or from a
local `fig3_clr_contrasts` rebuild.

Script: `src/compute_compositional_correlations.py`

```bash
python analyses/sfig11_compositional_correlations/src/compute_compositional_correlations.py \
  --clr-long /path/to/clr_long.csv \
  --lineage-map /path/to/celltype_lineage_map.csv \
  --outdir /tmp/correlations
```

On the bundled subset many pairs fail the support gates; that is
expected.
