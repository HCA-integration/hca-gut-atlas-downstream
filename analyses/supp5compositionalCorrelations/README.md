# P05 — Compositional correlations (Supp. Fig. 11)

Spearman correlations on joint CLR abundances, with support gates.

**DEMO MODE: the slice is sparse; many pairs will fail the n≥30 / detect≥20
gates. That is expected.**

## System requirements

- OS tested: macOS 15
- Python 3.12 with `pandas`, `numpy`, `scipy`, `statsmodels`

## Demo

Requires CLR tables from the main demo first:

```bash
python analyses/supp5compositionalCorrelations/src/compute_compositional_correlations.py \
  --clr-long data/demo/expected/clr/clr_long.csv \
  --lineage-map data/demo/expected/clr/celltype_lineage_map.csv \
  --outdir data/demo/expected/correlations
```

Or `python src/run_demo.py`. Runtime: under a minute.
