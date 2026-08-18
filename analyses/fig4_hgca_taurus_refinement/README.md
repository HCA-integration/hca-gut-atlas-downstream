# Figure 4a–b — TAURUS label refinement

Per-cell concordance between author labels and HGCA-transferred v1
labels.

Input: TAURUS obs CSV (`TAURUS_OBS` or `--obs`) and the taxonomy CSV.
The healthy atlas subset is not sufficient.

Script: `src/build_fig4_metrics.py`

```bash
export TAURUS_OBS=/path/to/taurus_gca_v1_label_transfer_obs.csv
python analyses/fig4_hgca_taurus_refinement/src/build_fig4_metrics.py \
  --obs "$TAURUS_OBS" \
  --taxonomy data/demo/GCA_taxonomy_2026_CAP.csv \
  --out-dir /tmp/fig4
```

Milo, SampleCLR, and patpy embeddings are not in this directory.
