# P11 — TAURUS label refinement (Fig. 4a–b)

Per-cell concordance between author labels and HGCA-transferred v1 labels.

**No laptop demo on the healthy slice.** Needs a TAURUS obs CSV.

## Instructions for use

```bash
export TAURUS_OBS=/path/to/taurus_gca_v1_label_transfer_obs.csv
python analyses/fig4_hgca_taurus_refinement/src/build_fig4_metrics.py \
  --obs "$TAURUS_OBS" \
  --taxonomy data/demo/GCA_taxonomy_2026_CAP.csv \
  --out-dir /tmp/fig4
```

Milo, SampleCLR, and patpy embeddings are not in this folder (Chris Lance /
patpy).
