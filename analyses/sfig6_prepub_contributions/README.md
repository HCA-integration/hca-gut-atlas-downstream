# Supplementary Figure 6 — contribution and DESeq2 power

Published-versus-contributed coverage and DESeq2 Wald power
(Supplementary Figure 6; also Figure 2e).

Input: four lineage h5ads (`HGCA_OBJECTS`). Bar-panel rendering also
needs `HGCA_CAP_DIR`. These are not the bundled subset.

Scripts: `src/compute_deseq2_analytical_power.py` (and related power
scripts); `src/render_s7_prepub.py` (filename is historical).

```bash
export HGCA_OBJECTS=/path/to/lineage-h5ads
python analyses/sfig6_prepub_contributions/src/compute_deseq2_analytical_power.py \
  --objects "$HGCA_OBJECTS"
```

Requires `scanpy` and, for the power curves, pyDESeq2.
