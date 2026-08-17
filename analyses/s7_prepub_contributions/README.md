# P04 — Prepublication contribution / DESeq2 power (paper Supp. Fig. 6)

Folder name is historical (S7). The submitted PDF numbers this as **Supp. Fig. 6**.

Scripts compute published-vs-contributed coverage and DESeq2 Wald power.
They expect the full lineage objects, not the demo slice.

```bash
export HGCA_OBJECTS=/path/to/lineage-h5ads
python analyses/s7_prepub_contributions/src/compute_deseq2_analytical_power.py --objects "$HGCA_OBJECTS"
```

Bar-panel rendering also needs `HGCA_CAP_DIR` (the `for_cap` objects). Neither
path is a laptop demo.

A few-cell-type pyDESeq2 demo on the slice is still to be added.
