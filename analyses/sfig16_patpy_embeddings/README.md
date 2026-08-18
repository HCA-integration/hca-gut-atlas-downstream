# Supplementary Figure 16a,c — patpy embeddings and remission AUC

patpy sample representations on TAURUS and leave-one-patient-out
prediction of pretreatment anti-TNF remission. SampleCLR attention
(panel b) is not here.

Input: TAURUS h5ad (`TAURUS_H5AD`) with HGCA-transferred labels, or
`TAURUS_OBS` for metadata-only stages. Stage 1 exits if `TAURUS_H5AD`
is unset. Optional sidecars (`HGCA_V1_REMAP_SIDECAR`,
`PANGI_LABEL_SIDECAR`) are applied only when those files exist.

```bash
export TAURUS_H5AD=/path/to/taurus.h5ad
python analyses/sfig16_patpy_embeddings/src/stage1_run_representations.py
python analyses/sfig16_patpy_embeddings/src/stage2_benchmark.py
python analyses/sfig16_patpy_embeddings/src/stage3_predict_response.py
python analyses/sfig16_patpy_embeddings/src/stage4_plots.py
```

Requires [patpy](https://github.com/lueckenlab/patpy). Stage 1 is heavy; stages 2–4 read the CSVs it writes
under `data/<labelset>/`. `LABELSET` is `hgca_v1` (default) or
`author_final_analysis`.
