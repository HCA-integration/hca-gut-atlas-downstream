# Figure 2c — leave-one-dataset-out SCANVI

SCANVI LODO used for the v0 / v1 / PanGI macro-F1 comparison (Methods,
LODO Benchmarks). Every contributing dataset is held out in turn; a
model is trained on the rest and the holdout is mapped back.

This is not a demo. A full lineage screen is GPU work on the order of
a day or more (lymphoid mapping-stability was ~24 hours; LODO is as
long or longer because every dataset is a fold). Checkpoints and fold
predictions are not stored in git.

```bash
export HGCA_OBJECTS=/path/to/lineage-h5ads
python analyses/fig2c_fig4c_sfig14_scanvi/lodo/run_lodo.py \
  --lineage myeloid --lodo-cv-all
```

PanGI configs (`pangi_*`) use `PANGI_H5AD` and subset on `level_1_annot`.
`plot_hgca_v1_per_class_f1.py` aggregates fold CSVs after a run.

Recipe (shared HVG / architecture): 4000 HVGs, `n_latent=30`,
`n_layers=2`, NB, batch key `sample_id` (HGCA) or `sampleID` (PanGI).

Submitted Fig. 2c LODO used **two different training protocols**:
HGCA lineage YAMLs run scVI pretrain plus a 10-epoch scANVI retrain
per fold (`n_epochs: 10`, `use_scvi_pretrain: true`). PanGI YAMLs
(`pangi_*`) set `use_pretrained_embedding: true`, `n_epochs: 0`, and
`frozen_latent_solver: saga` — multinomial logistic regression on the
published `X_scANVI` embedding, no scANVI retrain. Fig. 4c mapping
stability retrains both atlases with the same SCANVI recipe.

The manuscript Methods currently describe a single shared scANVI
LODO pipeline. That wording is a revision item; do not silently
equalize the configs here.

For mapping a *new* query onto the published lineage references, use
[ArchMap](https://www.archmap.bio/) once the four scANVI models are
uploaded. That is Fig. 4b-style transfer, not this LODO screen.

scimilarity / Nicheformer / SampleCLR scripts from the original
`reference_mapping_benchmark` tree are not copied. They are not the
paper LODO.
