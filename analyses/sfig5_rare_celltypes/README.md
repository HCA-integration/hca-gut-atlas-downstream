# Supplementary Figure 5 — rare-type marker dot plots

INFLARE / Brunner subclusters, sinus endothelium, resident macrophage
subtypes, and granulocytes versus other myeloid. Extracted from the
archive notebook `vignettes/RareCellTypes.ipynb` (branch
`archive/kk-local-archive-pre-publication-2026-08-17`). Gene lists
match the submitted panels.

Needs the four lineage objects (`HGCA_OBJECTS`) with expression,
`gene_symbol`, `leiden_lineage_l2`, and `hgca_celltype_v1`. The bundled
subset is too small. Not in `python src/run_demo.py`.

```bash
export HGCA_OBJECTS=/path/to/lineage-h5ads
python analyses/sfig5_rare_celltypes/src/render_sfig5_dotplots.py
```

`--panel a,b,c,d` restricts panels. INFLARE subclusters are `24_4` and
`11_3`. Sinus endothelium is `13_0`–`13_5`.
