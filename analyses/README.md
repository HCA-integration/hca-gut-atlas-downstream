# Analyses

Manuscript figures and tables mapped to directories in this repository.
Directories are named for the submitted figure or table. Analyses not
listed here are still being prepared.

| Manuscript item | Analysis | Code | Input |
|---|---|---|---|
| Figure 1 donor-age counts | Unique donors by age range, ileum and colon | [`fig1_donor_age`](fig1_donor_age/) | All-cells h5ad |
| Supplementary Figure 1a | Dataset inclusion flow | [`sfig1_dataset_selection`](sfig1_dataset_selection/) | Hardcoded study-level audit (no atlas object) |
| Figure 2 support tables | Author-label crosswalk and cell-type / dataset support | [`fig2_label_set`](fig2_label_set/) | All-cells h5ad; taxonomy CSV. CAP fixtures in `data/cap/`. LODO / PanGI caches optional |
| Supplementary Figure 4 | CAP project-901 vote pull | [`sfig4_cap_votes`](sfig4_cap_votes/) | Public GraphQL API, or the tracked `data/cap/` snapshot |
| Figure 2b | ARBOL taxonomy trees | [`fig2_label_set`](fig2_label_set/) | Install [ARBOL](https://github.com/jo-m-lab/ARBOL). Frozen Post-CAP SVGs in `data/arbol/` |
| Supplementary Figure 2 | Metadata availability donut; schema validation | [`sfig2_metamanager`](sfig2_metamanager/) | MetaManager commit `9bed5fc73ef2`; plot fixtures in `data/sfig2/` |
| Figure 2c, 4c, Supplementary Figure 14 | SCANVI LODO and shared-study mapping stability | [`fig2c_fig4c_sfig14_scanvi`](fig2c_fig4c_sfig14_scanvi/) | Lineage objects + GPU. Not in the subset demo. Path-distance check uses the bundled taxonomy |
| Figure 3a | Anatomical CLR heatmap (`row_z` of global category-mean CLR) | [`fig3_clr_contrasts`](fig3_clr_contrasts/) | `data/fig2/celltype_compositional_enrichment_long.csv` from the Fig. 2 builder. Not `clr_long.csv` |
| Figure 3b–f | Within-lineage CLR contrasts, volcanoes, collection/age splines | [`fig3_clr_contrasts`](fig3_clr_contrasts/) | Full-atlas tables in `data/composition/`, or recompute from the all-cells h5ad |
| Figure 3c; Supplementary Figure 7 | Theil’s U confounding; composition-versus-expression revision | [`fig3_clr_contrasts/fig3c_sfig7_covariates`](fig3_clr_contrasts/fig3c_sfig7_covariates/) | All-cells obs for Theil’s U. Expression PCR needs lineage objects |
| Figure 3i; Supplementary Figure 12 | Follicle niche capture from composition | [`fig3_clr_contrasts/fig3i_sfig12_follicle`](fig3_clr_contrasts/fig3i_sfig12_follicle/) | Full-atlas `data/composition/clr_long.csv` (HGCA only) |
| Figure 4a–b | TAURUS author versus HGCA-transferred labels | [`fig4_hgca_taurus_refinement`](fig4_hgca_taurus_refinement/) | TAURUS obs CSV (`TAURUS_OBS`); taxonomy CSV |
| Figure 5 | Organoid benchmark on frozen sysVI outputs | [`fig5_organoid_benchmark`](fig5_organoid_benchmark/) | Frozen HEOCA query, metadata, and distance files |
| Supplementary Figure 5 | Rare-type marker dot plots (INFLARE, sinus, macs, granulocytes) | [`sfig5_rare_celltypes`](sfig5_rare_celltypes/) | Four lineage h5ads with expression (`HGCA_OBJECTS`). Not the bundled subset |
| Supplementary Figure 6 | Published versus contributed coverage; DESeq2 power | [`sfig6_prepub_contributions`](sfig6_prepub_contributions/) | Four lineage h5ads (`HGCA_OBJECTS`) |
| Supplementary Figure 8 | Example gut-axis CLR splines (PV macs, Paneth, goblet, Tfr) | [`fig3_clr_contrasts`](fig3_clr_contrasts/) `src/render_sfig8_segment_examples.R` | `data/composition/clr_long.csv` |
| Supplementary Figure 9 | Visium cell2location + NMF compartments | [`sfig9_visium_cell2location`](sfig9_visium_cell2location/) | Teichmann notebooks as received (including NMF). Four healthy adult Visium sections + 5k/type HGCA reference; GPU |
| Supplementary Figure 11 | Sample-level CLR Spearman correlations | [`sfig11_compositional_correlations`](sfig11_compositional_correlations/) | Full-atlas tables in `data/composition/` |
| Supplementary Figure 13 | LIANA rank-aggregate CCC | [`sfig13_liana`](sfig13_liana/) | All-cells h5ad; not the bundled subset |
| Supplementary Figure 16a,c | patpy embeddings and pretreatment remission AUC | [`sfig16_patpy_embeddings`](sfig16_patpy_embeddings/) | TAURUS h5ad (`TAURUS_H5AD`). SampleCLR attention (16b) is not here |
| Extended Data Table 1 | Dataset inventory with live counts | [`ed_table1_datasets`](ed_table1_datasets/) | All-cells h5ad. DOI / PI enrichment needs local metadata |
| Extended Data Table 2 | CAP taxonomy | `data/demo/GCA_taxonomy_2026_CAP.csv` | Bundled file |
| Extended Data Table 5 | GC-module gene list | `data/demo/follicle_gsva_gc_b_gene_list.csv` | Bundled file |

Atlas construction (Figure 1 integration, Supplementary Figures 1b–c
and 3), Xenium (Figure 3g, 3j), segment GO (Supplementary Figure 10),
and TAURUS Milo / SampleCLR attention (Figure 4e, 4h–i, Supplementary
Figure 16b) are not in this tree yet. Figure 2b overlays need
[ARBOL](https://github.com/jo-m-lab/ARBOL) if the trees are rebuilt.

Full-atlas within-lineage CLR tables are in
[`data/composition/`](../data/composition/). The Figure 3a enrichment
table is in [`data/fig2/`](../data/fig2/).

Install once from the repository root (`python -m pip install -r requirements.txt`).
A small subset can be used to check that several of these scripts run
(`python src/run_demo.py`); those outputs are not manuscript values.
