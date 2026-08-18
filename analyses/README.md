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
| Figure 2c, 4c, Supplementary Figure 14 | Frozen SCANVI recipes; taxonomy path-distance check | [`fig2c_fig4c_sfig14_scanvi`](fig2c_fig4c_sfig14_scanvi/) | Bundled taxonomy for the path-distance check. Full LODO / jackknife needs GPU and `reference_mapping_benchmark` |
| Figure 3a–f | Within-lineage CLR compositions and Mann–Whitney contrasts | [`fig3_clr_contrasts`](fig3_clr_contrasts/) | Full-atlas tables in `data/composition/`, or recompute from the all-cells h5ad |
| Figure 3c; Supplementary Figure 7 | Theil’s U confounding; composition-versus-expression revision | [`fig3_clr_contrasts/fig3c_sfig7_covariates`](fig3_clr_contrasts/fig3c_sfig7_covariates/) | All-cells obs for Theil’s U. Expression PCR needs lineage objects |
| Figure 3i; Supplementary Figure 12 | Follicle niche capture from composition | [`fig3_clr_contrasts/fig3i_sfig12_follicle`](fig3_clr_contrasts/fig3i_sfig12_follicle/) | Full-atlas `data/composition/clr_long.csv` (HGCA only) |
| Figure 4a–b | TAURUS author versus HGCA-transferred labels | [`fig4_hgca_taurus_refinement`](fig4_hgca_taurus_refinement/) | TAURUS obs CSV (`TAURUS_OBS`); taxonomy CSV |
| Figure 5 | Organoid benchmark on frozen sysVI outputs | [`fig5_organoid_benchmark`](fig5_organoid_benchmark/) | Frozen HEOCA query, metadata, and distance files |
| Supplementary Figure 6 | Published versus contributed coverage; DESeq2 power | [`sfig6_prepub_contributions`](sfig6_prepub_contributions/) | Four lineage h5ads (`HGCA_OBJECTS`) |
| Supplementary Figure 9 | Visium cell2location | [`sfig9_visium_cell2location`](sfig9_visium_cell2location/) | Visium sections and a downsampled reference; GPU |
| Supplementary Figure 11 | Sample-level CLR Spearman correlations | [`sfig11_compositional_correlations`](sfig11_compositional_correlations/) | Full-atlas tables in `data/composition/` |
| Supplementary Figure 13 | LIANA rank-aggregate CCC | [`sfig13_liana`](sfig13_liana/) | All-cells h5ad; not the bundled subset |
| Supplementary Figure 16a,c | patpy embeddings and pretreatment remission AUC | [`sfig16_patpy_embeddings`](sfig16_patpy_embeddings/) | TAURUS h5ad (`TAURUS_H5AD`). SampleCLR attention (16b) is not here |
| Extended Data Table 1 | Dataset inventory with live counts | [`ed_table1_datasets`](ed_table1_datasets/) | All-cells h5ad. DOI / PI enrichment needs local metadata |
| Extended Data Table 2 | CAP taxonomy | `data/demo/GCA_taxonomy_2026_CAP.csv` | Bundled file |
| Extended Data Table 5 | GC-module gene list | `data/demo/follicle_gsva_gc_b_gene_list.csv` | Bundled file |

Atlas construction (Figure 1 integration, Supplementary Figures 1b–c
and 3), Xenium (Figure 3g, 3j), segment GO (Supplementary Figure 10),
and TAURUS Milo / SampleCLR attention (Figure 4e, 4h–i, Supplementary
Figure 16b) are not in this tree yet.

Full-atlas CLR tables for the composition analyses are in
[`data/composition/`](../data/composition/).

Install once from the repository root (`python -m pip install -r requirements.txt`).
A small subset can be used to check that several of these scripts run
(`python src/run_demo.py`); those outputs are not manuscript values.
