# Analyses

| Folder | Paper | Laptop demo |
|---|---|---|
| `fig1_donor_age` | Figure 1 donor-age counts | yes |
| `fig2_label_set` | Figure 2 metadata tables | yes (`--demo`) |
| `fig_sampling_depth_radial` | Figure 3 CLR and contrasts | yes |
| `fig_sampling_depth_radial/rare_cell_defensibility` | Figure 3i follicle niche | yes |
| `fig_sampling_depth_radial/revision` | Supp. Fig. 7 Theil’s U | yes (heatmap); expression PCR needs lineage objects |
| `supp5compositionalCorrelations` | Supp. Fig. 11 | yes, from demo CLR |
| `supp_table_datasets` | ED Table 1 live counts | yes (slice counts ≠ paper) |
| `s7_prepub_contributions` | Supp. Fig. 6 | no (full objects) |
| `visium_cell2location` | Supp. Fig. 9 | no (GPU / Visium) |
| `fig4_hgca_taurus_refinement` | Figure 4a–b | no (TAURUS obs) |
| `fig5_organoid_benchmark` | Figure 5 | no (frozen HEOCA files) |
| `reference_uncertainty` | Fig. 2c / 4c / S14 | yes (taxonomy path-distance only; no SCANVI) |

Run the laptop set with `python src/run_demo.py` from the repository root.
See [docs/CHECKLIST.md](../docs/CHECKLIST.md) for package status.
