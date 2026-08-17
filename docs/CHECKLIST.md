# Nature software checklist (HGCA v1)

Status on `hca-gut-atlas-downstream`. `[x]` done, `[~]` partial, `[ ]` not started.
Demo numbers are for software checking, not manuscript figures.

## Repo-wide

- [x] OSI license (`LICENSE`, MIT)
- [x] Root README with Nature sections (system requirements, install, demo, usage)
- [x] `requirements.txt` + `environment.yml`
- [x] Laptop demo: `python src/run_demo.py` (~20–40 s)
- [x] Demo slice in git (`data/demo/hgca_all_lineages_v1_demo.h5ad`, 1.4 MB)
- [x] Generated demo tables gitignored (`data/demo/expected/`)
- [x] Archive branches documented (`docs/archive.md`)
- [ ] Colleague who did not write the code runs the demo
- [ ] Code Availability statement rewritten in the manuscript
- [ ] Reporting Summary software/code section
- [ ] Release tag / commit SHA named in the manuscript
- [ ] Zenodo DOI of the accepted tag (publication, not review)

## Packages

| ID | Paper | In this repo | Laptop demo | Blocker |
|---|---|---|---|---|
| P01 MetaManager | S2 | pointer only | no | pin tagged release |
| P02 Atlas construction | Fig. 1, S1, S3 | no | no | **Chris Lance** |
| P03 Fig. 2 evidence | Fig. 2, S4 | `analyses/fig2_label_set` | yes (`--demo`) | CAP vote fixture optional |
| P04 Composition / covariates | Fig. 3a–f, S6–8 | `fig1_donor_age`, `fig_sampling_depth_radial` (+ `revision/`) | yes (CLR, MWU, donor-age, Theil’s U) | CLR pseudocount 0.5 vs Methods 1; pyDESeq2 power |
| P05 Follicle / correlations | Fig. 3i, S11–12 | `rare_cell_defensibility`, `supp5compositionalCorrelations` | yes (niche + correlations from demo CLR) | GSVA still needs lineage objects |
| P06 Visium cell2location | S9 | `analyses/visium_cell2location` (notebooks; NMF not copied) | no | tiny Visium section; strip `/nfs` paths |
| P07 Xenium | Fig. 3g | no | no | **Chris Lance** / LMU |
| P08 LIANA | S13 | no | no | collect primary script |
| P09 Segment GO | S10 | no | no | collaborator DESeq2 + gProfiler2 |
| P10 Mapping / jackknife | Fig. 2c, 4c, S14 | `analyses/reference_uncertainty` | yes (taxonomy path-distance) | Frozen prediction F1 table; full SCANVI is GPU |
| P11 TAURUS downstream | Fig. 4 | `fig4_hgca_taurus_refinement` (source) | no | TAURUS obs CSV; Milo/SampleCLR; **Chris** |
| P12 Organoid | Fig. 5 | `fig5_organoid_benchmark` (source + config) | no | frozen HEOCA query / distances |
| P13 Tables | ED T1–2, T5 | dataset builder + bundled taxonomy + GC gene list | yes (live counts from demo) | DOI/tier1 enrichment needs local metadata |

## Next

1. You: commit the current working tree (gitignore + new analyses). Do not ask the agent to commit.
2. Colleague clone-and-run of `python src/run_demo.py`.
3. Chris: P02 / P07 / P11.
4. Collect P08 / P09 scripts; a public or authorized Visium section for P06.
5. Manuscript Code Availability + Reporting Summary.
