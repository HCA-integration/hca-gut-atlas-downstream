# Supplementary Figure 9 — Visium cell2location

Teichmann lab notebooks (cell2location 0.1.4), received as
`Notebooks-20260817T212322Z-1-001.zip`. They are the paper analysis.
Sanger `/nfs/...` paths are the original run paths; leave them. This
is not a laptop demo and is not in `python src/run_demo.py`.

## What the manuscript uses

Results: Cell2location mapping of healthy 10X Visium sections recovered
higher-order niches (Supplementary Fig. 9). Factor 4 is the
perivascular / S3 / mast-cell compartment in deep colon mucosa. Factor
8 is the lymphoid follicle (CXCR4 / GC B DZ with CCL19 / FRC).

Methods: downsample ≥5,000 cells per `hgca_celltype_v1`; gene filter
`cell_count_cutoff=5`, `cell_percentage_cutoff2=0.03`,
`nonz_mean_cutoff=1.12`; reference batch `sample_id` plus categorical
`donor_id` and `assay`; then cell2location
`CoLocatedGroupsSklearnNMF` over 10–20 factors.

Those settings are in the notebooks as written.

## Notebooks

| Notebook | Role in S9 |
|---|---|
| `Notebooks/Python_read.ipynb` | Downsample HGCA (paper object: 5,000 / `hgca_celltype_v1`) |
| `Notebooks/Python_C2L_hgca.ipynb` | Train the reference regression model |
| `Notebooks/Python_C2L_infer_hgca.ipynb` | Map four healthy adult Visium sections |
| `Notebooks/Python_NMF_hgca.ipynb` | Colocation NMF (`n_fact` 10–19) and spatial factor maps — this is the S9 compartment figure |
| `Notebooks/Python_visualize_c2l.ipynb` | Per-type Visium plots (BEST4 example); not the factor figure |

The four Visium batches in the infer / NMF notebooks:

- `WSSS_A_GUTsp9518706`
- `WSSS_A_GUTsp9518707`
- `WSSS_A_GUTsp9518708`
- `WSSS_A_GUTsp9518709`

concatenated as `Gut_Visium_healthy_adult_4_samples.h5ad`. The NMF
notebook plots `n_fact13` loadings and spatial factors on each slide.

Reference input used in the training notebook:

`/nfs/users/nfs_d/dp26/nfs_storage/Megagut/downsampled_5000_each_celltype_v1_hgca_all_lineages_v1.h5ad`

Rerun needs those Visium sections, that downsampled reference (or a
rebuild from `Python_read.ipynb`), GPU, and the
`cell2location_cuda118_torch22` environment recorded in the notebook
metadata (Python 3.10, cell2location 0.1.4).
