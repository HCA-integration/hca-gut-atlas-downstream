# Supplementary Figure 9 — Visium cell2location

Collaborator notebooks (cell2location 0.1.4). Not runnable from the
bundled HGCA subset.

Input: Visium sections and a downsampled HGCA reference. GPU
recommended. Gene filters from Methods: `cell_count_cutoff=5`,
`cell_percentage_cutoff2=0.03`, `nonz_mean_cutoff=1.12`. Batch keys:
`sample_id`, `donor_id`, `assay`.

| Notebook | Role |
|---|---|
| `Notebooks/Python_read.ipynb` | Prepare inputs |
| `Notebooks/Python_C2L_hgca.ipynb` | Train reference |
| `Notebooks/Python_C2L_infer_hgca.ipynb` | Infer on Visium |
| `Notebooks/Python_visualize_c2l.ipynb` | Visualization |

The NMF notebook is omitted because stored outputs were large. Notebooks
still contain Sanger `/nfs/...` paths; replace those with local files
before rerunning.

Still needed: one public or authorized Visium section, CUDA / Python
versions from the paper run, and a collaborator authorship line.
