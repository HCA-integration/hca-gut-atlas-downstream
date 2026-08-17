# P06 — Visium cell2location (Supp. Fig. 9)

Collaborator notebooks (cell2location v0.1.4). These are the paper methods
source. They are **not** a laptop demo.

## System requirements

- cell2location 0.1.4, GPU recommended
- Gene filters from Methods: `cell_count_cutoff=5`,
  `cell_percentage_cutoff2=0.03`, `nonz_mean_cutoff=1.12`
- Batch keys: `sample_id`, plus `donor_id` and `assay`

## Notebooks in this repo

| File | Role |
|---|---|
| `Notebooks/Python_read.ipynb` | Prepare inputs |
| `Notebooks/Python_C2L_hgca.ipynb` | Train reference |
| `Notebooks/Python_C2L_infer_hgca.ipynb` | Infer on Visium |
| `Notebooks/Python_visualize_c2l.ipynb` | Visualization |

The NMF notebook is omitted here because stored outputs were ~14 MB. It
remains on the local publication tree until outputs are stripped.

Notebooks still contain Sanger `/nfs/...` paths. Replace those with local
Visium and a downsampled reference before rerunning. The bundled HGCA demo
slice is **not** a Visium section and is not a substitute.

## Still needed

- One public or authorized Visium section for a CPU stub
- CUDA / Python versions used on the paper run
- Collaborator authorship line
