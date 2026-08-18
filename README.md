# Human Gut Cell Atlas v1 analyses

Analysis and figure-generation code for *The Human Gut Cell Atlas v1.0*.


This repository is for reproducing the manuscript analyses. Tutorials for
using the atlas as a resource are maintained separately in the
[HGCA CodeBook](LINK GOES HERE).

## Analyses

[analyses/README.md](analyses/README.md) maps manuscript figures and
tables to directories and scripts.

## Environment

So far the demos here were tested by Kyle (with AI help) 
on macOS 15 with: 

Python 3.12 and the packages in `requirements.txt` 
(anndata 0.12.3, pandas 2.3.3, numpy 2.3.4,
scipy 1.16.2, statsmodels 0.14.6, matplotlib 3.10.7, h5py 3.15.1).

Atlas construction, mapping stability / LODO and scANVI mapping, 
and possibly cell2location and Xenium analysis can be sped up with GPUs.
lymphoid mapping-stability took about 24 hours on a GPU, 
and LODO is as long or longer. Other lineages took ~12 (epi), 6 (stroma), 2 hours. 


Query mapping onto the four lineage scANVI models can be done without GPUs (or any code)
at [ArchMap](https://www.archmap.bio/) UPDATE THIS LINK ONCE UPLOAD IS DONE. 

To reproduce figures, install the requirements!
```bash
git clone https://github.com/HCA-integration/hca-gut-atlas-downstream.git
cd hca-gut-atlas-downstream
python -m pip install -r requirements.txt
```

Install time should be no more than a few minutes if Python is already present.
A conda specification is provided in `environment.yml`.

## Verification

`data/demo/hgca_all_lineages_v1_demo.h5ad` is a 3,185-cell subset of the
all-cells object (94 cell types). Might be useful to check that
representative scripts execute on your machine. 

```bash
python src/run_demo.py
```

Runtime = ~30 seconds. Output is written under
`data/demo/expected/` (gitignored). A shorter check is
`python src/smoke_demo_slice.py` (94 ileum biopsy-versus-resection
Mann–Whitney tests; 17 samples with at least three germinal-centre B cells).

To run a script on the full atlas, pass the object path or set
`HGCA_H5AD`. Lineage objects use `HGCA_OBJECTS`. Column names must match
the atlas (`hgca_celltype_v1`, `sample_id`, `donor_id`, `dataset_id`,
`tissue_level_1`, and the covariates listed in each analysis README).

## License

MIT. See [LICENSE](LICENSE).
