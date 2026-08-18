# Human Gut Cell Atlas v1 analyses

Analysis and figure-generation code for *The Human Gut Cell Atlas v1.0*.
Code is still being consolidated during manuscript preparation; not every
paper analysis is in the tree yet.

This repository is for reproducing the manuscript analyses. Tutorials for
using the atlas as a resource are maintained separately in the
[HGCA CodeBook](https://github.com/HCA-integration/hca-gut-atlas-tutorial).

## Analyses

[analyses/README.md](analyses/README.md) maps manuscript figures and
tables to directories and scripts.

## Environment

Tested on macOS 15 with Python 3.12 and the packages in
`requirements.txt` (anndata 0.12.3, pandas 2.3.3, numpy 2.3.4,
scipy 1.16.2, statsmodels 0.14.6, matplotlib 3.10.7, h5py 3.15.1).
Some figure scripts also use R with ggplot2. Atlas construction,
scVI, cell2location, and Xenium require a GPU and live in other
repositories ([docs/related-repositories.md](docs/related-repositories.md)).

```bash
git clone https://github.com/HCA-integration/hca-gut-atlas-downstream.git
cd hca-gut-atlas-downstream
python -m pip install -r requirements.txt
```

Typical install time is a few minutes if Python is already present.
A conda specification is in `environment.yml`.

## Verification

`data/demo/hgca_all_lineages_v1_demo.h5ad` is a 3,185-cell subset of the
all-cells object (94 cell types). It is only for checking that
representative scripts execute. Values from this subset are not the
manuscript results.

```bash
python src/run_demo.py
```

Runtime is about 30 seconds. Output is written under
`data/demo/expected/` (gitignored). A shorter check is
`python src/smoke_demo_slice.py` (94 ileum biopsy-versus-resection
Mann–Whitney tests; 17 samples with at least three germinal-centre B cells).

To run a script on the full atlas, pass the object path or set
`HGCA_H5AD`. Lineage objects use `HGCA_OBJECTS`. Column names must match
the atlas (`hgca_celltype_v1`, `sample_id`, `donor_id`, `dataset_id`,
`tissue_level_1`, and the covariates listed in each analysis README).

## License

MIT. See [LICENSE](LICENSE).
