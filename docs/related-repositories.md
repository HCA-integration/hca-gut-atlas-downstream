# Related HGCA v1 software

This repository is the downstream-analysis companion for the paper.
Other Code Availability entries are separate:

| Resource | Role |
|---|---|
| [MetaManager](https://github.com/CellDiscoveryNetwork/MetaManager) | Metadata ingestion and validation |
| [hca-gut-atlas-tutorial](https://github.com/HCA-integration/hca-gut-atlas-tutorial) | User-facing atlas tutorial |
| [ARBOL](https://github.com/jo-m-lab/ARBOL) | Taxonomy tree rendering |
| [patpy](https://github.com/connerlambden/patpy) | Patient-level embeddings / composition helpers |
| [scAtlasTb](https://github.com/HCA-integration/scAtlasTb) | Integration method benchmark |

Atlas construction (QC, scVI, scAtlasTb recipes), Xenium Segger/resolVI,
TAURUS Milo/SampleCLR, LIANA, and segment GO are still collaborator
packages. Visium cell2location notebooks are in
`analyses/visium_cell2location` but are not a laptop demo (Sanger `/nfs`
paths; GPU). Frozen SCANVI recipes for the mapping jackknife are in
`analyses/reference_uncertainty`.

A DOI for a tagged release (Zenodo) is required at publication, not at
initial review.
