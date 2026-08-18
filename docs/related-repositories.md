# Related software

This repository is the manuscript analysis companion. Tutorials for
using the atlas are in the
[HGCA CodeBook](https://github.com/HCA-integration/hca-gut-atlas-tutorial).

| Resource | Role | Pin |
|---|---|---|
| [MetaManager](https://github.com/CellDiscoveryNetwork/MetaManager) | Metadata ingestion and Supplementary Figure 2 validation | commit `9bed5fc73ef2` (`main`; no release tag) |
| [ARBOL](https://github.com/jo-m-lab/ARBOL) | Figure 2b taxonomy trees | install from that repo; frozen SVGs in `data/arbol/` |
| [patpy](https://github.com/connerlambden/patpy) | Patient-level embeddings (S16) | — |
| [scAtlasTb](https://github.com/HCA-integration/scAtlasTb) | Integration method benchmark (S3) | — |
| [ArchMap](https://www.archmap.bio/) | Query-to-reference mapping onto the four lineage scANVI models (Fig. 4b-style) | IDs after upload |

SCANVI LODO (Fig. 2c) and mapping stability (Fig. 4c, S14) are under
[`analyses/fig2c_fig4c_sfig14_scanvi/`](../analyses/fig2c_fig4c_sfig14_scanvi/)
(`lodo/` and `mapping_stability/`). Those scripts, configs, and omit
lists are in this tree so the paper screens can be rerun. They are not
demoed. Lymphoid mapping-stability took about 24 hours on a GPU; LODO
is as long or longer. Models are not stored here.

ArchMap is the no-code mapping front end for a *new* query dataset. It
does not reproduce LODO, mapping stability, SampleCLR, or patpy.

Atlas construction (QC, scVI), Xenium Segger/resolVI, TAURUS Milo,
SampleCLR attention, and segment GO are not in this tree.
