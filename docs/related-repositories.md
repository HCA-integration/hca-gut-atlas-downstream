# Related software

This repository is the manuscript analysis companion. Tutorials for
using the atlas are in the
[HGCA CodeBook](https://github.com/HCA-integration/hca-gut-atlas-tutorial).

| Repo | what we used it for | Pin |
|---|---|---|
| [MetaManager](https://github.com/CellDiscoveryNetwork/MetaManager) | Metadata ingestion and Supplementary Figure 2 validation | `9bed5fc73ef2` |
| [ARBOL](https://github.com/jo-m-lab/ARBOL) | Figure 2b taxonomy trees | `9265029f29a7` |
| [patpy](https://github.com/lueckenlab/patpy) | Sample embeddings and pretreatment remission AUC (S16a,c) | `22d3d8000a57` |
| [scAtlasTb](https://github.com/HCA-integration/scAtlasTb) | Integration benchmarking (Supplementary Figure 3) | `e4cfba03b267` |
| [ArchMap](https://www.archmap.bio/) | Query-to-reference mapping onto the four lineage scANVI models (Fig. 4b-style) | web service; pin the uploaded HGCA models when they are public |

SCANVI LODO (Fig. 2c) and mapping stability (Fig. 4c, S14) are under
[`analyses/fig2c_fig4c_sfig14_scanvi/`](../analyses/fig2c_fig4c_sfig14_scanvi/)
(`lodo/` and `mapping_stability/`). Reminder, these scripts aren't
demoed as they are heavy-compute.
Lymphoid mapping-stability took about 24 hours on a GPU; LODO
is as long or longer. Models for LODO label transfer are not stored/provided.
You can use the code provided to recreate these.

ArchMap is the no-code mapping front end for a *new* query dataset.
