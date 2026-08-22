# Supplementary Figure 2 — metadata schema

Schema definition, contributor-sheet generation, and validation are in
[MetaManager](https://github.com/CellDiscoveryNetwork/MetaManager). There
is no tagged release; the paper run used commit `9bed5fc73ef2` on
`main`.

```bash
pip install git+https://github.com/CellDiscoveryNetwork/MetaManager.git@9bed5fc73ef2
```

This directory only re-renders the availability donut from the exported
counts in `data/sfig2/`.

```bash
Rscript analyses/sfig2_metamanager/src/render_sfig2_metadata_availability.R
```

Optional arguments: input TSV, then output stem. Requires dplyr, ggplot2,
ggforce, and ggnewscale.
