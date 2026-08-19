# Supplementary Figure 3 — integration benchmark via scAtlasTb

Central to reproducing the integration benchmark is the config file `sfig3_scAtlasTb/src/hgca_integration_benchmark.yaml` serving as input to scAtlasTb together with the benchmarking dataset

Environments were installed using the install scripts of scAtlasTB, visit https://scatlastb.readthedocs.io for detailed descriptions on the set up.

Ensure to adjust the cluster resources to your available cluster and the correct paths to the input data set.

The dataset for benchmarking is a subset of the full HGCA object. The benchmarking subset can be derived running the notebook `sfig3_scAtlasTb/src/integration_prep_atlasTb.ipynb`.

The benchmark itself was run using the script `sfig3_scAtlasTb/src/run_hgca.ipynb`.


