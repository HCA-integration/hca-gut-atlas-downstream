#!/usr/bin/env bash
set -e -x

snakemake \
        --profile .profiles/icb \
        --configfile /lustre/groups/ml01/workspace/christopher.lance/hca_gut/hca-gut-atlas-extension/src/integration_benchmark/scAtlasTb_configs/hgca_integration_benchmark.yaml \
        --snakefile workflow/Snakefile \
        --use-conda \
        "$@"