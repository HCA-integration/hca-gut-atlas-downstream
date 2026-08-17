#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate scanpy
python -u 01_build_sample_expr.py
python -u 02_score_test_modules.py
python -u 03_figures_and_report.py
echo "DONE → /Users/kylekimler/Projects/GCA/github_vignette_output/LIANA_rank_aggregate/follicle_associated_lr"
