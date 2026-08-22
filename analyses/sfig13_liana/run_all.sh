#!/usr/bin/env bash
# Rebuild Supplementary Figure 13 curated tables and panels.
# Requires CCC_EDGE_CSV (or LIANA_COMBINED_CSV) pointing at
# combined_lr_per_tissue_level_1.csv from a rank_aggregate run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT/logs" "$ROOT/out" "$ROOT/data"
python "$ROOT/src/00_extract_curated_tables.py" 2>&1 | tee "$ROOT/logs/00_extract.log"
Rscript "$ROOT/src/01_render_sfig13.R" 2>&1 | tee "$ROOT/logs/01_render.log"
echo "Done → $ROOT/out"
