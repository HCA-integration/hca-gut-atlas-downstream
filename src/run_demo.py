#!/usr/bin/env python3
"""Run representative analysis scripts on the bundled subset.

Writes under data/demo/expected/. Results are not manuscript values.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "data" / "demo" / "hgca_all_lineages_v1_demo.h5ad"
TAX = REPO / "data" / "demo" / "GCA_taxonomy_2026_CAP.csv"
EXPECTED = REPO / "data" / "demo" / "expected"
ANALYSES = REPO / "analyses"


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    print("Verification subset: results are not manuscript values.")
    if not DEMO.is_file():
        raise SystemExit(f"Missing demo slice: {DEMO}")
    t0 = time.time()

    run([sys.executable, str(REPO / "src" / "smoke_demo_slice.py")])
    run(
        [
            sys.executable,
            str(ANALYSES / "fig1_donor_age" / "src" / "build_donor_age_counts.py"),
            "--h5ad",
            str(DEMO),
            "--outdir",
            str(EXPECTED / "fig1"),
        ]
    )
    run(
        [
            sys.executable,
            str(ANALYSES / "fig2_label_set" / "src" / "build_fig2_atlas_evidence.py"),
            "--demo",
            "--metadata",
            str(DEMO),
            "--taxonomy",
            str(TAX),
            "--figure-dir",
            str(EXPECTED / "fig2"),
        ]
    )
    run(
        [
            sys.executable,
            str(ANALYSES / "fig3_clr_contrasts" / "src" / "recompute_clr_tables.py"),
            "--all-cells",
            str(DEMO),
            "--outdir",
            str(EXPECTED / "clr"),
        ]
    )
    run(
        [
            sys.executable,
            str(
                ANALYSES
                / "fig3_clr_contrasts"
                / "fig3i_sfig12_follicle"
                / "src"
                / "compute_defensibility.py"
            ),
            "--clr-long",
            str(EXPECTED / "clr" / "clr_long.csv"),
            "--outdir",
            str(EXPECTED / "follicle"),
        ]
    )
    run(
        [
            sys.executable,
            str(
                ANALYSES
                / "fig3_clr_contrasts"
                / "fig3c_sfig7_covariates"
                / "src"
                / "08_theils_u_heatmap.py"
            ),
            "--h5ad",
            str(DEMO),
            "--outdir",
            str(EXPECTED / "theils_u"),
        ]
    )
    run(
        [
            sys.executable,
            str(ANALYSES / "ed_table1_datasets" / "src" / "build_extended_data_table1.py"),
            "--h5ad",
            str(DEMO),
            "--outdir",
            str(EXPECTED / "tables"),
        ]
    )
    run(
        [
            sys.executable,
            str(
                ANALYSES
                / "sfig11_compositional_correlations"
                / "src"
                / "compute_compositional_correlations.py"
            ),
            "--clr-long",
            str(EXPECTED / "clr" / "clr_long.csv"),
            "--lineage-map",
            str(EXPECTED / "clr" / "celltype_lineage_map.csv"),
            "--outdir",
            str(EXPECTED / "correlations"),
        ]
    )
    run(
        [
            sys.executable,
            str(ANALYSES / "fig2c_fig4c_sfig14_scanvi" / "src" / "smoke_taxonomy_paths.py"),
            "--taxonomy",
            str(TAX),
            "--h5ad",
            str(DEMO),
            "--outdir",
            str(EXPECTED / "mapping"),
        ]
    )

    elapsed = time.time() - t0
    print(f"\nFinished in {elapsed:.0f}s. Outputs under {EXPECTED} (not manuscript values).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
