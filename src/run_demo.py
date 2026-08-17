#!/usr/bin/env python3
"""Laptop demo for Nature software review (Figures 1–3 tables).

Runs the wired analysis scripts against the bundled demo slice and writes
only under data/demo/expected/.

DEMO MODE: results are for software checking, not manuscript figures.
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
    print("DEMO MODE: results are for software checking, not manuscript figures.")
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
            str(ANALYSES / "fig_sampling_depth_radial" / "src" / "recompute_clr_tables.py"),
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
                / "fig_sampling_depth_radial"
                / "rare_cell_defensibility"
                / "src"
                / "compute_defensibility.py"
            ),
            "--clr-long",
            str(EXPECTED / "clr" / "clr_long.csv"),
            "--outdir",
            str(EXPECTED / "follicle"),
        ]
    )

    elapsed = time.time() - t0
    print(f"\nDemo finished in {elapsed:.0f}s. Outputs under {EXPECTED}")
    print("Compare donor-age, CLR, and follicle tables to the checked-in expected/ copies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
