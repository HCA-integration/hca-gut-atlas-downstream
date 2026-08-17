"""Repo-relative paths for the HGCA v1 demo and optional full-atlas rebuilds.

Public default is the bundled demo slice. Point at the full atlas with:

    export HGCA_H5AD=/path/to/hgca_all_lineages_v1.h5ad
    export HGCA_OBJECTS=/path/to/lineage-h5ads
    export HGCA_TAXONOMY=/path/to/GCA_taxonomy_2026_CAP.csv
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEMO_H5AD = REPO / "data" / "demo" / "hgca_all_lineages_v1_demo.h5ad"
DEMO_TAXONOMY = REPO / "data" / "demo" / "GCA_taxonomy_2026_CAP.csv"
EXPECTED = REPO / "data" / "demo" / "expected"
LINEAGE_NAMES = ("epithelial", "lymphoid", "myeloid", "stroma")


def atlas_h5ad(cli_path: Path | None = None) -> Path:
    if cli_path is not None:
        return Path(cli_path)
    env = os.environ.get("HGCA_H5AD")
    if env:
        return Path(env)
    return DEMO_H5AD


def taxonomy_csv(cli_path: Path | None = None) -> Path:
    if cli_path is not None:
        return Path(cli_path)
    env = os.environ.get("HGCA_TAXONOMY")
    if env:
        return Path(env)
    return DEMO_TAXONOMY


def lineage_h5ads() -> dict[str, Path]:
    root = os.environ.get("HGCA_OBJECTS")
    if not root:
        return {}
    base = Path(root)
    return {name: base / f"{name}.h5ad" for name in LINEAGE_NAMES}


def is_demo(path: Path) -> bool:
    return "demo" in Path(path).name
