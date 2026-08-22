"""Paths for atlas mapping-stability (shared-study omission) analyses.

The paper calls this Atlas Mapping Stability / stable mapping resolution.
Outputs are not written into the git tree: set MAPPING_STABILITY_MODELS and
MAPPING_STABILITY_PREDICTIONS, or they default under this directory.
"""
from __future__ import annotations

import os
from pathlib import Path

STABILITY = Path(__file__).resolve().parents[1]
SCANVI_DIR = STABILITY.parent
LODO_ROOT = SCANVI_DIR / "lodo"
REPO = SCANVI_DIR.parents[1]

MANIFESTS = SCANVI_DIR / "manifests"
CONFIGS = SCANVI_DIR / "configs"
RMB = LODO_ROOT

DATA = Path(os.environ.get("MAPPING_STABILITY_DATA", STABILITY / "data"))
MODELS = Path(os.environ.get("MAPPING_STABILITY_MODELS", STABILITY / "models"))
PREDICTIONS = Path(
    os.environ.get("MAPPING_STABILITY_PREDICTIONS", STABILITY / "predictions")
)
TABLES = Path(os.environ.get("MAPPING_STABILITY_TABLES", STABILITY / "tables"))
FIGURES = Path(os.environ.get("MAPPING_STABILITY_FIGURES", STABILITY / "figures"))
REPORTS = STABILITY / "reports"
LOGS = STABILITY / "logs"

_objects = Path(os.environ["HGCA_OBJECTS"]) if os.environ.get("HGCA_OBJECTS") else None
HGCA_H5ADS = {
    "epithelial": (
        _objects / "epithelial.h5ad" if _objects else DATA / "hgca_epithelial.h5ad"
    ),
    "lymphoid": (
        _objects / "lymphoid.h5ad" if _objects else DATA / "hgca_lymphoid.h5ad"
    ),
    "myeloid": _objects / "myeloid.h5ad" if _objects else DATA / "hgca_myeloid.h5ad",
    "stroma": _objects / "stroma.h5ad" if _objects else DATA / "hgca_stroma.h5ad",
}
HGCA_STROMA_H5AD = HGCA_H5ADS["stroma"]

PANGI_H5AD = Path(
    os.environ.get("PANGI_H5AD", str(DATA / "pangi_healthy_full.h5ad"))
)
TAURUS_H5AD = Path(os.environ.get("TAURUS_H5AD", str(DATA / "taurus_query.h5ad")))
HGCA_TAXONOMY = Path(
    os.environ.get(
        "HGCA_TAXONOMY",
        str(REPO / "data" / "demo" / "GCA_taxonomy_2026_CAP.csv"),
    )
)
RECIPE = CONFIGS / "stroma_scanvi_recipe_frozen.json"
ANNOT_PARQUET = Path(os.environ.get("HGCA_ANNOT_PARQUET", ""))
