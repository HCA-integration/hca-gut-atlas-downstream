"""
Map external atlas coarse lineages (e.g. PanGI ``level_1_annot``) to HGCA benchmark lineages.

PanGI Healthy ``level_1_annot`` to HGCA slug (see ``configs/pangi_*.yaml`` ``lineage_subset``):

- Myeloid → myeloid (1:1)
- Epithelial → epithelial (1:1)
- T and NK cells + B and B plasma → lymphoid (combined in ``pangi_lymphoid``)
- Mesenchymal + Endothelial → stroma (combined in ``pangi_stroma``)
- Neural → neural (PanGI-only; no HGCA neural LODO)
"""

from __future__ import annotations

from typing import Dict, Optional

# PanGI Healthy (and similar) level_1_annot -> HGCA-style lineage slug
PANGI_LEVEL1_TO_BENCHMARK_LINEAGE: Dict[str, str] = {
    "Myeloid": "myeloid",
    "T and NK cells": "lymphoid",
    "B and B plasma": "lymphoid",
    "Epithelial": "epithelial",
    "Mesenchymal": "stroma",
    "Endothelial": "stroma",
    "Neural": "neural",
}


def benchmark_lineage_from_pangi_level1(level1: Optional[str]) -> Optional[str]:
    """Return benchmark lineage slug for a PanGI ``level_1_annot`` value, or None if unknown."""
    if level1 is None or (isinstance(level1, float) and str(level1) == "nan"):
        return None
    s = str(level1).strip()
    return PANGI_LEVEL1_TO_BENCHMARK_LINEAGE.get(s)
