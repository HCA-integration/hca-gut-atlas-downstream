"""
Per-``hgca_celltype_v1`` HGCA ``.h5ad`` shards: stable filenames, manifest lookup.

Used by ``scripts/data_preparation/export_hgca_celltype_h5ads.py`` and notebook 04.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


def slug_celltype_for_filename(label: str) -> str:
    """Filesystem-safe stem fragment from a cell-type label (no suffix)."""
    s = str(label).strip()
    if not s or s.lower() in ("nan", "none"):
        return "unknown_celltype"
    for ch in '/\\:*?"<>|':
        s = s.replace(ch, "_")
    s = "_".join(s.split())
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        return "unknown_celltype"
    return s[:220]


def allocate_stems_for_labels(labels: Iterable[str]) -> dict[str, str]:
    """
    Map each exact label string to a unique stem (no ``_hgca_v1`` suffix).
    Deterministic: sorted by string so collisions get ``__2``, ``__3``, …
    """
    used: set[str] = set()
    out: dict[str, str] = {}
    for lab in sorted(labels, key=lambda x: str(x)):
        s = str(lab)
        base = slug_celltype_for_filename(s)
        candidate = base
        i = 2
        while candidate in used:
            candidate = f"{base}__{i}"
            i += 1
        used.add(candidate)
        out[s] = candidate
    return out


def resolve_celltype_h5ad_path(out_dir: Path, label: str) -> Path:
    """
    Resolve path to ``{stem}_hgca_v1.h5ad`` for an exact ``hgca_celltype_v1`` label.

    Prefers ``hgca_celltype_manifest.json`` (written by the export script) so
    collision-safe stems match. Falls back to ``{slug(label)}_hgca_v1.h5ad``.
    """
    out_dir = Path(out_dir)
    manifest_path = out_dir / "hgca_celltype_manifest.json"
    key = str(label)
    if manifest_path.is_file():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        fn = (data.get("by_label") or {}).get(key)
        if fn:
            return out_dir / fn
    stem = slug_celltype_for_filename(key)
    return out_dir / f"{stem}_hgca_v1.h5ad"
