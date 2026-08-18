#!/usr/bin/env python3
"""CPU smoke for the taxonomy path-distance metric used in Fig. 4 / S14.

Does not train SCANVI. DEMO MODE: software check, not manuscript figures.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import pandas as pd

LEVELS = [
    "hgca_celltype_level1",
    "hgca_celltype_level2",
    "hgca_celltype_level3",
    "hgca_celltype_level4",
    "hgca_celltype_level5",
    "hgca_celltype_v1",
]
LZ = "GC B Light Zone (GC B LZ)"
DZ = "GC B Dark Zone (GC B DZ)"
FARM = "Follicle Associated Resident Macrophages"


def _clean(label: str) -> str:
    return " ".join(str(label).replace("\n", " ").split())


def _path(row: pd.Series) -> tuple[str, ...]:
    out: list[str] = []
    for col in LEVELS:
        if col not in row.index or pd.isna(row[col]):
            continue
        value = str(row[col]).strip()
        if not value or value.lower() == "nan":
            continue
        if not out or out[-1] != value:
            out.append(value)
    return tuple(out)


def _distance(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    shared = 0
    for left, right in zip(a, b):
        if left != right:
            break
        shared += 1
    return (len(a) - shared) + (len(b) - shared)


def main() -> int:
    print("DEMO MODE: results are for software checking, not manuscript figures.")
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=repo / "data" / "demo" / "GCA_taxonomy_2026_CAP.csv",
    )
    parser.add_argument(
        "--h5ad",
        type=Path,
        default=repo / "data" / "demo" / "hgca_all_lineages_v1_demo.h5ad",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=repo / "data" / "demo" / "expected" / "mapping",
    )
    args = parser.parse_args()

    tax = pd.read_csv(args.taxonomy)
    paths = {}
    for _, row in tax.iterrows():
        leaf = _clean(row["hgca_celltype_v1"]) if pd.notna(row["hgca_celltype_v1"]) else ""
        if not leaf or leaf.lower() == "nan":
            continue
        path = _path(row)
        if path:
            paths[leaf] = path

    obs = ad.read_h5ad(args.h5ad, backed="r").obs
    demo_types = {_clean(x) for x in obs["hgca_celltype_v1"].astype(str)}
    missing = sorted(demo_types - set(paths))
    if missing:
        raise SystemExit(f"Demo cell types missing from taxonomy: {missing[:8]}")

    pairs = [
        (LZ, LZ),
        (LZ, DZ),
        (LZ, FARM),
    ]
    rows = []
    for a, b in pairs:
        if a not in paths or b not in paths:
            raise SystemExit(f"Taxonomy missing required labels: {a!r}, {b!r}")
        dist = _distance(paths[a], paths[b])
        rows.append(
            {
                "label_a": a,
                "label_b": b,
                "path_a": "|".join(paths[a]),
                "path_b": "|".join(paths[b]),
                "path_distance": dist,
            }
        )
    table = pd.DataFrame(rows)
    if int(table.loc[table["label_a"].eq(LZ) & table["label_b"].eq(LZ), "path_distance"].iloc[0]) != 0:
        raise SystemExit("Self path-distance must be 0")
    sibling = int(
        table.loc[table["label_a"].eq(LZ) & table["label_b"].eq(DZ), "path_distance"].iloc[0]
    )
    cross = int(
        table.loc[table["label_a"].eq(LZ) & table["label_b"].eq(FARM), "path_distance"].iloc[0]
    )
    if sibling <= 0:
        raise SystemExit("LZ vs DZ should be a positive sibling distance")
    if cross <= sibling:
        raise SystemExit("Cross-lineage distance should exceed LZ vs DZ")

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "taxonomy_path_distance_smoke.csv"
    table.to_csv(out, index=False)
    print(
        f"taxonomy_leaves={len(paths)} demo_types={len(demo_types)} "
        f"lz_dz={sibling} lz_farm={cross}"
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
