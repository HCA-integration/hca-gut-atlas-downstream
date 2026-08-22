#!/usr/bin/env python3
"""Compute stable naming depth from stroma jackknife soft predictions.

Primary tau=0.90. Also 0.70, 0.80, 1.00.
Uses leaf predictions across omit_* realizations (all seeds).
Separately summarizes full-reference seed variability.
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paths import FIGURES, HGCA_TAXONOMY, MANIFESTS, PREDICTIONS, TABLES  # noqa: E402

LOGGER = logging.getLogger("stable_naming_depth")
TAUS = (0.70, 0.80, 0.90, 1.00)
PRIMARY_TAU = 0.90
SHARED = ["Kinchen2018", "Lee2020", "Martin2019", "Uzzan2022", "Yu2021"]


def _hgca_paths() -> dict[str, list[str]]:
    tax = pd.read_csv(HGCA_TAXONOMY)
    leaf_col = "hgca_celltype_v1"
    levels = [
        "hgca_celltype_level1",
        "hgca_celltype_level2",
        "hgca_celltype_level3",
        "hgca_celltype_level4",
        "hgca_celltype_level5",
        leaf_col,
    ]
    paths = {}
    for _, row in tax.iterrows():
        leaf = str(row[leaf_col]).strip() if pd.notna(row[leaf_col]) else ""
        if not leaf or leaf.lower() == "nan":
            continue
        path = ["Stroma"]
        for col in levels:
            v = row[col]
            if pd.isna(v):
                continue
            s = str(v).strip()
            if not s or s.lower() == "nan":
                continue
            if not path or path[-1] != s:
                path.append(s)
        paths[leaf] = path
    return paths


def _pangi_paths() -> dict[str, list[str]]:
    hier = pd.read_csv(MANIFESTS / "pangi_stroma_taxonomy_edges.csv")
    paths = {}
    for _, row in hier.iterrows():
        leaf = str(row["level_3_annot"])
        path = ["Stroma", str(row["level_1_annot"]), str(row["level_2_annot"]), leaf]
        # collapse duplicate consecutive
        out = []
        for p in path:
            if not out or out[-1] != p:
                out.append(p)
        paths[leaf] = out
    return paths


def _load_preds(atlas: str, kind: str) -> pd.DataFrame:
    """kind: 'jackknife' (omit_*) or 'full' (full/seed*)."""
    base = PREDICTIONS / atlas / "stroma"
    frames = []
    for d in sorted(base.glob("*/seed*")):
        tag = d.parent.name
        if kind == "jackknife" and not tag.startswith("omit_"):
            continue
        if kind == "full" and tag != "full":
            continue
        pred = d / "predictions.parquet"
        if not pred.exists():
            continue
        df = pd.read_parquet(pred, columns=["query_cell_id", "leaf_prediction", "omitted_study", "model_seed"])
        df["realization"] = f"{tag}/seed{df['model_seed'].iloc[0]}"
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No predictions for {atlas} {kind}")
    return pd.concat(frames, ignore_index=True)


def _stable_depth_for_cell(leaves: list[str], paths: dict[str, list[str]], tau: float) -> tuple[float, str, int]:
    """Return (normalized_depth, deepest_node, depth_index)."""
    # path for each leaf; unknown leaves -> Stroma only
    cell_paths = [paths.get(leaf, ["Stroma", leaf]) for leaf in leaves]
    n = len(cell_paths)
    # max depth among paths
    max_len = max(len(p) for p in cell_paths)
    deepest_node = "Stroma"
    deepest_idx = 0
    for depth in range(max_len):
        # nodes at this depth
        nodes = []
        for p in cell_paths:
            nodes.append(p[depth] if depth < len(p) else p[-1])
        # support of majority node at this depth among realizations
        # For each candidate node appearing, support = fraction of paths that contain it at this depth or as ancestor
        # Spec: support(node) = fraction of realizations whose predicted leaf descends from that node
        # At depth d, check each unique node whether all realizations with that ancestor...
        # Efficient: for each realization path, the node at depth d (clipped) must match for support of that node
        counts = pd.Series(nodes).value_counts()
        best_node, best_n = counts.index[0], int(counts.iloc[0])
        support = best_n / n
        if support >= tau:
            deepest_node = str(best_node)
            deepest_idx = depth
        else:
            break
    # normalize: 0 = lineage root (Stroma), 1 = leaf (max depth among this cell's paths)
    denom = max(max_len - 1, 1)
    norm = deepest_idx / denom
    return float(norm), deepest_node, int(deepest_idx)


def compute_atlas(atlas: str, paths: dict[str, list[str]]) -> pd.DataFrame:
    jack = _load_preds(atlas, "jackknife")
    LOGGER.info("%s jackknife rows=%s realizations=%s", atlas, f"{len(jack):,}", jack.realization.nunique())
    # pivot to cell x realization leaf
    wide = jack.pivot_table(
        index="query_cell_id",
        columns="realization",
        values="leaf_prediction",
        aggfunc="first",
    )
    # drop cells missing any realization
    wide = wide.dropna(axis=0, how="any")
    LOGGER.info("%s cells with complete jackknife panel: %s", atlas, f"{len(wide):,}")

    records = []
    leaf_agree = []
    for cell_id, row in wide.iterrows():
        leaves = row.astype(str).tolist()
        leaf_agree.append(float(pd.Series(leaves).value_counts().iloc[0] / len(leaves)))
        for tau in TAUS:
            norm, node, idx = _stable_depth_for_cell(leaves, paths, tau)
            records.append(
                {
                    "query_cell_id": cell_id,
                    "atlas": atlas,
                    "tau": tau,
                    "stable_naming_depth_norm": norm,
                    "stable_node": node,
                    "stable_depth_index": idx,
                    "n_realizations": len(leaves),
                    "leaf_agreement": float(pd.Series(leaves).value_counts().iloc[0] / len(leaves)),
                    "n_unique_leaves": int(pd.Series(leaves).nunique()),
                }
            )
    out = pd.DataFrame(records)
    # attach metadata from one prediction file
    meta = pd.read_parquet(
        next((PREDICTIONS / atlas / "stroma" / "full").glob("seed*/predictions.parquet")),
        columns=["query_cell_id", "sample_id", "Patient", "Disease"],
    )
    out = out.merge(meta, on="query_cell_id", how="left")
    return out


def seed_variability(atlas: str) -> pd.DataFrame:
    full = _load_preds(atlas, "full")
    wide = full.pivot_table(index="query_cell_id", columns="realization", values="leaf_prediction", aggfunc="first")
    wide = wide.dropna(axis=0, how="any")
    rows = []
    for cell_id, row in wide.iterrows():
        leaves = row.astype(str).tolist()
        rows.append(
            {
                "query_cell_id": cell_id,
                "atlas": atlas,
                "seed_leaf_agreement": float(pd.Series(leaves).value_counts().iloc[0] / len(leaves)),
                "seed_n_unique_leaves": int(pd.Series(leaves).nunique()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    hgca_paths = _hgca_paths()
    pangi_paths = _pangi_paths()
    # ensure all predicted leaves have a path
    for atlas, paths in [("HGCA", hgca_paths), ("PanGI", pangi_paths)]:
        sample = pd.read_parquet(
            next((PREDICTIONS / atlas / "stroma" / "full").glob("seed*/predictions.parquet")),
            columns=["leaf_prediction"],
        )
        for leaf in sample.leaf_prediction.astype(str).unique():
            if leaf not in paths:
                paths[leaf] = ["Stroma", leaf]
                LOGGER.warning("%s missing taxonomy path for leaf %s; using Stroma->leaf", atlas, leaf)

    frames = []
    for atlas, paths in [("HGCA", hgca_paths), ("PanGI", pangi_paths)]:
        frames.append(compute_atlas(atlas, paths))
    depth = pd.concat(frames, ignore_index=True)
    depth.to_parquet(TABLES / "stable_naming_depth_cells.parquet", index=False)

    # summaries
    summary = (
        depth.groupby(["atlas", "tau"])
        .agg(
            n_cells=("query_cell_id", "count"),
            median_depth=("stable_naming_depth_norm", "median"),
            mean_depth=("stable_naming_depth_norm", "mean"),
            median_leaf_agreement=("leaf_agreement", "median"),
            mean_unique_leaves=("n_unique_leaves", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(TABLES / "stable_naming_depth_summary.csv", index=False)
    LOGGER.info("\n%s", summary.to_string(index=False))

    # disease split at primary tau
    primary = depth[depth.tau == PRIMARY_TAU]
    by_dis = (
        primary.groupby(["atlas", "Disease"])
        .agg(
            n=("query_cell_id", "count"),
            median_depth=("stable_naming_depth_norm", "median"),
            median_leaf_agreement=("leaf_agreement", "median"),
        )
        .reset_index()
    )
    by_dis.to_csv(TABLES / "stable_naming_depth_by_disease.csv", index=False)

    seed_frames = [seed_variability(a) for a in ("HGCA", "PanGI")]
    seed = pd.concat(seed_frames, ignore_index=True)
    seed.to_parquet(TABLES / "seed_leaf_agreement_cells.parquet", index=False)
    seed_sum = seed.groupby("atlas").agg(
        median_seed_leaf_agreement=("seed_leaf_agreement", "median"),
        mean_seed_unique=("seed_n_unique_leaves", "mean"),
    )
    seed_sum.to_csv(TABLES / "seed_leaf_agreement_summary.csv")
    LOGGER.info("Seed variability:\n%s", seed_sum.to_string())

    # simple distribution figure
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, atlas, color in zip(axes, ["HGCA", "PanGI"], ["#0072B2", "#009E73"]):
        sub = primary[primary.atlas == atlas]["stable_naming_depth_norm"]
        ax.hist(sub, bins=20, color=color, edgecolor="white", linewidth=0.3)
        ax.axvline(sub.median(), color="black", lw=1.0, ls="--")
        ax.set_title(f"{atlas}\nmedian={sub.median():.2f}", fontsize=9)
        ax.set_xlabel("Stable naming depth (?=0.90)\n0=lineage ... 1=leaf", fontsize=8)
        ax.set_xlim(0, 1)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("TAURUS stroma cells", fontsize=8)
    fig.suptitle("How specifically can each reference name cells across jackknives?", fontsize=10, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIGURES / f"stable_naming_depth_distribution.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_tau": PRIMARY_TAU,
        "taus": list(TAUS),
        "shared_studies": SHARED,
        "n_jackknife_realizations_expected": 15,
        "outputs": [
            "tables/stable_naming_depth_cells.parquet",
            "tables/stable_naming_depth_summary.csv",
            "tables/stable_naming_depth_by_disease.csv",
            "tables/seed_leaf_agreement_summary.csv",
            "figures/stable_naming_depth_distribution.pdf",
        ],
    }
    (TABLES / "stable_naming_depth_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
