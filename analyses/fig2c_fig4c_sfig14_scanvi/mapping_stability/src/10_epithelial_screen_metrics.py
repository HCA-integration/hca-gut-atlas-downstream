#!/usr/bin/env python3
"""Primary epithelial screening metrics (same definitions as stroma)."""
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

LOGGER = logging.getLogger("epithelial_metrics")
TAUS = (0.70, 0.80, 0.90, 1.00)
PSEUDO = 0.5
SHARED = MANIFESTS / "epithelial_shared_studies_exact_name.csv"


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
        path = ["Epithelial"]
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
    hier = pd.read_csv(MANIFESTS / "pangi_epithelial_taxonomy_edges.csv")
    paths = {}
    for _, row in hier.iterrows():
        leaf = str(row["level_3_annot"])
        path = ["Epithelial", str(row["level_1_annot"]), str(row["level_2_annot"]), leaf]
        out = []
        for p in path:
            if not out or out[-1] != p:
                out.append(p)
        paths[leaf] = out
    return paths


def _stable_depth_for_cell(leaves, paths, tau):
    cell_paths = [paths.get(leaf, ["Epithelial", leaf]) for leaf in leaves]
    n = len(cell_paths)
    max_len = max(len(p) for p in cell_paths)
    deepest_node = "Epithelial"
    deepest_idx = 0
    for depth in range(max_len):
        nodes = [p[depth] if depth < len(p) else p[-1] for p in cell_paths]
        counts = pd.Series(nodes).value_counts()
        best_n = int(counts.iloc[0])
        if best_n / n >= tau:
            deepest_node = str(counts.index[0])
            deepest_idx = depth
        else:
            break
    denom = max(max_len - 1, 1)
    return deepest_idx / denom, deepest_node


def _load_preds(atlas: str, kind: str) -> pd.DataFrame:
    base = PREDICTIONS / atlas / "epithelial"
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
        df = pd.read_parquet(
            pred, columns=["query_cell_id", "leaf_prediction", "omitted_study", "model_seed", "sample_id", "Patient", "Disease"]
        )
        df["realization"] = f"{tag}/seed{df['model_seed'].iloc[0]}"
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No epithelial predictions for {atlas} {kind}")
    return pd.concat(frames, ignore_index=True)


def stable_naming(atlas: str, paths: dict) -> pd.DataFrame:
    jack = _load_preds(atlas, "jackknife")
    wide = jack.pivot_table(index="query_cell_id", columns="realization", values="leaf_prediction", aggfunc="first")
    wide = wide.dropna(axis=0, how="any")
    meta = jack.drop_duplicates("query_cell_id").set_index("query_cell_id")[
        ["sample_id", "Patient", "Disease"]
    ]
    rows = []
    for cell_id, row in wide.iterrows():
        leaves = row.astype(str).tolist()
        leaf_agree = float(pd.Series(leaves).value_counts().iloc[0] / len(leaves))
        n_unique = int(pd.Series(leaves).nunique())
        for tau in TAUS:
            norm, node = _stable_depth_for_cell(leaves, paths, tau)
            rows.append(
                {
                    "query_cell_id": cell_id,
                    "atlas": atlas,
                    "tau": tau,
                    "stable_naming_depth_norm": norm,
                    "stable_node": node,
                    "leaf_agreement": leaf_agree,
                    "n_unique_leaves": n_unique,
                    "n_realizations": len(leaves),
                }
            )
    out = pd.DataFrame(rows).merge(meta, left_on="query_cell_id", right_index=True, how="left")
    return out


def seed_control(atlas: str) -> dict:
    full = _load_preds(atlas, "full")
    wide = full.pivot_table(index="query_cell_id", columns="realization", values="leaf_prediction", aggfunc="first")
    wide = wide.dropna(axis=0, how="any")
    agrees = []
    uniques = []
    for _, row in wide.iterrows():
        leaves = row.astype(str).tolist()
        agrees.append(float(pd.Series(leaves).value_counts().iloc[0] / len(leaves)))
        uniques.append(int(pd.Series(leaves).nunique()))
    return {
        "atlas": atlas,
        "median_seed_leaf_agreement": float(np.median(agrees)),
        "mean_seed_unique": float(np.mean(uniques)),
        "n_full_seeds": int(wide.shape[1]),
        "n_cells": int(wide.shape[0]),
    }


def _clr(mat):
    x = mat + PSEUDO
    logx = np.log(x)
    return logx - logx.mean(axis=1, keepdims=True)


def aitchison(atlas: str) -> pd.DataFrame:
    base = PREDICTIONS / atlas / "epithelial"
    full0 = pd.read_parquet(base / "full" / "seed0" / "predictions.parquet")
    ref = pd.crosstab(full0["sample_id"], full0["leaf_prediction"])
    labels = list(ref.columns)
    samples = list(ref.index)
    ref_clr = _clr(ref.reindex(index=samples, columns=labels, fill_value=0).to_numpy(float))
    rows = []
    for d in sorted(base.glob("omit_*/seed*")):
        tab = pd.crosstab(
            pd.read_parquet(d / "predictions.parquet")["sample_id"],
            pd.read_parquet(d / "predictions.parquet")["leaf_prediction"],
        )
        mat = tab.reindex(index=samples, columns=labels, fill_value=0).to_numpy(float)
        dist = np.sqrt(((_clr(mat) - ref_clr) ** 2).sum(axis=1))
        omit = d.parent.name.replace("omit_", "")
        seed = int(d.name.replace("seed", ""))
        for sid, di in zip(samples, dist):
            rows.append(
                {
                    "atlas": atlas,
                    "sample_id": sid,
                    "omitted_study": omit,
                    "model_seed": seed,
                    "aitchison_to_full_seed0": float(di),
                }
            )
    out = pd.DataFrame(rows)
    meta = full0[["sample_id", "Disease", "Patient"]].drop_duplicates("sample_id")
    return out.merge(meta, on="sample_id", how="left")


def size_adjust(disp: pd.DataFrame) -> dict:
    impact = pd.read_csv(MANIFESTS / "epithelial_study_omission_impact.csv")
    impact = impact.rename(columns={"frac_lineage": "frac_lineage_removed"})
    shared = pd.read_csv(SHARED)["study"].astype(str).tolist()
    summ = (
        disp.groupby(["atlas", "omitted_study"])["aitchison_to_full_seed0"]
        .median()
        .reset_index()
        .rename(columns={"aitchison_to_full_seed0": "median_dist"})
    )
    m = summ.merge(
        impact[["atlas", "study", "frac_lineage_removed"]],
        left_on=["atlas", "omitted_study"],
        right_on=["atlas", "study"],
        how="left",
    )
    m = m[m.omitted_study.isin(shared)]
    # paired
    piv = m.pivot(index="omitted_study", columns="atlas", values="median_dist")
    paired = (piv["PanGI"] - piv["HGCA"]).dropna()
    # OLS mean_dist ~ frac + atlas
    X_frac = m["frac_lineage_removed"].to_numpy()
    y = m["median_dist"].to_numpy()
    atlas_p = (m["atlas"] == "PanGI").astype(float).to_numpy()
    A = np.column_stack([np.ones(len(y)), X_frac, atlas_p])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return {
        "ols_intercept": float(coef[0]),
        "coef_frac_removed": float(coef[1]),
        "coef_PanGI": float(coef[2]),
        "paired_median_PanGI_minus_HGCA": float(paired.median()) if len(paired) else None,
        "paired_by_study": paired.to_dict(),
        "per_study": m.to_dict(orient="records"),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    path_fns = {"HGCA": _hgca_paths, "PanGI": _pangi_paths}
    depth_frames = []
    seed_rows = []
    disp_frames = []
    for atlas in ("HGCA", "PanGI"):
        paths = path_fns[atlas]()
        depth = stable_naming(atlas, paths)
        depth.to_parquet(TABLES / f"epithelial_stable_naming_depth_cells_{atlas}.parquet", index=False)
        depth_frames.append(depth)
        seed_rows.append(seed_control(atlas))
        disp = aitchison(atlas)
        disp_frames.append(disp)

    depths = pd.concat(depth_frames, ignore_index=True)
    depths.to_parquet(TABLES / "epithelial_stable_naming_depth_cells.parquet", index=False)
    summary = (
        depths.groupby(["atlas", "tau"])
        .agg(
            n_cells=("query_cell_id", "nunique"),
            median_depth=("stable_naming_depth_norm", "median"),
            mean_depth=("stable_naming_depth_norm", "mean"),
            median_leaf_agreement=("leaf_agreement", "median"),
            mean_unique_leaves=("n_unique_leaves", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(TABLES / "epithelial_stable_naming_depth_summary.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(TABLES / "epithelial_seed_leaf_agreement_summary.csv", index=False)

    all_d = pd.concat(disp_frames, ignore_index=True)
    all_d.to_parquet(TABLES / "epithelial_sample_aitchison_displacement.parquet", index=False)
    dsum = (
        all_d.groupby(["atlas", "omitted_study"])
        .agg(median_dist=("aitchison_to_full_seed0", "median"), mean_dist=("aitchison_to_full_seed0", "mean"))
        .reset_index()
    )
    dsum.to_csv(TABLES / "epithelial_sample_aitchison_displacement_summary.csv", index=False)
    overall = all_d.groupby("atlas")["aitchison_to_full_seed0"].agg(["median", "mean"]).reset_index()
    overall.to_csv(TABLES / "epithelial_sample_aitchison_displacement_overall.csv", index=False)
    size = size_adjust(all_d)
    (TABLES / "epithelial_size_adjusted_displacement.json").write_text(json.dumps(size, indent=2) + "\n")

    man = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stable_naming_summary": summary.to_dict(orient="records"),
        "seed_controls": seed_rows,
        "aitchison_overall": overall.to_dict(orient="records"),
        "size_adjusted": size,
    }
    (TABLES / "epithelial_screen_metrics_manifest.json").write_text(json.dumps(man, indent=2, default=str) + "\n")
    LOGGER.info("\n%s", summary.to_string(index=False))
    LOGGER.info("\n%s", overall.to_string(index=False))
    LOGGER.info("size-adjusted coef_PanGI=%s paired_median_delta=%s", size["coef_PanGI"], size["paired_median_PanGI_minus_HGCA"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
