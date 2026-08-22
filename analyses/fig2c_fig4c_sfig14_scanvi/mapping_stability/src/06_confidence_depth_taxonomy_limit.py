#!/usr/bin/env python3
"""Confidence depth, healthy calibration, overconfidence, confidence x stable depth.

Uses EXISTING stroma soft predictions (full/seed0 primary).
Does not retrain. Writes tables + updates evidence for report 01.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paths import ANNOT_PARQUET, FIGURES, HGCA_TAXONOMY, MANIFESTS, PREDICTIONS, TABLES  # noqa: E402

LOGGER = logging.getLogger("confidence_depth")
TAUS = (0.80, 0.90, 0.95)
PRIMARY_TAU = 0.90


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
        out = []
        for p in path:
            if not out or out[-1] != p:
                out.append(p)
        paths[leaf] = path if False else out
    return paths


def _ancestor_mass(soft_row: pd.Series, paths: dict[str, list[str]]) -> dict[str, float]:
    """Aggregate posterior mass to every node on leaf paths."""
    mass: dict[str, float] = {}
    for leaf, p in soft_row.items():
        leaf = str(leaf)
        path = paths.get(leaf, ["Stroma", leaf])
        for node in path:
            mass[node] = mass.get(node, 0.0) + float(p)
    return mass


def confidence_depth_cell(soft_row: pd.Series, paths: dict[str, list[str]], tau: float) -> tuple[float, str]:
    mass = _ancestor_mass(soft_row, paths)
    # walk depth by considering all leaf paths; deepest node with mass>=tau
    # Build candidate depths from union of paths weighted by mass
    # Spec: deepest taxonomy node containing >= tau posterior probability mass
    # Among nodes with mass>=tau, pick the deepest; if ties, prefer higher mass then name
    # Depth of a node = max path index among paths containing that node
    node_depth: dict[str, int] = {}
    for leaf, path in paths.items():
        for i, node in enumerate(path):
            node_depth[node] = max(node_depth.get(node, 0), i)
    # also include unknown leaves from soft
    for leaf in soft_row.index.astype(str):
        if leaf not in paths:
            node_depth.setdefault("Stroma", 0)
            node_depth[leaf] = 1
    eligible = [(node, node_depth.get(node, 0), m) for node, m in mass.items() if m >= tau]
    if not eligible:
        return 0.0, "Stroma"
    eligible.sort(key=lambda x: (x[1], x[2]), reverse=True)
    best_node, best_d, _ = eligible[0]
    max_d = max(node_depth.values()) if node_depth else 1
    denom = max(max_d, 1)
    # normalize: 0 = lineage root depth 0, 1 = deepest possible in this taxonomy
    # Prefer cell-specific max among nodes that receive mass
    cell_max = max((node_depth.get(n, 0) for n in mass), default=1)
    denom = max(cell_max, 1)
    return float(best_d / denom), str(best_node)


def load_full_seed0(atlas: str):
    base = PREDICTIONS / atlas / "stroma" / "full" / "seed0"
    pred = pd.read_parquet(base / "predictions.parquet")
    post = pd.read_parquet(base / "posterior.parquet")
    return pred, post


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    disp = pd.read_csv(TABLES / "healthy_displacement_stroma.csv")
    # sample-level displacement (canonical column name)
    if "mahalanobis" not in disp.columns and "displacement_mahalanobis" in disp.columns:
        disp = disp.rename(columns={"displacement_mahalanobis": "mahalanobis"})
    disp_s = disp.set_index("sample_id")

    # stable naming depth if present
    stable_path = TABLES / "stable_naming_depth_cells.parquet"
    stable = pd.read_parquet(stable_path) if stable_path.exists() else None

    ann = pd.read_parquet(ANNOT_PARQUET)
    stroma_ann = ann[ann.assigned_lineage.astype(str) == "stroma"][
        ["barcode", "sample_id", "Patient", "Disease"]
    ].copy()
    stroma_ann["query_cell_id"] = stroma_ann["barcode"].astype(str)

    cell_rows = []
    sample_rows = []
    overconf_rows = []
    conf_stable_rows = []

    for atlas, path_fn in (("HGCA", _hgca_paths), ("PanGI", _pangi_paths)):
        paths = path_fn()
        pred, post = load_full_seed0(atlas)
        label_cols = [c for c in post.columns if c != "query_cell_id"]
        post = post.set_index("query_cell_id")
        pred = pred.set_index("query_cell_id")
        # align
        common = pred.index.intersection(post.index)
        pred = pred.loc[common]
        post = post.loc[common, label_cols]

        # confidence depth for primary + sensitivity
        depths = {tau: [] for tau in TAUS}
        deepest_nodes = {tau: [] for tau in TAUS}
        # vectorized-ish loop in chunks
        soft_np = post.to_numpy(dtype=float)
        labels = list(post.columns)
        for i, cell_id in enumerate(post.index):
            soft_row = pd.Series(soft_np[i], index=labels)
            for tau in TAUS:
                d, node = confidence_depth_cell(soft_row, paths, tau)
                depths[tau].append(d)
                deepest_nodes[tau].append(node)

        cell = pred[["sample_id", "Patient", "Disease", "max_posterior", "normalized_entropy", "leaf_prediction"]].copy()
        cell["atlas"] = atlas
        for tau in TAUS:
            cell[f"confidence_depth_tau{tau}"] = depths[tau]
            cell[f"confidence_node_tau{tau}"] = deepest_nodes[tau]
        cell = cell.reset_index()

        # Healthy calibration: percentiles within Healthy cells
        healthy_mask = cell["Disease"].astype(str) == "Healthy"
        if healthy_mask.sum() < 50:
            LOGGER.warning("%s Healthy cells for calibration: %s", atlas, healthy_mask.sum())
        for col, out_col, invert in (
            ("normalized_entropy", "entropy_pctile_healthy", False),
            ("max_posterior", "maxpost_uncert_pctile_healthy", True),  # high maxpost = low uncertainty
            (f"confidence_depth_tau{PRIMARY_TAU}", "confdepth_pctile_healthy", False),
        ):
            href = cell.loc[healthy_mask, col].to_numpy()
            # percentile of each value within healthy reference (higher = more unusual vs healthy)
            # For max_posterior: convert to uncertainty = 1-maxpost so high = uncertain
            if invert:
                href_u = 1.0 - href
                vals_u = 1.0 - cell[col].to_numpy()
                pct = np.array([stats.percentileofscore(href_u, v, kind="mean") for v in vals_u])
            else:
                pct = np.array([stats.percentileofscore(href, v, kind="mean") for v in cell[col].to_numpy()])
            cell[out_col] = pct / 100.0

        cell.to_parquet(TABLES / f"confidence_depth_cells_{atlas}.parquet", index=False)
        cell_rows.append(cell)

        # sample x family aggregates: family = top-level under Stroma from predicted leaf path
        def family_of(leaf: str) -> str:
            p = paths.get(str(leaf), ["Stroma", str(leaf)])
            return p[1] if len(p) > 1 else "Stroma"

        cell["family"] = cell["leaf_prediction"].map(family_of)
        # merge displacement
        cell = cell.merge(
            disp_s[["mahalanobis", "Disease"]].rename(columns={"Disease": "Disease_disp"}),
            left_on="sample_id",
            right_index=True,
            how="left",
        )

        samp = (
            cell.groupby(["sample_id", "Patient", "Disease", "family"], dropna=False)
            .agg(
                n_cells=("query_cell_id", "count"),
                mean_entropy_pct=("entropy_pctile_healthy", "mean"),
                mean_maxpost_uncert_pct=("maxpost_uncert_pctile_healthy", "mean"),
                mean_confdepth=("confidence_depth_tau0.9", "mean"),
                mean_confdepth_pct=("confdepth_pctile_healthy", "mean"),
                mahalanobis=("mahalanobis", "first"),
            )
            .reset_index()
        )
        samp = samp[samp.n_cells >= 10]
        samp["atlas"] = atlas
        sample_rows.append(samp)

        # Overconfidence: freeze thresholds on Healthy only
        # HIGH bio displacement: >95th pct of Healthy sample mahalanobis (use LOO if available)
        loo = TABLES / "healthy_displacement_loo_stroma.csv"
        if loo.exists():
            hdisp = pd.read_csv(loo)
            loo_col = (
                "mahalanobis_loo"
                if "mahalanobis_loo" in hdisp.columns
                else "displacement_mahalanobis_loo"
            )
            h_scores = hdisp.loc[hdisp.Disease.astype(str) == "Healthy", loo_col].dropna()
            if len(h_scores) == 0:
                h_scores = disp.loc[disp.Disease.astype(str) == "Healthy", "mahalanobis"]
        else:
            h_scores = disp.loc[disp.Disease.astype(str) == "Healthy", "mahalanobis"]
        thr_disp = float(np.nanpercentile(h_scores, 95))
        # high-confidence fine: confidence depth within top 20% of healthy (depth >= healthy 80th pct)
        h_depth = cell.loc[healthy_mask, f"confidence_depth_tau{PRIMARY_TAU}"]
        thr_depth = float(np.nanpercentile(h_depth, 80))
        h_maxpost = cell.loc[healthy_mask, "max_posterior"]
        thr_maxpost = float(np.nanpercentile(h_maxpost, 80))

        # cell-level with sample displacement
        cell_d = cell.copy()
        high_disp = cell_d["mahalanobis"].to_numpy() > thr_disp
        high_conf = (cell_d[f"confidence_depth_tau{PRIMARY_TAU}"].to_numpy() >= thr_depth) & (
            cell_d["max_posterior"].to_numpy() >= thr_maxpost
        )
        disease_mask = cell_d["Disease"].astype(str).isin(["CD", "UC"])
        denom = int((high_disp & disease_mask).sum())
        numer = int((high_disp & disease_mask & high_conf).sum())
        overconf_rows.append(
            {
                "atlas": atlas,
                "thr_disp_p95_healthy": thr_disp,
                "thr_confdepth_p80_healthy": thr_depth,
                "thr_maxpost_p80_healthy": thr_maxpost,
                "n_high_disp_disease_cells": denom,
                "n_high_disp_high_conf": numer,
                "overconfidence_rate": (numer / denom) if denom else np.nan,
            }
        )
        # threshold sensitivity
        for p_disp, p_conf in ((90, 80), (95, 80), (95, 90), (99, 80)):
            td = float(np.nanpercentile(h_scores, p_disp))
            tdepth = float(np.nanpercentile(h_depth, p_conf))
            tmpost = float(np.nanpercentile(h_maxpost, p_conf))
            hd = cell_d["mahalanobis"].to_numpy() > td
            hc = (cell_d[f"confidence_depth_tau{PRIMARY_TAU}"].to_numpy() >= tdepth) & (
                cell_d["max_posterior"].to_numpy() >= tmpost
            )
            dnm = int((hd & disease_mask).sum())
            num = int((hd & disease_mask & hc).sum())
            overconf_rows.append(
                {
                    "atlas": atlas,
                    "thr_disp_p95_healthy": td,
                    "thr_confdepth_p80_healthy": tdepth,
                    "thr_maxpost_p80_healthy": tmpost,
                    "n_high_disp_disease_cells": dnm,
                    "n_high_disp_high_conf": num,
                    "overconfidence_rate": (num / dnm) if dnm else np.nan,
                    "sensitivity": f"disp_p{p_disp}_conf_p{p_conf}",
                }
            )

        # Confidence x stable naming depth
        if stable is not None:
            st = stable[stable.atlas == atlas]
            # expect columns query_cell_id, depth_tau0.9 or similar
            depth_cols = [c for c in st.columns if "depth" in c.lower() and "0.9" in c]
            if not depth_cols:
                depth_cols = [c for c in st.columns if c.startswith("stable") or "norm" in c]
            # read schema
            LOGGER.info("stable columns %s: %s", atlas, st.columns.tolist()[:20])
            # try common names
            sdepth_col = None
            for c in ("stable_depth_tau0.9", "depth_tau0.9", "norm_depth_0.9", "stable_naming_depth"):
                if c in st.columns:
                    sdepth_col = c
                    break
            if sdepth_col is None:
                if "tau" in st.columns and "stable_naming_depth_norm" in st.columns:
                    st9 = st[np.isclose(st.tau.to_numpy(dtype=float), 0.9)][
                        ["query_cell_id", "stable_naming_depth_norm"]
                    ].rename(columns={"stable_naming_depth_norm": "stable_depth"})
                elif "tau" in st.columns and "normalized_depth" in st.columns:
                    st9 = st[np.isclose(st.tau.to_numpy(dtype=float), 0.9)][
                        ["query_cell_id", "normalized_depth"]
                    ].rename(columns={"normalized_depth": "stable_depth"})
                else:
                    st9 = None
            else:
                st9 = st[["query_cell_id", sdepth_col]].rename(columns={sdepth_col: "stable_depth"})
            if st9 is not None:
                m = cell.merge(st9, on="query_cell_id", how="inner")
                # dichotomize at 0.75
                hi_conf = m[f"confidence_depth_tau{PRIMARY_TAU}"] >= 0.75
                hi_stab = m["stable_depth"] >= 0.75
                m["class"] = np.select(
                    [
                        hi_conf & hi_stab,
                        (~hi_conf) & hi_stab,
                        (~hi_conf) & (~hi_stab),
                        hi_conf & (~hi_stab),
                    ],
                    [
                        "high_conf_high_stab",
                        "low_conf_high_stab",
                        "low_conf_low_stab",
                        "high_conf_low_stab",
                    ],
                    default="other",
                )
                m["atlas"] = atlas
                m["family"] = m["leaf_prediction"].map(family_of)
                conf_stable_rows.append(m)
                summary = (
                    m.groupby(["atlas", "Disease", "class"])
                    .size()
                    .rename("n")
                    .reset_index()
                )
                summary["frac"] = summary.groupby(["atlas", "Disease"])["n"].transform(lambda x: x / x.sum())
                summary.to_csv(TABLES / f"confidence_vs_stability_{atlas}.csv", index=False)

        # Spearman at sample x family
        for metric in ("mean_entropy_pct", "mean_maxpost_uncert_pct", "mean_confdepth_pct"):
            sub = samp.dropna(subset=["mahalanobis", metric])
            if len(sub) < 10:
                continue
            r, p = stats.spearmanr(sub["mahalanobis"], sub[metric])
            LOGGER.info("%s %s vs mahalanobis: spearman=%.3f p=%.3g n=%d", atlas, metric, r, p, len(sub))

    cells_all = pd.concat(cell_rows, ignore_index=True)
    cells_all.to_parquet(TABLES / "confidence_depth_cells_stroma.parquet", index=False)
    samp_all = pd.concat(sample_rows, ignore_index=True)
    samp_all.to_csv(TABLES / "confidence_displacement_sample_family.csv", index=False)
    pd.DataFrame(overconf_rows).to_csv(TABLES / "overconfidence_stroma.csv", index=False)

    # donor bootstrap CIs for spearman
    boot_rows = []
    rng = np.random.default_rng(0)
    for atlas in ("HGCA", "PanGI"):
        sub = samp_all[samp_all.atlas == atlas].dropna(subset=["mahalanobis", "mean_entropy_pct"])
        patients = sub["Patient"].astype(str).unique()
        for metric in ("mean_entropy_pct", "mean_maxpost_uncert_pct", "mean_confdepth_pct"):
            rs = []
            for _ in range(1000):
                draw = rng.choice(patients, size=len(patients), replace=True)
                parts = [sub[sub.Patient.astype(str) == p] for p in draw]
                b = pd.concat(parts, ignore_index=True)
                if b[metric].nunique() < 2 or b["mahalanobis"].nunique() < 2:
                    continue
                r, _ = stats.spearmanr(b["mahalanobis"], b[metric])
                if np.isfinite(r):
                    rs.append(r)
            if rs:
                boot_rows.append(
                    {
                        "atlas": atlas,
                        "metric": metric,
                        "spearman_point": float(stats.spearmanr(sub["mahalanobis"], sub[metric])[0]),
                        "boot_median": float(np.median(rs)),
                        "boot_lo": float(np.percentile(rs, 2.5)),
                        "boot_hi": float(np.percentile(rs, 97.5)),
                        "n_boot": len(rs),
                        "n_sample_family": len(sub),
                        "n_patients": len(patients),
                    }
                )
    pd.DataFrame(boot_rows).to_csv(TABLES / "confidence_displacement_spearman_bootstrap.csv", index=False)

    if conf_stable_rows:
        cs = pd.concat(conf_stable_rows, ignore_index=True)
        cs.to_parquet(TABLES / "confidence_vs_stability_cells.parquet", index=False)
        fail = cs[cs["class"] == "high_conf_low_stab"]
        fail_sum = (
            fail.groupby(["atlas", "Disease", "family"])
            .size()
            .rename("n")
            .reset_index()
            .sort_values("n", ascending=False)
        )
        fail_sum.to_csv(TABLES / "high_conf_low_stability_summary.csv", index=False)

    man = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_tau": PRIMARY_TAU,
        "taus": list(TAUS),
        "n_cells": {a: int((cells_all.atlas == a).sum()) for a in ("HGCA", "PanGI")},
        "overconfidence": overconf_rows[:2],
        "spearman_bootstrap": boot_rows,
    }
    (TABLES / "confidence_depth_manifest.json").write_text(json.dumps(man, indent=2, default=str) + "\n")
    LOGGER.info("Done confidence depth taxonomy-limit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
