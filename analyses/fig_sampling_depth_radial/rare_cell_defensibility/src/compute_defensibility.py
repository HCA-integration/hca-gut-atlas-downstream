#!/usr/bin/env python3
"""Rare-cell defensibility analyses for featured sampling-depth claims.

Reads ../fig_sampling_depth_radial/data/clr_long.csv (and lineage map) and
writes compact tables under ../data/ for Nature-style panels:

  1. Leave-one-dataset-out (LODO) full-thickness ΔCLR
  2. Dataset-level forest effects (within-study vs between-study)
  3. Donor-aggregated contrasts (one weight per donor)
  4. Cell-count / min-depth sensitivity
  5. Negative-control cell types (same pipeline)
  6. Follicle / TLS niche capture probabilities across contexts

Featured cell types (main text): Tfr, medullary sinus endothelial, FARM,
tuft progenitors, BEST4 enterocytes, BEST4 colonocytes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO_ROOT = ROOT.parents[2]
PARENT_DATA = ROOT.parent / "data"
OUT = ROOT / "data"
EXPECTED_FOLLICLE = REPO_ROOT / "data" / "demo" / "expected" / "follicle"

FEATURED = [
    "CD4 Tfr",
    "Medullary Sinus Endothelial",
    "Follicle Associated Resident Macrophages",
    "Tuft Progenitors",
    "BEST4 Enterocytes",
    "BEST4 Colonocytes",
]
NEGATIVE = [
    "Lamina propria Fibroblasts (S1)",
    "Goblet Cells",
    "Villus Tip Enterocytes",
    "CD8 IEL",
    "Homeostatic Macrophages",
    "Paneth Cells",
]
SHORT = {
    "CD4 Tfr": "Tfr",
    "Medullary Sinus Endothelial": "Med. sinus endo.",
    "Follicle Associated Resident Macrophages": "FARM",
    "Tuft Progenitors": "Tuft progenitors",
    "BEST4 Enterocytes": "BEST4 enterocytes",
    "BEST4 Colonocytes": "BEST4 colonocytes",
    "Lamina propria Fibroblasts (S1)": "S1 fibroblasts",
    "Goblet Cells": "Goblet cells",
    "Villus Tip Enterocytes": "Villus-tip enterocytes",
    "CD8 IEL": "CD8 IEL",
    "Homeostatic Macrophages": "Homeostatic mac.",
    "Paneth Cells": "Paneth cells",
}

# Core follicle / TLS markers for niche capture (coordinated presence).
# Require a B-cell GC arm + at least one stromal organizer or Tfh/Tfr.
FOLLICLE_CORE_B = {
    "GC B Light Zone (GC B LZ)",
    "GC B Dark Zone (GC B DZ)",
}
FOLLICLE_SUPPORT = {
    "Follicular Dendritic Cells (fDC)",
    "CD4 Tfh",
    "CD4 Tfr",
    "Follicle Associated Resident Macrophages",
    "Fibroblastic Reticular Cells (FRC)",
    "Mesenchymal Lymphoid Tissue Organizer Cells (mLTo Cells)",
}

MIN_CELLS_DETECT = 3  # cells of a type to call "detected" in a sample
MIN_SUPPORT_TYPES = 1  # support types alongside ≥1 GC B arm
CANONICAL_SEGMENTS = ["duodenum", "jejunum", "ileum", "colon"]


def _mw_delta(a: np.ndarray, b: np.ndarray) -> dict:
    """Mann-Whitney rest(A) vs full(B); return delta = mean_B - mean_A."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    out = {
        "n_A": int(len(a)),
        "n_B": int(len(b)),
        "mean_A": float(np.mean(a)) if len(a) else np.nan,
        "mean_B": float(np.mean(b)) if len(b) else np.nan,
        "delta_CLR": np.nan,
        "p_value": np.nan,
    }
    if len(a) < 2 or len(b) < 2:
        if len(a) and len(b):
            out["delta_CLR"] = float(np.mean(b) - np.mean(a))
        return out
    out["delta_CLR"] = float(np.mean(b) - np.mean(a))
    try:
        out["p_value"] = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        out["p_value"] = np.nan
    return out


def _bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 800, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        deltas[i] = bb.mean() - aa.mean()
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return float(lo), float(hi)


def load_clr(path: Path | None = None) -> pd.DataFrame:
    d = pd.read_csv(path or PARENT_DATA / "clr_long.csv")
    r = d["radial_tissue_term"].astype(str).str.lower()
    d["depth_arm"] = pd.Series(
        np.where(
            r.eq("epi_lp_musc"),
            "full",
            np.where(r.isin(["epi", "epi_lp", "lp", "wm"]), "rest", "unknown"),
        ),
        index=d.index,
    )
    d.loc[d["depth_arm"].eq("unknown"), "depth_arm"] = np.nan
    d["short_name"] = d["celltype"].map(SHORT).fillna(d["celltype"])
    d["role"] = np.where(
        d["celltype"].isin(FEATURED),
        "featured",
        np.where(d["celltype"].isin(NEGATIVE), "negative_control", "other"),
    )
    # sample-level total cells across all cell types present in clr_long
    tot = d.groupby("sample_id", as_index=False)["n_cells"].sum().rename(
        columns={"n_cells": "sample_total_cells"}
    )
    d = d.merge(tot, on="sample_id", how="left")
    return d


def contrast_table(
    d: pd.DataFrame,
    celltypes: list[str],
    *,
    unit: str = "sample",
    exclude_datasets: set[str] | None = None,
    min_sample_cells: int | None = None,
    min_ct_cells: int | None = None,
    keep_datasets: set[str] | None = None,
) -> pd.DataFrame:
    """Compute rest vs full ΔCLR for each cell type under optional filters."""
    rows = []
    x = d[d["depth_arm"].isin(["rest", "full"])].copy()
    if exclude_datasets:
        x = x[~x["dataset_id"].isin(exclude_datasets)]
    if keep_datasets is not None:
        x = x[x["dataset_id"].isin(keep_datasets)]
    if min_sample_cells is not None:
        x = x[x["sample_total_cells"] >= min_sample_cells]
    if min_ct_cells is not None:
        # keep samples where the cell type reaches the min OR is absent (CLR still defined)
        # Sensitivity: drop samples with 0 < n_cells < min (ambiguous low counts)
        pass

    for ct in celltypes:
        y = x[x["celltype"] == ct].copy()
        if min_ct_cells is not None:
            # Exclude samples with sparse positive detection of this type
            y = y[~((y["n_cells"] > 0) & (y["n_cells"] < min_ct_cells))]
        if y.empty:
            continue
        if unit == "donor":
            y = (
                y.groupby(["donor_id", "depth_arm", "dataset_id", "tissue_level_1"], as_index=False)
                .agg(clr=("clr", "mean"), n_cells=("n_cells", "sum"), n_samples=("sample_id", "nunique"))
            )
            id_col = "donor_id"
        else:
            id_col = "sample_id"

        a = y.loc[y["depth_arm"] == "rest", "clr"].to_numpy()
        b = y.loc[y["depth_arm"] == "full", "clr"].to_numpy()
        stats = _mw_delta(a, b)
        lo, hi = _bootstrap_ci(a, b)
        rows.append(
            {
                "celltype": ct,
                "short_name": SHORT.get(ct, ct),
                "role": "featured" if ct in FEATURED else "negative_control",
                "unit": unit,
                "exclude_datasets": ",".join(sorted(exclude_datasets)) if exclude_datasets else "",
                "min_sample_cells": min_sample_cells if min_sample_cells is not None else "",
                "min_ct_cells": min_ct_cells if min_ct_cells is not None else "",
                "n_ids_A": stats["n_A"],
                "n_ids_B": stats["n_B"],
                "mean_CLR_A": stats["mean_A"],
                "mean_CLR_B": stats["mean_B"],
                "delta_CLR": stats["delta_CLR"],
                "ci_lo": lo,
                "ci_hi": hi,
                "p_value": stats["p_value"],
                "n_datasets": int(y["dataset_id"].nunique()),
                "id_col": id_col,
            }
        )
    return pd.DataFrame(rows)


def run_lodo(d: pd.DataFrame, celltypes: list[str]) -> pd.DataFrame:
    """Leave-one-dataset-out for datasets that contribute to either arm."""
    x = d[d["depth_arm"].isin(["rest", "full"]) & d["celltype"].isin(celltypes)]
    datasets = sorted(x["dataset_id"].dropna().unique())
    rows = []
    # full estimate
    base = contrast_table(d, celltypes, unit="sample")
    base["left_out"] = "none (all data)"
    base["analysis"] = "all"
    rows.append(base)
    for ds in datasets:
        sub = contrast_table(d, celltypes, unit="sample", exclude_datasets={ds})
        if sub.empty:
            continue
        sub["left_out"] = ds
        sub["analysis"] = "lodo"
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True)
    # stability vs all-data estimate
    ref = out[out["analysis"] == "all"][["celltype", "delta_CLR"]].rename(
        columns={"delta_CLR": "delta_all"}
    )
    out = out.merge(ref, on="celltype", how="left")
    out["delta_shift"] = out["delta_CLR"] - out["delta_all"]
    out["same_sign"] = np.sign(out["delta_CLR"]) == np.sign(out["delta_all"])
    return out


def run_forest(d: pd.DataFrame, celltypes: list[str]) -> pd.DataFrame:
    """Per-dataset effects; label within-study vs between-study."""
    rows = []
    x = d[d["depth_arm"].isin(["rest", "full"]) & d["celltype"].isin(celltypes)].copy()
    for ct in celltypes:
        y = x[x["celltype"] == ct]
        # overall (between-study nested)
        a = y.loc[y["depth_arm"] == "rest", "clr"].to_numpy()
        b = y.loc[y["depth_arm"] == "full", "clr"].to_numpy()
        stats = _mw_delta(a, b)
        lo, hi = _bootstrap_ci(a, b)
        # how many datasets have both arms?
        g = y.groupby(["dataset_id", "depth_arm"])["sample_id"].nunique().unstack(fill_value=0)
        for col in ("rest", "full"):
            if col not in g.columns:
                g[col] = 0
        n_both = int(((g["rest"] > 0) & (g["full"] > 0)).sum())
        overall_label = "within-study" if n_both >= 1 and len(g) == 1 else (
            "mixed" if n_both >= 1 else "between-study"
        )
        rows.append(
            {
                "celltype": ct,
                "short_name": SHORT.get(ct, ct),
                "role": "featured" if ct in FEATURED else "negative_control",
                "dataset_id": "ALL (pooled)",
                "inference": overall_label,
                "n_A": stats["n_A"],
                "n_B": stats["n_B"],
                "delta_CLR": stats["delta_CLR"],
                "ci_lo": lo,
                "ci_hi": hi,
                "p_value": stats["p_value"],
                "n_datasets_both_arms": n_both,
                "n_datasets_total": int(len(g)),
                "is_pooled": True,
            }
        )
        for ds, sub in y.groupby("dataset_id"):
            na = int(sub.loc[sub["depth_arm"] == "rest", "sample_id"].nunique())
            nb = int(sub.loc[sub["depth_arm"] == "full", "sample_id"].nunique())
            if na > 0 and nb > 0:
                aa = sub.loc[sub["depth_arm"] == "rest", "clr"].to_numpy()
                bb = sub.loc[sub["depth_arm"] == "full", "clr"].to_numpy()
                st = _mw_delta(aa, bb)
                clo, chi = _bootstrap_ci(aa, bb, seed=hash(str(ds)) % 10_000)
                rows.append(
                    {
                        "celltype": ct,
                        "short_name": SHORT.get(ct, ct),
                        "role": "featured" if ct in FEATURED else "negative_control",
                        "dataset_id": ds,
                        "inference": "within-study",
                        "n_A": st["n_A"],
                        "n_B": st["n_B"],
                        "delta_CLR": st["delta_CLR"],
                        "ci_lo": clo,
                        "ci_hi": chi,
                        "p_value": st["p_value"],
                        "n_datasets_both_arms": 1,
                        "n_datasets_total": 1,
                        "is_pooled": False,
                    }
                )
            else:
                # one-arm only: report arm mean, no within-study delta
                arm = "full" if nb > 0 else "rest"
                mean_clr = float(sub.loc[sub["depth_arm"] == arm, "clr"].mean())
                rows.append(
                    {
                        "celltype": ct,
                        "short_name": SHORT.get(ct, ct),
                        "role": "featured" if ct in FEATURED else "negative_control",
                        "dataset_id": ds,
                        "inference": "one-arm (no within-study contrast)",
                        "n_A": na,
                        "n_B": nb,
                        "delta_CLR": np.nan,
                        "ci_lo": np.nan,
                        "ci_hi": np.nan,
                        "p_value": np.nan,
                        "arm_mean_CLR": mean_clr,
                        "arm_present": arm,
                        "n_datasets_both_arms": 0,
                        "n_datasets_total": 1,
                        "is_pooled": False,
                    }
                )
    return pd.DataFrame(rows)


def run_donor_agg(d: pd.DataFrame, celltypes: list[str]) -> pd.DataFrame:
    sample = contrast_table(d, celltypes, unit="sample")
    sample["aggregation"] = "sample"
    donor = contrast_table(d, celltypes, unit="donor")
    donor["aggregation"] = "donor"
    out = pd.concat([sample, donor], ignore_index=True)
    # donor overweight diagnostic: samples per donor among analyzed
    diag_rows = []
    x = d[d["depth_arm"].isin(["rest", "full"]) & d["celltype"].isin(celltypes)]
    for ct in celltypes:
        y = x[x["celltype"] == ct]
        ns = y.groupby("donor_id")["sample_id"].nunique()
        diag_rows.append(
            {
                "celltype": ct,
                "short_name": SHORT.get(ct, ct),
                "n_donors": int(ns.shape[0]),
                "n_samples": int(y["sample_id"].nunique()),
                "n_multi_sample_donors": int((ns > 1).sum()),
                "max_samples_per_donor": int(ns.max()) if len(ns) else 0,
                "mean_samples_per_donor": float(ns.mean()) if len(ns) else np.nan,
                "frac_samples_from_multi": float(
                    y[y["donor_id"].isin(ns[ns > 1].index)]["sample_id"].nunique()
                    / max(y["sample_id"].nunique(), 1)
                ),
            }
        )
    diag = pd.DataFrame(diag_rows)
    return out, diag


def run_sensitivity(d: pd.DataFrame, celltypes: list[str]) -> pd.DataFrame:
    rows = []
    # min sample total cells
    for thr in [0, 500, 1000, 2000, 5000]:
        tab = contrast_table(d, celltypes, unit="sample", min_sample_cells=thr if thr else None)
        tab["filter"] = f"min_sample_cells≥{thr}" if thr else "none"
        tab["filter_family"] = "sample_depth"
        tab["threshold"] = thr
        rows.append(tab)
    # min cell-type count (exclude sparse positives)
    for thr in [0, 1, 3, 5, 10]:
        tab = contrast_table(
            d, celltypes, unit="sample", min_ct_cells=thr if thr else None
        )
        tab["filter"] = f"drop_sparse_0<n<{thr}" if thr else "none"
        tab["filter_family"] = "ct_min_cells"
        tab["threshold"] = thr
        rows.append(tab)
    # donor aggregation as sensitivity
    tab = contrast_table(d, celltypes, unit="donor")
    tab["filter"] = "donor_mean"
    tab["filter_family"] = "aggregation"
    tab["threshold"] = -1
    rows.append(tab)
    return pd.concat(rows, ignore_index=True)


def run_niche_capture(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Infer follicle/TLS capture from coordinated cell-type detection.

    Rule (primary):
      detected_GC = any of FOLLICLE_CORE_B with n_cells ≥ MIN_CELLS_DETECT
      support     = count of FOLLICLE_SUPPORT with n_cells ≥ MIN_CELLS_DETECT
      niche_captured = detected_GC AND support ≥ MIN_SUPPORT_TYPES

    Also report a stricter GC+fDC rule and a looser any-follicle-flag rule.
    """
    meta_cols = [
        "sample_id",
        "donor_id",
        "dataset_id",
        "tissue_level_1",
        "radial_tissue_term",
        "sample_collection_method",
        "depth_arm",
        "sample_total_cells",
    ]
    # wide presence for follicle types
    niche_types = sorted(FOLLICLE_CORE_B | FOLLICLE_SUPPORT)
    sub = d[d["celltype"].isin(niche_types)].copy()
    # one row per sample metadata
    meta = (
        d.drop_duplicates("sample_id")[meta_cols]
        .copy()
    )
    # presence matrix
    piv = sub.pivot_table(
        index="sample_id", columns="celltype", values="n_cells", aggfunc="sum", fill_value=0
    )
    for ct in niche_types:
        if ct not in piv.columns:
            piv[ct] = 0
    piv = piv[niche_types]
    det = piv >= MIN_CELLS_DETECT

    gc = det[list(FOLLICLE_CORE_B)].any(axis=1)
    support_n = det[list(FOLLICLE_SUPPORT)].sum(axis=1)
    fdc = det.get("Follicular Dendritic Cells (fDC)", pd.Series(False, index=det.index))
    tfh = det.get("CD4 Tfh", pd.Series(False, index=det.index))
    tfr = det.get("CD4 Tfr", pd.Series(False, index=det.index))

    sample = meta.set_index("sample_id")
    sample = sample.join(piv.add_prefix("n_"), how="left").fillna(0)
    sample["detect_GC"] = gc.reindex(sample.index, fill_value=False).astype(bool)
    sample["n_support"] = support_n.reindex(sample.index, fill_value=0).astype(int)
    sample["detect_fDC"] = fdc.reindex(sample.index, fill_value=False).astype(bool)
    sample["detect_Tfh"] = tfh.reindex(sample.index, fill_value=False).astype(bool)
    sample["detect_Tfr"] = tfr.reindex(sample.index, fill_value=False).astype(bool)
    sample["niche_primary"] = sample["detect_GC"] & (sample["n_support"] >= MIN_SUPPORT_TYPES)
    sample["niche_strict"] = sample["detect_GC"] & sample["detect_fDC"]
    sample["niche_loose"] = sample["detect_GC"] | (
        (sample["detect_Tfh"] | sample["detect_Tfr"]) & sample["detect_fDC"]
    )
    sample = sample.reset_index()

    # rates by context
    def _rate(df: pd.DataFrame, group_cols: list[str], flag: str) -> pd.DataFrame:
        g = (
            df.groupby(group_cols, dropna=False)
            .agg(n_samples=("sample_id", "nunique"), n_pos=(flag, "sum"))
            .reset_index()
        )
        g["capture_rate"] = g["n_pos"] / g["n_samples"].clip(lower=1)
        g["rule"] = flag
        return g

    sample["segment"] = sample["tissue_level_1"].astype(str).str.lower()
    sample["radial_layer"] = sample["radial_tissue_term"].astype(str).str.upper()
    # Gut-wall analyses only: drop mesentery / accessory / other non-canonical sites
    sample = sample[sample["segment"].isin(CANONICAL_SEGMENTS)].copy()
    sample["context"] = pd.Series(
        np.where(
            sample["depth_arm"].eq("full"),
            "full_thickness",
            np.where(
                sample["depth_arm"].eq("rest"),
                "not_full_thickness",
                "unknown",
            ),
        ),
        index=sample.index,
    )
    sample["collection"] = sample["sample_collection_method"].astype(str)

    rate_parts = []
    for flag in ("niche_primary", "niche_strict", "niche_loose"):
        rate_parts.append(_rate(sample, ["context"], flag).assign(strata="depth"))
        rate_parts.append(
            _rate(sample, ["radial_layer"], flag).assign(strata="radial_layer")
        )
        rate_parts.append(_rate(sample, ["segment"], flag).assign(strata="segment"))
        rate_parts.append(
            _rate(sample, ["dataset_id"], flag).assign(strata="dataset")
        )
        rate_parts.append(
            _rate(sample, ["segment", "context"], flag).assign(strata="segment×depth")
        )
        rate_parts.append(
            _rate(sample, ["segment", "radial_layer"], flag).assign(
                strata="segment×radial_layer"
            )
        )
        rate_parts.append(
            _rate(sample, ["collection", "context"], flag).assign(strata="collection×depth")
        )
        rate_parts.append(
            _rate(sample, ["collection", "radial_layer"], flag).assign(
                strata="collection×radial_layer"
            )
        )
        rate_parts.append(
            _rate(sample, ["dataset_id", "radial_layer"], flag).assign(
                strata="dataset×radial_layer"
            )
        )
    rates = pd.concat(rate_parts, ignore_index=True)

    # per-marker detection rates by context (for interpretability)
    marker_rows = []
    for ct in niche_types:
        det_ct = (piv[ct] >= MIN_CELLS_DETECT).rename("detected")
        tmp = sample[
            ["sample_id", "context", "segment", "dataset_id", "radial_layer"]
        ].merge(det_ct.reset_index(), on="sample_id", how="left")
        for strata, cols in [
            ("depth", ["context"]),
            ("radial_layer", ["radial_layer"]),
            ("segment", ["segment"]),
            ("segment×depth", ["segment", "context"]),
            ("segment×radial_layer", ["segment", "radial_layer"]),
        ]:
            g = (
                tmp.groupby(cols, dropna=False)
                .agg(n_samples=("sample_id", "nunique"), n_pos=("detected", "sum"))
                .reset_index()
            )
            g["detection_rate"] = g["n_pos"] / g["n_samples"].clip(lower=1)
            g["celltype"] = ct
            g["short_name"] = SHORT.get(ct, ct)
            g["strata"] = strata
            marker_rows.append(g)
    markers = pd.concat(marker_rows, ignore_index=True)

    # sample-level export (compact)
    export_cols = [
        "sample_id",
        "donor_id",
        "dataset_id",
        "segment",
        "radial_layer",
        "context",
        "collection",
        "sample_total_cells",
        "detect_GC",
        "n_support",
        "detect_fDC",
        "detect_Tfh",
        "detect_Tfr",
        "niche_primary",
        "niche_strict",
        "niche_loose",
    ]
    return sample[export_cols], rates, markers


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clr-long", type=Path, default=PARENT_DATA / "clr_long.csv")
    parser.add_argument("--outdir", type=Path, default=OUT)
    args = parser.parse_args()
    outdir = args.outdir
    if "demo" in str(args.clr_long) and outdir == ROOT / "data":
        outdir = EXPECTED_FOLLICLE
        print(f"Demo input: writing to {outdir}")
        print("DEMO MODE: results are for software checking, not manuscript figures.")
    OUT = outdir
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_clr(args.clr_long)
    all_cts = FEATURED + NEGATIVE

    print("Computing LODO…")
    lodo = run_lodo(d, all_cts)
    lodo.to_csv(OUT / "lodo_full_thickness.csv", index=False)

    print("Computing forest…")
    forest = run_forest(d, all_cts)
    forest.to_csv(OUT / "forest_full_thickness.csv", index=False)

    print("Computing donor aggregation…")
    donor_tab, donor_diag = run_donor_agg(d, all_cts)
    donor_tab.to_csv(OUT / "donor_vs_sample_contrasts.csv", index=False)
    donor_diag.to_csv(OUT / "donor_multiplicity_diag.csv", index=False)

    print("Computing sensitivity…")
    sens = run_sensitivity(d, all_cts)
    sens.to_csv(OUT / "cellcount_sensitivity.csv", index=False)

    print("Computing niche capture…")
    niche_samples, niche_rates, niche_markers = run_niche_capture(d)
    niche_samples.to_csv(OUT / "niche_capture_samples.csv", index=False)
    niche_rates.to_csv(OUT / "niche_capture_rates.csv", index=False)
    niche_markers.to_csv(OUT / "niche_marker_detection_rates.csv", index=False)

    # Study-level rates + Wilson CIs by radial layer (± segment)
    def _study_rate_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        rows = []
        for keys, g in df.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            n = int(g["sample_id"].nunique())
            pos = int(g["niche_primary"].sum())
            rate = pos / n if n else np.nan
            lo, hi = (
                proportion_confint(pos, n, alpha=0.05, method="wilson")
                if n
                else (np.nan, np.nan)
            )
            row = dict(zip(group_cols, keys))
            row.update(
                {
                    "n_samples": n,
                    "n_pos": pos,
                    "capture_rate": rate,
                    "ci_lo": float(lo),
                    "ci_hi": float(hi),
                    "collection": ",".join(
                        sorted(g["collection"].dropna().astype(str).unique())
                    ),
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    study_rad = _study_rate_table(niche_samples, ["radial_layer", "dataset_id"])
    study_rad.to_csv(OUT / "niche_capture_by_dataset_radial.csv", index=False)
    study_seg = _study_rate_table(
        niche_samples, ["segment", "radial_layer", "dataset_id"]
    )
    study_seg.to_csv(OUT / "niche_capture_by_dataset_segment_radial.csv", index=False)

    layer_summ = []
    for layer in ["EPI", "EPI_LP", "LP", "EPI_LP_MUSC", "WM"]:
        sub = study_rad[
            (study_rad["radial_layer"] == layer) & (study_rad["n_samples"] >= 3)
        ]
        allsub = study_rad[study_rad["radial_layer"] == layer]
        pn, pp = int(allsub["n_samples"].sum()), int(allsub["n_pos"].sum())
        plo, phi = (
            proportion_confint(pp, pn, method="wilson") if pn else (np.nan, np.nan)
        )
        layer_summ.append(
            {
                "radial_layer": layer,
                "n_studies": int(allsub["dataset_id"].nunique()),
                "n_studies_ge3": int(len(sub)),
                "n_samples": pn,
                "pooled_rate": pp / pn if pn else np.nan,
                "pooled_ci_lo": float(plo),
                "pooled_ci_hi": float(phi),
                "median_study_rate": float(sub["capture_rate"].median())
                if len(sub)
                else np.nan,
                "single_study": int(allsub["dataset_id"].nunique() == 1),
            }
        )
    pd.DataFrame(layer_summ).to_csv(
        OUT / "niche_capture_radial_study_summary.csv", index=False
    )

    # compact summary for featured + negatives (primary contrasts)
    summary = contrast_table(d, all_cts, unit="sample")
    summary_d = contrast_table(d, all_cts, unit="donor")
    summary["aggregation"] = "sample"
    summary_d["aggregation"] = "donor"
    # FDR within role
    for role, idx in summary.groupby("role").groups.items():
        p = summary.loc[idx, "p_value"].to_numpy()
        ok = np.isfinite(p)
        padj = np.full_like(p, np.nan, dtype=float)
        if ok.sum():
            padj[ok] = multipletests(p[ok], method="fdr_bh")[1]
        summary.loc[idx, "p_adj"] = padj
    summary.to_csv(OUT / "primary_contrasts_sample.csv", index=False)
    summary_d.to_csv(OUT / "primary_contrasts_donor.csv", index=False)

    # brief console report
    print("\n=== Primary full-thickness ΔCLR (sample / donor) ===")
    m = summary.merge(
        summary_d[["celltype", "delta_CLR", "p_value", "n_ids_A", "n_ids_B"]],
        on="celltype",
        suffixes=("_sample", "_donor"),
    )
    for _, r in m.iterrows():
        print(
            f"{r['short_name']:22s}  sample Δ={r['delta_CLR_sample']:+.3f} "
            f"(n={r['n_ids_A_sample']}/{r['n_ids_B_sample']}, p={r['p_value_sample']:.3g})  "
            f"donor Δ={r['delta_CLR_donor']:+.3f} "
            f"(n={r['n_ids_A_donor']}/{r['n_ids_B_donor']}, p={r['p_value_donor']:.3g})"
        )

    print("\n=== Niche capture (primary rule: GC B + ≥1 support) ===")
    prim = niche_rates[
        (niche_rates["rule"] == "niche_primary") & (niche_rates["strata"] == "depth")
    ]
    print(prim[["context", "n_samples", "n_pos", "capture_rate"]].to_string(index=False))
    rad = niche_rates[
        (niche_rates["rule"] == "niche_primary")
        & (niche_rates["strata"] == "radial_layer")
    ]
    print("\nBy radial_layer:")
    print(
        rad.sort_values("radial_layer")[
            ["radial_layer", "n_samples", "n_pos", "capture_rate"]
        ].to_string(index=False)
    )
    seg = niche_rates[
        (niche_rates["rule"] == "niche_primary") & (niche_rates["strata"] == "segment×depth")
    ]
    print("\nBy segment × depth:")
    print(
        seg.sort_values(["segment", "context"])[
            ["segment", "context", "n_samples", "n_pos", "capture_rate"]
        ].to_string(index=False)
    )
    segrad = niche_rates[
        (niche_rates["rule"] == "niche_primary")
        & (niche_rates["strata"] == "segment×radial_layer")
    ]
    print("\nBy segment × radial_layer (canonical segments):")
    segrad = segrad[
        segrad["segment"].isin(["duodenum", "jejunum", "ileum", "colon"])
    ]
    print(
        segrad.sort_values(["segment", "radial_layer"])[
            ["segment", "radial_layer", "n_samples", "n_pos", "capture_rate"]
        ].to_string(index=False)
    )

    # within-study availability
    print("\n=== Within-study both-arm datasets (featured) ===")
    for ct in FEATURED:
        both = forest[
            (forest["celltype"] == ct)
            & (forest["inference"] == "within-study")
            & (~forest["is_pooled"])
        ]
        print(f"{SHORT[ct]}: {list(both['dataset_id'])} deltas={both['delta_CLR'].tolist()}")

    print(f"\nWrote tables to {OUT}")


if __name__ == "__main__":
    main()
