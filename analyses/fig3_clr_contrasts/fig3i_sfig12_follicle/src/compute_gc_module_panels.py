#!/usr/bin/env python3
"""Build GC-module / LN-Tfh tables for gut-wall and lymph-node panel sets.

Gut scope: canonical gut segments only (no mesentery/accessory).
  Primary call: GC B LZ or DZ ≥ 3 cells ("GC-associated lymphoid module").

LN scope: gut segments + accessory + mesentery (mLN positive control).
  Primary call for pooled/study panels: Tfh ≥ 3 ("LN Tfh program"),
  because mesentery samples are Tfh-rich but almost never GC-annotated.
  Marker heatmap still reports GC, Tfh, FARM, stroma, med. sinus, etc.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent.parent
DATA = HERE.parent / "data"
DATA.mkdir(parents=True, exist_ok=True)

MARKERS = {
    "GC B LZ": "GC B Light Zone (GC B LZ)",
    "GC B DZ": "GC B Dark Zone (GC B DZ)",
    "Tfh": "CD4 Tfh",
    "Tfr": "CD4 Tfr",
    "FARM": "Follicle Associated Resident Macrophages",
    "fDC": "Follicular Dendritic Cells (fDC)",
    "FRC": "Fibroblastic Reticular Cells (FRC)",
    "mLTo": "Mesenchymal Lymphoid Tissue Organizer Cells (mLTo Cells)",
    "MRC": "Marginal Reticular Cells (MRC)",
    "Med. sinus endo.": "Medullary Sinus Endothelial",
    "Lymphatic endo.": "Lymphatic Endothelial",
}

GUT_CONTEXT_ORDER = ["EPI", "EPI_LP", "LP", "EPI_LP_MUSC", "WM"]
GUT_CONTEXT_LAB = {
    "EPI": "Epi",
    "EPI_LP": "Epi+LP",
    "LP": "LP",
    "EPI_LP_MUSC": "Full thickness",
    "WM": "WM",
}
LN_CONTEXT_ORDER = [
    "Epi",
    "Epi+LP",
    "LP",
    "Full thickness",
    "WM",
    "Accessory",
    "Mesentery (mLN)",
]
THR = 3


def wilson(pos: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    lo, hi = proportion_confint(pos, n, method="wilson")
    return float(lo), float(hi)


def load_base() -> tuple[pd.DataFrame, pd.DataFrame]:
    clr = pd.read_csv(PARENT / "data" / "clr_long.csv")
    meta = clr.drop_duplicates("sample_id")[
        [
            "sample_id",
            "donor_id",
            "dataset_id",
            "tissue_level_1",
            "radial_tissue_term",
            "sampled_site_condition",
            "sample_collection_method",
        ]
    ].copy()
    meta["segment"] = meta.tissue_level_1.astype(str).str.lower()
    meta["radial"] = meta.radial_tissue_term.astype(str).str.upper()
    meta = meta[meta.sampled_site_condition.isin(["healthy", "adjacent"])]

    piv = (
        clr[clr.celltype.isin(MARKERS.values())]
        .pivot_table(
            index="sample_id",
            columns="celltype",
            values="n_cells",
            aggfunc="sum",
            fill_value=0,
        )
    )
    for c in MARKERS.values():
        if c not in piv.columns:
            piv[c] = 0
    return meta, piv


def assign_context(meta: pd.DataFrame, scope: str) -> pd.DataFrame:
    m = meta.copy()
    if scope == "gut":
        m = m[m.segment.isin(["duodenum", "jejunum", "ileum", "colon"])].copy()
        m["context"] = m["radial"]
        m["context_lab"] = m["context"].map(GUT_CONTEXT_LAB)
        m = m[m.context.isin(GUT_CONTEXT_ORDER)].copy()
    else:
        m = m[
            m.segment.isin(
                ["duodenum", "jejunum", "ileum", "colon", "mesentery", "accessory"]
            )
        ].copy()
        m["context"] = np.where(
            m.segment.eq("mesentery"),
            "Mesentery (mLN)",
            np.where(
                m.segment.eq("accessory"),
                "Accessory",
                m["radial"].map(GUT_CONTEXT_LAB).fillna(m["radial"]),
            ),
        )
        m["context_lab"] = m["context"]
        m = m[m.context.isin(LN_CONTEXT_ORDER)].copy()
    return m


def build_scope(meta: pd.DataFrame, piv: pd.DataFrame, scope: str):
    m = assign_context(meta, scope).set_index("sample_id").join(piv, how="inner")
    m = m.reset_index()
    m["gc_module"] = (
        (m[MARKERS["GC B LZ"]] >= THR) | (m[MARKERS["GC B DZ"]] >= THR)
    ).astype(bool)
    m["tfh_program"] = (m[MARKERS["Tfh"]] >= THR).astype(bool)
    # Primary call for bars/boxplot
    if scope == "gut":
        m["primary_call"] = m["gc_module"]
        m["primary_metric"] = "gc_module"
        m["primary_label"] = "GC-associated lymphoid module"
    else:
        m["primary_call"] = m["tfh_program"]
        m["primary_metric"] = "tfh_program"
        m["primary_label"] = "LN Tfh program"
    m["scope"] = scope

    # Pooled rates for both metrics (useful for LN dual annotation)
    pooled_rows = []
    for metric, col, label in [
        ("gc_module", "gc_module", "GC-associated lymphoid module"),
        ("tfh_program", "tfh_program", "LN Tfh program"),
        ("primary", "primary_call", m["primary_label"].iloc[0]),
    ]:
        for ctx, g in m.groupby("context"):
            n = len(g)
            pos = int(g[col].sum())
            lo, hi = wilson(pos, n)
            pooled_rows.append(
                dict(
                    scope=scope,
                    metric=metric,
                    metric_label=label,
                    context=ctx,
                    context_lab=g["context_lab"].iloc[0],
                    n_samples=n,
                    n_pos=pos,
                    capture_rate=pos / n if n else np.nan,
                    ci_lo=lo,
                    ci_hi=hi,
                )
            )
    pooled = pd.DataFrame(pooled_rows)

    marker_rows = []
    for lab, ct in MARKERS.items():
        for ctx, g in m.groupby("context"):
            n = len(g)
            pos = int((g[ct] >= THR).sum())
            marker_rows.append(
                dict(
                    scope=scope,
                    context=ctx,
                    context_lab=g["context_lab"].iloc[0],
                    marker=lab,
                    celltype=ct,
                    n_samples=n,
                    n_pos=pos,
                    detection_rate=pos / n if n else np.nan,
                )
            )
    markers = pd.DataFrame(marker_rows)

    study_rows = []
    for (ctx, ds), g in m.groupby(["context", "dataset_id"]):
        n = len(g)
        if n < 1:
            continue
        pos = int(g["primary_call"].sum())
        lo, hi = wilson(pos, n)
        study_rows.append(
            dict(
                scope=scope,
                metric=m["primary_metric"].iloc[0],
                metric_label=m["primary_label"].iloc[0],
                context=ctx,
                context_lab=g["context_lab"].iloc[0],
                dataset_id=ds,
                segment=g["segment"].mode().iloc[0] if len(g) else "",
                n_samples=n,
                n_pos=pos,
                capture_rate=pos / n,
                ci_lo=lo,
                ci_hi=hi,
            )
        )
    study = pd.DataFrame(study_rows)

    # Segment × context study table for gut-style faceted boxplot
    seg_study_rows = []
    for (seg, ctx, ds), g in m.groupby(["segment", "context", "dataset_id"]):
        n = len(g)
        if n < 1:
            continue
        pos = int(g["primary_call"].sum())
        lo, hi = wilson(pos, n)
        seg_study_rows.append(
            dict(
                scope=scope,
                metric=m["primary_metric"].iloc[0],
                metric_label=m["primary_label"].iloc[0],
                segment=seg,
                context=ctx,
                context_lab=g["context_lab"].iloc[0],
                dataset_id=ds,
                n_samples=n,
                n_pos=pos,
                capture_rate=pos / n,
                ci_lo=lo,
                ci_hi=hi,
            )
        )
    seg_study = pd.DataFrame(seg_study_rows)

    return m, pooled, markers, study, seg_study


def main() -> None:
    meta, piv = load_base()
    for scope in ("gut", "ln"):
        m, pooled, markers, study, seg_study = build_scope(meta, piv, scope)
        m.to_csv(DATA / f"gc_module_samples_{scope}.csv", index=False)
        pooled.to_csv(DATA / f"gc_module_pooled_{scope}.csv", index=False)
        markers.to_csv(DATA / f"gc_module_markers_{scope}.csv", index=False)
        study.to_csv(DATA / f"gc_module_study_{scope}.csv", index=False)
        seg_study.to_csv(DATA / f"gc_module_study_segment_{scope}.csv", index=False)
        prim = pooled[pooled.metric == "primary"].sort_values(
            "capture_rate", ascending=False
        )
        print(f"\n=== {scope} n={len(m)} primary={m['primary_label'].iloc[0]} ===")
        print(
            prim[
                ["context_lab", "n_samples", "n_pos", "capture_rate", "ci_lo", "ci_hi"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
