#!/usr/bin/env python3
"""Follicle (GC B) capture rates by gut segment × sampled_site_condition.

Same call as Fig. 3e / 4e: a sample is follicle+ if GC B LZ ≥ k OR GC B DZ ≥ k
cells. Universe: healthy + adjacent samples in ileum/colon (clr_long).

Primary contrast: Healthy vs disease-adjacent within each segment.
Secondary: ileum vs colon within each site-condition class.
Also exports collection-stratified rates (transparency / confounding check).

Outputs (../data):
  follicle_capture_by_site_condition.csv
  follicle_fisher_site_within_segment.csv
  follicle_fisher_segment_within_site.csv
  follicle_capture_by_site_condition_collection.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
CLR = HERE.parent.parent / "data" / "clr_long.csv"

GC_TYPES = [
    "GC B Light Zone (GC B LZ)",
    "GC B Dark Zone (GC B DZ)",
]
CUTOFFS = [3, 5, 10]
SEGMENTS = ["ileum", "colon"]

SITE_MAP = {
    "healthy": "Healthy",
    "adjacent": "Disease-adjacent",
}
SITE_LEVELS = ["Healthy", "Disease-adjacent"]


def wilson(pos: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    lo, hi = proportion_confint(pos, n, method="wilson")
    return float(lo), float(hi)


def load_sample_table() -> pd.DataFrame:
    clr = pd.read_csv(CLR)
    meta = (
        clr.drop_duplicates("sample_id")[
            [
                "sample_id",
                "dataset_id",
                "tissue_level_1",
                "sample_collection_method",
                "sampled_site_condition",
                "radial_tissue_term",
            ]
        ]
        .copy()
    )
    meta["segment"] = meta["tissue_level_1"].astype(str).str.lower()
    meta["site"] = meta["sampled_site_condition"].map(SITE_MAP)
    meta["collection"] = (
        meta["sample_collection_method"]
        .astype(str)
        .str.lower()
        .map({"biopsy": "Biopsy", "surgical resection": "Resection"})
    )
    meta = meta[
        meta["site"].isin(SITE_LEVELS)
        & meta["segment"].isin(SEGMENTS)
        & meta["collection"].notna()
    ]

    piv = (
        clr[clr["celltype"].isin(GC_TYPES)]
        .pivot_table(
            index="sample_id",
            columns="celltype",
            values="n_cells",
            aggfunc="sum",
            fill_value=0,
        )
    )
    for c in GC_TYPES:
        if c not in piv.columns:
            piv[c] = 0
    m = meta.set_index("sample_id").join(piv, how="inner").reset_index()
    for k in CUTOFFS:
        m[f"gc_ge{k}"] = (m[GC_TYPES[0]] >= k) | (m[GC_TYPES[1]] >= k)
    return m


def rate_rows(m: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for k in CUTOFFS:
        flag = f"gc_ge{k}"
        for keys, g in m.groupby(group_cols, observed=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            rec = dict(zip(group_cols, keys))
            n = len(g)
            pos = int(g[flag].sum())
            lo, hi = wilson(pos, n)
            rec.update(
                cutoff=k,
                n=n,
                n_pos=pos,
                rate=(pos / n) if n else np.nan,
                ci_lo=lo,
                ci_hi=hi,
            )
            rows.append(rec)
    return pd.DataFrame(rows)


def fisher_site_within_segment(m: pd.DataFrame) -> pd.DataFrame:
    """Healthy vs disease-adjacent within each segment."""
    rows = []
    for k in CUTOFFS:
        flag = f"gc_ge{k}"
        for seg in SEGMENTS:
            g = m[m["segment"] == seg]
            a = g[g["site"] == "Healthy"]
            b = g[g["site"] == "Disease-adjacent"]
            # table: rows = site, cols = neg/pos
            tab = [
                [int((~a[flag]).sum()), int(a[flag].sum())],
                [int((~b[flag]).sum()), int(b[flag].sum())],
            ]
            if min(len(a), len(b)) == 0:
                OR, p = np.nan, np.nan
            else:
                OR, p = fisher_exact(tab)
            rows.append(
                dict(
                    cutoff=k,
                    segment=seg,
                    contrast="Disease-adjacent vs Healthy",
                    odds_ratio=float(OR) if np.isfinite(OR) else np.nan,
                    p_value=float(p) if np.isfinite(p) else np.nan,
                    rate_healthy=float(a[flag].mean()) if len(a) else np.nan,
                    n_healthy=len(a),
                    pos_healthy=int(a[flag].sum()),
                    rate_adjacent=float(b[flag].mean()) if len(b) else np.nan,
                    n_adjacent=len(b),
                    pos_adjacent=int(b[flag].sum()),
                )
            )
    fish = pd.DataFrame(rows)
    for _, g in fish.groupby("cutoff"):
        ok = g["p_value"].notna()
        adj = np.full(len(g), np.nan)
        if ok.any():
            adj[ok.to_numpy()] = multipletests(
                g.loc[ok, "p_value"], method="fdr_bh"
            )[1]
        fish.loc[g.index, "p_adj"] = adj
    return fish


def fisher_segment_within_site(m: pd.DataFrame) -> pd.DataFrame:
    """Ileum vs colon within each site-condition class."""
    rows = []
    for k in CUTOFFS:
        flag = f"gc_ge{k}"
        for site in SITE_LEVELS:
            g = m[m["site"] == site]
            ile = g[g["segment"] == "ileum"]
            col = g[g["segment"] == "colon"]
            tab = [
                [int((~ile[flag]).sum()), int(ile[flag].sum())],
                [int((~col[flag]).sum()), int(col[flag].sum())],
            ]
            if min(len(ile), len(col)) == 0:
                OR, p = np.nan, np.nan
            else:
                OR, p = fisher_exact(tab)
            rows.append(
                dict(
                    cutoff=k,
                    site=site,
                    contrast="colon vs ileum",
                    odds_ratio=float(OR) if np.isfinite(OR) else np.nan,
                    p_value=float(p) if np.isfinite(p) else np.nan,
                    rate_ileum=float(ile[flag].mean()) if len(ile) else np.nan,
                    n_ileum=len(ile),
                    pos_ileum=int(ile[flag].sum()),
                    rate_colon=float(col[flag].mean()) if len(col) else np.nan,
                    n_colon=len(col),
                    pos_colon=int(col[flag].sum()),
                )
            )
    fish = pd.DataFrame(rows)
    for _, g in fish.groupby("cutoff"):
        ok = g["p_value"].notna()
        adj = np.full(len(g), np.nan)
        if ok.any():
            adj[ok.to_numpy()] = multipletests(
                g.loc[ok, "p_value"], method="fdr_bh"
            )[1]
        fish.loc[g.index, "p_adj"] = adj
    return fish


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    m = load_sample_table()
    print(
        "Universe (healthy/adjacent × ileum/colon):",
        len(m),
        "samples;",
        m.groupby(["segment", "site"]).size().to_dict(),
    )

    rates = rate_rows(m, ["segment", "site"])
    rates.to_csv(DATA / "follicle_capture_by_site_condition.csv", index=False)

    fish_site = fisher_site_within_segment(m)
    fish_site.to_csv(DATA / "follicle_fisher_site_within_segment.csv", index=False)

    fish_seg = fisher_segment_within_site(m)
    fish_seg.to_csv(DATA / "follicle_fisher_segment_within_site.csv", index=False)

    rates_coll = rate_rows(m, ["segment", "site", "collection"])
    rates_coll.to_csv(
        DATA / "follicle_capture_by_site_condition_collection.csv", index=False
    )

    print("\n=== Capture by segment × site (cutoff ≥5) ===")
    print(
        rates.query("cutoff == 5")[
            ["segment", "site", "n", "n_pos", "rate", "ci_lo", "ci_hi"]
        ].to_string(index=False)
    )
    print("\n=== Fisher: disease-adjacent vs healthy within segment (≥5) ===")
    print(
        fish_site.query("cutoff == 5")[
            [
                "segment",
                "rate_healthy",
                "n_healthy",
                "rate_adjacent",
                "n_adjacent",
                "odds_ratio",
                "p_value",
                "p_adj",
            ]
        ].to_string(index=False)
    )
    print("\n=== Fisher: colon vs ileum within site (≥5) ===")
    print(
        fish_seg.query("cutoff == 5")[
            [
                "site",
                "rate_ileum",
                "n_ileum",
                "rate_colon",
                "n_colon",
                "odds_ratio",
                "p_value",
                "p_adj",
            ]
        ].to_string(index=False)
    )
    print("\nWrote CSVs to", DATA)


if __name__ == "__main__":
    main()
