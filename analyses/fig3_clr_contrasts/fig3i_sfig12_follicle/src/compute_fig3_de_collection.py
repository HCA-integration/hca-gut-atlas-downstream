#!/usr/bin/env python3
"""Recompute Fig. 3d/e collection stats with pooled ileum+colon.

Writes:
  fig3_e_capture_by_cutoff.csv   — segment rows + ileum+colon pooled
  fig3_e_fisher_by_cutoff.csv    — per-segment + pooled Fisher (BH within scope)
  fig3_de_biopsy_resection_stats.csv — summary used elsewhere

Bimodality (fig3_d_bimodality_stats.csv) already has an ileum+colon row.
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

GC = ["GC B Light Zone (GC B LZ)", "GC B Dark Zone (GC B DZ)"]
CUTOFFS = [3, 5, 10]
SEGMENTS = ["ileum", "colon"]


def wilson(pos: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    lo, hi = proportion_confint(int(pos), int(n), method="wilson")
    return float(lo), float(hi)


def load_samples() -> pd.DataFrame:
    clr = pd.read_csv(CLR)
    meta = clr.drop_duplicates("sample_id")[
        [
            "sample_id",
            "tissue_level_1",
            "sample_collection_method",
            "sampled_site_condition",
        ]
    ].copy()
    meta["segment"] = meta["tissue_level_1"].astype(str).str.lower()
    meta["collection"] = (
        meta["sample_collection_method"]
        .astype(str)
        .str.lower()
        .map({"biopsy": "Biopsy", "surgical resection": "Resection"})
    )
    meta["site"] = meta["sampled_site_condition"].map(
        {"healthy": "Healthy", "adjacent": "Disease-adjacent"}
    )
    # Match prior fig3_e universe: ileum/colon × biopsy/resection with
    # healthy + adjacent (same as follicle supplement / Fig. 4e).
    meta = meta[
        meta["segment"].isin(SEGMENTS)
        & meta["collection"].notna()
        & meta["site"].isin(["Healthy", "Disease-adjacent"])
    ]
    piv = (
        clr[clr["celltype"].isin(GC)]
        .pivot_table(
            index="sample_id",
            columns="celltype",
            values="n_cells",
            aggfunc="sum",
            fill_value=0,
        )
    )
    for c in GC:
        if c not in piv.columns:
            piv[c] = 0
    m = meta.set_index("sample_id").join(piv, how="inner").reset_index()
    return m


def rates_for(m: pd.DataFrame, cutoff: int, segment_label: str) -> list[dict]:
    rows = []
    for coll in ("Biopsy", "Resection"):
        g = m[m["collection"] == coll]
        n = len(g)
        pos = int(((g[GC[0]] >= cutoff) | (g[GC[1]] >= cutoff)).sum())
        lo, hi = wilson(pos, n)
        rows.append(
            dict(
                cutoff=cutoff,
                segment=segment_label,
                collection=coll,
                n=n,
                n_pos=pos,
                rate=pos / n if n else np.nan,
                ci_lo=lo,
                ci_hi=hi,
            )
        )
    return rows


def fisher_for(m: pd.DataFrame, cutoff: int, segment_label: str) -> dict:
    bio = m[m["collection"] == "Biopsy"]
    res = m[m["collection"] == "Resection"]
    pos_b = int(((bio[GC[0]] >= cutoff) | (bio[GC[1]] >= cutoff)).sum())
    neg_b = len(bio) - pos_b
    pos_r = int(((res[GC[0]] >= cutoff) | (res[GC[1]] >= cutoff)).sum())
    neg_r = len(res) - pos_r
    oddsr, p = fisher_exact([[pos_b, neg_b], [pos_r, neg_r]])
    return dict(
        cutoff=cutoff,
        segment=segment_label,
        odds_ratio=float(oddsr),
        p_value=float(p),
        rate_bio=pos_b / len(bio) if len(bio) else np.nan,
        n_bio=len(bio),
        pos_bio=pos_b,
        rate_res=pos_r / len(res) if len(res) else np.nan,
        n_res=len(res),
        pos_res=pos_r,
    )


def main() -> None:
    m = load_samples()
    print(f"Universe n={len(m)}")

    rate_rows: list[dict] = []
    fish_rows: list[dict] = []

    for cut in CUTOFFS:
        for seg in SEGMENTS:
            sub = m[m["segment"] == seg]
            rate_rows.extend(rates_for(sub, cut, seg))
            fish_rows.append(fisher_for(sub, cut, seg))
        # pooled ileum+colon
        rate_rows.extend(rates_for(m, cut, "ileum+colon"))
        fish_rows.append(fisher_for(m, cut, "ileum+colon"))

    rates = pd.DataFrame(rate_rows)
    # attach per-segment fisher OR/p onto rate rows for convenience
    fish = pd.DataFrame(fish_rows)
    # BH within each segment scope across cutoffs
    fish["p_adj"] = np.nan
    for seg, idx in fish.groupby("segment").groups.items():
        p = fish.loc[idx, "p_value"].to_numpy()
        fish.loc[idx, "p_adj"] = multipletests(p, method="fdr_bh")[1]

    # merge OR/p onto rates by cutoff×segment
    rates = rates.merge(
        fish[["cutoff", "segment", "odds_ratio", "p_value"]].rename(
            columns={"p_value": "p_fisher"}
        ),
        on=["cutoff", "segment"],
        how="left",
    )

    rates.to_csv(DATA / "fig3_e_capture_by_cutoff.csv", index=False)
    fish.to_csv(DATA / "fig3_e_fisher_by_cutoff.csv", index=False)

    # Compact stats summary (pooled + segment) for every cutoff; k=3 is primary
    summ_rows = []
    for _, r in fish.iterrows():
        summ_rows.append(
            dict(
                segment=r["segment"],
                panel="e",
                test="fisher_exact",
                n_bio=r["n_bio"],
                n_res=r["n_res"],
                median_bio=np.nan,
                median_res=np.nan,
                p_value=r["p_value"],
                p_adj=r["p_adj"],
                pos_bio=r["pos_bio"],
                rate_bio=r["rate_bio"],
                pos_res=r["pos_res"],
                rate_res=r["rate_res"],
                odds_ratio=r["odds_ratio"],
                cutoff=r["cutoff"],
            )
        )
    pd.DataFrame(summ_rows).to_csv(
        DATA / "fig3_de_biopsy_resection_stats.csv", index=False
    )

    print("Pooled capture (ileum+colon):")
    print(
        rates[rates["segment"] == "ileum+colon"][
            ["cutoff", "collection", "n", "n_pos", "rate", "p_fisher"]
        ].to_string(index=False)
    )
    print("Wrote", DATA / "fig3_e_capture_by_cutoff.csv")


if __name__ == "__main__":
    main()
