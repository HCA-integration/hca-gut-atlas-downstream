#!/usr/bin/env python3
"""Screen metadata covariates for association with follicle (GC B) capture.

Call: follicle+ if GC B LZ ≥ 5 OR GC B DZ ≥ 5 (same as Fig. 4e primary).
Universe: healthy + adjacent, ileum + colon.

Writes:
  follicle_covariate_screen.csv
  follicle_covariate_plot_rates.csv
  follicle_covariate_fisher_by_segment.csv
  follicle_capture_by_study.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
CLR = HERE.parent.parent / "data" / "clr_long.csv"

GC = ["GC B Light Zone (GC B LZ)", "GC B Dark Zone (GC B DZ)"]
CUT = 5
BAD = {"nan", "None", "unknown", "Unknown", "NA", "n/a", "N/A", "", "None"}

PRETTY = {
    "sampled_site_condition": "Site condition",
    "sample_collection_method": "Collection",
    "radial_tissue_term": "Radial layer",
    "age_range": "Age",
    "assay": "Assay",
    "sex_ontology_term": "Sex",
    "sample_preservation_method": "Preservation",
    "sequenced_fragment": "Sequenced fragment",
    "gene_annotation_version": "Gene annotation",
    "chemical_fractionation": "Chemical fractionation",
    "dataset_id": "Study",
    "tissue_level_1": "Gut segment",
}
LEVEL_MAP = {
    "sampled_site_condition": {
        "healthy": "Healthy",
        "adjacent": "Disease-adjacent",
    },
    "sample_collection_method": {
        "biopsy": "Biopsy",
        "surgical resection": "Resection",
    },
    "chemical_fractionation": {
        "unfractionated": "Unfractionated",
        "fractionated": "Fractionated",
    },
    "radial_tissue_term": {
        "EPI": "EPI",
        "LP": "LP",
        "EPI_LP": "EPI+LP",
        "EPI_LP_MUSC": "Full thickness",
        "WM": "Muscle wall",
    },
}


def wilson(pos: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    lo, hi = proportion_confint(int(pos), int(n), method="wilson")
    return float(lo), float(hi)


def prep(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    return s.where(~s.isin(BAD) & series.notna())


def load_m() -> pd.DataFrame:
    clr = pd.read_csv(CLR)
    meta = clr.drop_duplicates("sample_id")[list(PRETTY) + ["sample_id"]].copy()
    # tissue_level_1 already in PRETTY
    meta["segment"] = meta["tissue_level_1"].astype(str).str.lower()
    meta = meta[
        meta["segment"].isin(["ileum", "colon"])
        & meta["sampled_site_condition"].isin(["healthy", "adjacent"])
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
    m["follicle"] = (m[GC[0]] >= CUT) | (m[GC[1]] >= CUT)
    return m


def test_cov(df: pd.DataFrame, cov: str, min_n: int = 10):
    x = prep(df[cov])
    d = pd.DataFrame(
        {
            "follicle": df.loc[x.notna(), "follicle"].astype(bool).values,
            "level": x.loc[x.notna()].values,
        }
    )
    keep = d["level"].value_counts()
    keep = keep[keep >= min_n].index.tolist()
    d = d[d["level"].isin(keep)]
    if d["level"].nunique() < 2:
        return None, None
    tab = pd.crosstab(d["level"], d["follicle"])
    tab = tab.reindex(columns=[False, True], fill_value=0).astype(int)
    chi2, p, _, _ = chi2_contingency(tab.values)
    n = int(tab.to_numpy().sum())
    r, k = tab.shape
    v = (
        float(np.sqrt(chi2 / (n * min(k - 1, r - 1))))
        if n and min(k - 1, r - 1) > 0
        else np.nan
    )
    g = d.groupby("level")["follicle"].agg(n="count", n_pos="sum").reset_index()
    g["rate"] = g["n_pos"] / g["n"]
    g["ci_lo"], g["ci_hi"] = zip(
        *[wilson(p, n) for p, n in zip(g["n_pos"], g["n"])]
    )
    summary = dict(
        covariate=cov,
        n=n,
        n_levels=int(r),
        chi2=float(chi2),
        p=float(p),
        cramers_v=v,
        rate_min=float(g["rate"].min()),
        rate_max=float(g["rate"].max()),
        rate_span=float(g["rate"].max() - g["rate"].min()),
        top=g.loc[g["rate"].idxmax(), "level"],
        top_rate=float(g["rate"].max()),
        bottom=g.loc[g["rate"].idxmin(), "level"],
        bottom_rate=float(g["rate"].min()),
    )
    return summary, g


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    m = load_m()
    print(f"Universe n={len(m)} follicle+={int(m.follicle.sum())}")

    rows = []
    for cov in PRETTY:
        mn = 15 if cov == "dataset_id" else 10
        s, rates = test_cov(m, cov, min_n=mn)
        if s is None:
            print(cov, "skipped")
            continue
        rows.append(s)
        print(
            f"{cov:28s} p={s['p']:.2e} V={s['cramers_v']:.3f} "
            f"span={s['rate_span']:.2f}  {s['bottom']}→{s['top']}"
        )
    scr = pd.DataFrame(rows).sort_values("p")
    scr["p_adj"] = multipletests(scr["p"], method="fdr_bh")[1]
    scr.to_csv(DATA / "follicle_covariate_screen.csv", index=False)

    # Plot set: FDR<0.05 excluding study (too many levels); always keep known hits
    sel = scr.loc[
        (scr["p_adj"] < 0.05) & (scr["covariate"] != "dataset_id"), "covariate"
    ].tolist()
    print("Selected:", sel)

    plot_rows = []
    fish_rows = []
    for cov in sel:
        for scope, df in [
            ("pooled", m),
            ("ileum", m[m["segment"] == "ileum"]),
            ("colon", m[m["segment"] == "colon"]),
        ]:
            mn = 8 if scope != "pooled" else 10
            s, rates = test_cov(df, cov, min_n=mn)
            if rates is None:
                continue
            if scope in ("ileum", "colon"):
                fish_rows.append(
                    dict(
                        covariate=cov,
                        segment=scope,
                        n=int(rates["n"].sum()),
                        n_levels=len(rates),
                        p=s["p"],
                        rate_span=s["rate_span"],
                    )
                )
            for _, r in rates.iterrows():
                lev = str(r["level"])
                plot_rows.append(
                    dict(
                        covariate=cov,
                        covariate_label=PRETTY[cov],
                        scope=scope,
                        level=lev,
                        level_label=LEVEL_MAP.get(cov, {}).get(lev, lev),
                        n=int(r["n"]),
                        n_pos=int(r["n_pos"]),
                        rate=float(r["rate"]),
                        ci_lo=float(r["ci_lo"]),
                        ci_hi=float(r["ci_hi"]),
                        screen_p=float(scr.loc[scr.covariate == cov, "p"].iloc[0]),
                        screen_p_adj=float(
                            scr.loc[scr.covariate == cov, "p_adj"].iloc[0]
                        ),
                        screen_v=float(
                            scr.loc[scr.covariate == cov, "cramers_v"].iloc[0]
                        ),
                    )
                )

    plot = pd.DataFrame(plot_rows)
    plot.to_csv(DATA / "follicle_covariate_plot_rates.csv", index=False)

    fish = pd.DataFrame(fish_rows)
    if len(fish):
        fish["p_adj"] = fish.groupby("covariate")["p"].transform(
            lambda x: multipletests(x, method="fdr_bh")[1]
        )
    fish.to_csv(DATA / "follicle_covariate_fisher_by_segment.csv", index=False)

    _, rates = test_cov(m, "dataset_id", min_n=15)
    if rates is not None:
        rates.sort_values("rate", ascending=False).to_csv(
            DATA / "follicle_capture_by_study.csv", index=False
        )

    print("\nScreen (FDR):")
    print(
        scr[["covariate", "n_levels", "p_adj", "cramers_v", "rate_span"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
