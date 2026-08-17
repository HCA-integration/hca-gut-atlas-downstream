#!/usr/bin/env python3
"""Supplemental follicle-capture analyses.

a) Threshold scan: GC B detection rate across cutoffs k=1..K
   (+ GSVA separation metrics to mark a data-driven 'best' k)
b) Sample table for GSVA violins at best k
c) Collection (biopsy vs resection) + study tables (dataset_id confound)
d) Sample-level design matrix for logistic mixed model

Universe: healthy + adjacent, ileum + colon (same as Fig. 4e).
Follicle+ call: (GC B LZ ≥ k) OR (GC B DZ ≥ k).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
CLR = HERE.parent.parent / "data" / "clr_long.csv"

GC = ["GC B Light Zone (GC B LZ)", "GC B Dark Zone (GC B DZ)"]
K_MAX = 40
PRIMARY_K = 5  # main-text cutoff
FOLLICLE_PROGRAMS = ["GC_module", "GC_DZ", "GC_LZ", "Tfh", "Tfr", "FARM", "fDC"]


def wilson(pos: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    lo, hi = proportion_confint(int(pos), int(n), method="wilson")
    return float(lo), float(hi)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return np.nan
    return float((a.mean() - b.mean()) / pooled)


def load_samples() -> pd.DataFrame:
    clr = pd.read_csv(CLR)
    meta = clr.drop_duplicates("sample_id")[
        [
            "sample_id",
            "donor_id",
            "dataset_id",
            "tissue_level_1",
            "sampled_site_condition",
            "radial_tissue_term",
            "sample_collection_method",
            "chemical_fractionation",
        ]
    ].copy()
    meta["segment"] = meta["tissue_level_1"].astype(str).str.lower()
    meta["site"] = meta["sampled_site_condition"].map(
        {"healthy": "Healthy", "adjacent": "Disease-adjacent"}
    )
    meta["collection"] = (
        meta["sample_collection_method"]
        .astype(str)
        .str.lower()
        .map({"biopsy": "Biopsy", "surgical resection": "Resection"})
    )
    meta["frac"] = meta["chemical_fractionation"].astype(str)
    meta = meta[
        meta["segment"].isin(["ileum", "colon"])
        & meta["site"].isin(["Healthy", "Disease-adjacent"])
        & meta["collection"].notna()
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
    tot = clr.groupby("sample_id")["n_cells"].sum().rename("total_cells")
    m = (
        meta.set_index("sample_id")
        .join(piv, how="inner")
        .join(tot, how="left")
        .reset_index()
    )
    m["gc_n"] = m[GC[0]] + m[GC[1]]
    m["gc_max"] = m[GC].max(axis=1)
    return m


def threshold_scan(m: pd.DataFrame, gsva_gc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Capture rates + GSVA separation across cutoffs."""
    # join GC_module GSVA (one row per sample)
    g = gsva_gc.set_index("sample_id")["gsva"]
    m = m.copy()
    m["gsva_gc"] = m["sample_id"].map(g)

    rate_rows = []
    sep_rows = []
    strata = [
        ("pooled", m),
        ("ileum", m[m["segment"] == "ileum"]),
        ("colon", m[m["segment"] == "colon"]),
        ("Biopsy", m[m["collection"] == "Biopsy"]),
        ("Resection", m[m["collection"] == "Resection"]),
        ("Healthy", m[m["site"] == "Healthy"]),
        ("Disease-adjacent", m[m["site"] == "Disease-adjacent"]),
    ]
    for k in range(1, K_MAX + 1):
        flag = (m[GC[0]] >= k) | (m[GC[1]] >= k)
        m[f"gc_ge{k}"] = flag
        for name, sub in strata:
            idx = sub.index
            f = flag.loc[idx]
            n = int(len(f))
            pos = int(f.sum())
            lo, hi = wilson(pos, n)
            rate_rows.append(
                dict(
                    cutoff=k,
                    strata=name,
                    n=n,
                    n_pos=pos,
                    rate=(pos / n) if n else np.nan,
                    ci_lo=lo,
                    ci_hi=hi,
                )
            )
        # GSVA separation on pooled
        pos_s = m.loc[flag, "gsva_gc"].to_numpy()
        neg_s = m.loc[~flag, "gsva_gc"].to_numpy()
        d = cohens_d(pos_s, neg_s)
        # Welch t p (descriptive)
        try:
            _, p = stats.ttest_ind(pos_s, neg_s, equal_var=False, nan_policy="omit")
        except Exception:
            p = np.nan
        # Youden-like: treat GSVA>0 as proxy "high mode", maximize sens+spec-1
        gs = m["gsva_gc"].to_numpy()
        high = np.isfinite(gs) & (gs > 0)
        if high.sum() > 0 and (~high & np.isfinite(gs)).sum() > 0:
            sens = float(flag[high].mean()) if high.sum() else np.nan
            spec = float((~flag)[~high].mean()) if (~high).sum() else np.nan
            youden = sens + spec - 1.0
        else:
            sens = spec = youden = np.nan
        sep_rows.append(
            dict(
                cutoff=k,
                n_pos=int(flag.sum()),
                n_neg=int((~flag).sum()),
                rate=float(flag.mean()),
                cohens_d=d,
                ttest_p=float(p) if np.isfinite(p) else np.nan,
                sens_vs_gsva_pos=sens,
                spec_vs_gsva_pos=spec,
                youden_vs_gsva_pos=youden,
                mean_gsva_pos=float(np.nanmean(pos_s)) if len(pos_s) else np.nan,
                mean_gsva_neg=float(np.nanmean(neg_s)) if len(neg_s) else np.nan,
            )
        )

    rates = pd.DataFrame(rate_rows)
    sep = pd.DataFrame(sep_rows)
    # best k by max Youden, fallback Cohen's d; prefer k in 3..15
    window = sep[(sep["cutoff"] >= 3) & (sep["cutoff"] <= 15)].copy()
    if window["youden_vs_gsva_pos"].notna().any():
        best = int(window.loc[window["youden_vs_gsva_pos"].idxmax(), "cutoff"])
        criterion = "youden_vs_gsva_pos"
    else:
        best = int(window.loc[window["cohens_d"].idxmax(), "cutoff"])
        criterion = "cohens_d"
    meta = pd.DataFrame(
        [
            dict(
                best_cutoff=best,
                criterion=criterion,
                primary_cutoff=PRIMARY_K,
                note=(
                    "Best k maximizes concordance of composition call with "
                    "GC-module GSVA>0 (Youden). Primary main-text cutoff remains k=5."
                ),
            )
        ]
    )
    return rates, sep, meta, best


def gsva_violin_table(m: pd.DataFrame, best_k: int) -> pd.DataFrame:
    gs = pd.read_csv(DATA / "niche_gsva_scores_long.csv")
    gs = gs[gs["scope_gut"] & gs["program"].isin(FOLLICLE_PROGRAMS)].copy()
    call = m[
        ["sample_id", f"gc_ge{best_k}", "segment", "site", "collection", "dataset_id"]
    ].rename(columns={f"gc_ge{best_k}": "follicle_pos"})
    # Drop overlapping metadata from GSVA table so call columns win
    drop_cols = [
        c
        for c in ("segment", "site", "collection", "dataset_id", "sampled_site_condition")
        if c in gs.columns
    ]
    out = gs.drop(columns=drop_cols).merge(call, on="sample_id", how="inner")
    out["cutoff"] = best_k
    out["follicle_label"] = np.where(
        out["follicle_pos"], "Follicle+", "Follicle−"
    )
    return out[
        [
            "sample_id",
            "program",
            "gsva",
            "follicle_pos",
            "follicle_label",
            "cutoff",
            "segment",
            "site",
            "collection",
            "dataset_id",
        ]
    ]


def _coll_mode(collections: pd.Series) -> str:
    has_b = (collections == "Biopsy").any()
    has_r = (collections == "Resection").any()
    if has_b and has_r:
        return "Mixed"
    if has_r:
        return "Resection"
    return "Biopsy"


def study_frac_tables(m: pd.DataFrame, best_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    flag = m[f"gc_ge{best_k}"]
    m = m.copy()
    m["follicle_pos"] = flag
    study = (
        m.groupby("dataset_id", as_index=False)
        .agg(
            n=("sample_id", "count"),
            n_pos=("follicle_pos", "sum"),
            n_donors=("donor_id", "nunique"),
            frac_mode=("frac", lambda s: s.value_counts().index[0]),
            n_frac_levels=("frac", "nunique"),
            segment_mode=("segment", lambda s: s.value_counts().index[0]),
            collection_mode=("collection", lambda s: s.value_counts().index[0]),
            coll_mode=("collection", _coll_mode),
            n_biopsy=("collection", lambda s: int((s == "Biopsy").sum())),
            n_resection=("collection", lambda s: int((s == "Resection").sum())),
            site_healthy_frac=("site", lambda s: (s == "Healthy").mean()),
        )
    )
    study["rate"] = study["n_pos"] / study["n"]
    study["ci_lo"], study["ci_hi"] = zip(
        *[wilson(int(p), int(n)) for p, n in zip(study["n_pos"], study["n"])]
    )
    study["is_kong"] = study["dataset_id"].astype(str).str.contains("Kong")
    study = study.sort_values("rate", ascending=False)

    # site × segment at best k (kept for secondary analyses; not panel c)
    site_rows = []
    for (seg, site), g in m.groupby(["segment", "site"]):
        n = len(g)
        pos = int(g["follicle_pos"].sum())
        lo, hi = wilson(pos, n)
        site_rows.append(
            dict(
                cutoff=best_k,
                segment=seg,
                site=site,
                n=n,
                n_pos=pos,
                rate=pos / n,
                ci_lo=lo,
                ci_hi=hi,
            )
        )
    return study, pd.DataFrame(site_rows)


def collection_tables(m: pd.DataFrame, best_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pooled biopsy vs resection + study-level rates for overlay points."""
    flag = m[f"gc_ge{best_k}"]
    m = m.copy()
    m["follicle_pos"] = flag

    pooled_rows = []
    for coll, g in m.groupby("collection"):
        n = len(g)
        pos = int(g["follicle_pos"].sum())
        lo, hi = wilson(pos, n)
        pooled_rows.append(
            dict(
                cutoff=best_k,
                scope="pooled",
                collection=coll,
                n=n,
                n_pos=pos,
                rate=pos / n if n else np.nan,
                ci_lo=lo,
                ci_hi=hi,
            )
        )
    pooled = pd.DataFrame(pooled_rows)

    # Fisher exact on pooled 2×2
    bio = m[m["collection"] == "Biopsy"]
    res = m[m["collection"] == "Resection"]
    table = [
        [int(bio["follicle_pos"].sum()), int((~bio["follicle_pos"]).sum())],
        [int(res["follicle_pos"].sum()), int((~res["follicle_pos"]).sum())],
    ]
    oddsr, p = stats.fisher_exact(table)
    pooled["odds_ratio"] = float(oddsr)
    pooled["p_fisher"] = float(p)

    study_rows = []
    for (ds, coll), g in m.groupby(["dataset_id", "collection"]):
        n = len(g)
        pos = int(g["follicle_pos"].sum())
        lo, hi = wilson(pos, n)
        study_rows.append(
            dict(
                cutoff=best_k,
                dataset_id=ds,
                collection=coll,
                n=n,
                n_pos=pos,
                rate=pos / n if n else np.nan,
                ci_lo=lo,
                ci_hi=hi,
            )
        )
    study_pts = pd.DataFrame(study_rows)
    return pooled, study_pts


def model_table(m: pd.DataFrame, best_k: int) -> pd.DataFrame:
    """Design matrix for ileum+colon logistic mixed model (estimable terms only)."""
    d = m.copy()
    d["gc"] = d[f"gc_ge{best_k}"].astype(int)
    d["log_total_cells"] = np.log1p(d["total_cells"].astype(float))
    # drop unknown fractionation for clean 2-level contrast (still study-nested!)
    d = d[d["frac"].isin(["unfractionated", "fractionated"])].copy()
    d["frac"] = pd.Categorical(
        d["frac"], categories=["unfractionated", "fractionated"]
    )
    d["collection"] = pd.Categorical(
        d["collection"], categories=["Biopsy", "Resection"]
    )
    d["site"] = pd.Categorical(
        d["site"], categories=["Healthy", "Disease-adjacent"]
    )
    d["segment"] = pd.Categorical(d["segment"], categories=["ileum", "colon"])
    return d[
        [
            "sample_id",
            "donor_id",
            "dataset_id",
            "gc",
            "segment",
            "site",
            "collection",
            "frac",
            "log_total_cells",
            "total_cells",
        ]
    ]


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    m = load_samples()
    print(f"Universe n={len(m)}")

    gsva = pd.read_csv(DATA / "niche_gsva_scores_long.csv")
    gsva_gc = gsva[
        (gsva["scope_gut"]) & (gsva["program"] == "GC_module")
    ][["sample_id", "gsva"]].drop_duplicates("sample_id")

    rates, sep, meta, best = threshold_scan(m, gsva_gc)
    # ensure gc_ge{best} columns exist on m (threshold_scan mutated a copy)
    for k in range(1, K_MAX + 1):
        m[f"gc_ge{k}"] = (m[GC[0]] >= k) | (m[GC[1]] >= k)

    rates.to_csv(DATA / "follicle_threshold_scan_rates.csv", index=False)
    sep.to_csv(DATA / "follicle_threshold_scan_separation.csv", index=False)
    meta.to_csv(DATA / "follicle_threshold_best.csv", index=False)
    print(meta.to_string(index=False))
    print(
        "Separation near best:\n",
        sep[sep["cutoff"].isin([best - 1, best, best + 1, PRIMARY_K])].to_string(
            index=False
        ),
    )

    # Use best for violins/model; also export primary-k variants in rates already
    viol = gsva_violin_table(m, best)
    viol.to_csv(DATA / "follicle_gsva_by_call_bestk.csv", index=False)
    # also at primary k=5 for comparison
    viol5 = gsva_violin_table(m, PRIMARY_K)
    viol5.to_csv(DATA / "follicle_gsva_by_call_k5.csv", index=False)

    study, site = study_frac_tables(m, best)
    study.to_csv(DATA / "follicle_capture_by_study_bestk.csv", index=False)
    site.to_csv(DATA / "follicle_capture_site_bestk.csv", index=False)

    coll_pooled, coll_study = collection_tables(m, best)
    coll_pooled.to_csv(DATA / "follicle_capture_by_collection_bestk.csv", index=False)
    coll_study.to_csv(
        DATA / "follicle_capture_by_collection_study_bestk.csv", index=False
    )
    print(
        "Collection (pooled):\n",
        coll_pooled.to_string(index=False),
    )

    # leave-Kong sensitivity for fractionation (ileum)
    ile = m[m["segment"] == "ileum"].copy()
    ile["follicle_pos"] = ile[f"gc_ge{best}"]
    sens_rows = []
    for label, sub in [
        ("all_ileum", ile),
        ("ileum_no_Kong", ile[~ile["dataset_id"].astype(str).str.contains("Kong")]),
    ]:
        for fr, g in sub.groupby("frac"):
            if fr not in ("unfractionated", "fractionated"):
                continue
            n = len(g)
            pos = int(g["follicle_pos"].sum())
            lo, hi = wilson(pos, n)
            sens_rows.append(
                dict(
                    scope=label,
                    cutoff=best,
                    frac=fr,
                    n=n,
                    n_pos=pos,
                    rate=pos / n if n else np.nan,
                    ci_lo=lo,
                    ci_hi=hi,
                )
            )
    pd.DataFrame(sens_rows).to_csv(
        DATA / "follicle_frac_leave_kong_sensitivity.csv", index=False
    )

    mod = model_table(m, best)
    mod.to_csv(DATA / "follicle_mixed_model_input.csv", index=False)
    print(f"Model input n={len(mod)} best_k={best}")
    print("Wrote supplemental CSVs to", DATA)


if __name__ == "__main__":
    main()
