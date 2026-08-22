#!/usr/bin/env python3
"""Variance explained + covariate confounding for follicle capture (panel c).

Universe: healthy + adjacent, ileum + colon.
Call: (GC B LZ ≥ k) OR (GC B DZ ≥ k), k from follicle_threshold_best.csv.

Writes:
  follicle_var_explained.csv   — univariate + study-adjusted pseudo-R²
  follicle_cov_cramers_v.csv   — pairwise Cramér's V among covariates
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
CLR = HERE.parent.parent / "data" / "clr_long.csv"

GC = ["GC B Light Zone (GC B LZ)", "GC B Dark Zone (GC B DZ)"]
BAD = {"nan", "None", "unknown", "Unknown", "NA", "n/a", "N/A", "", "None"}

# covariates to screen (column → plain label)
COVS = {
    "dataset_id": "Study",
    "chemical_fractionation": "Fractionation",
    "sample_collection_method": "Collection",
    "radial_tissue_term": "Radial layer",
    "sampled_site_condition": "Site condition",
    "tissue_level_1": "Gut segment",
    "age_range": "Age",
    "sex_ontology_term": "Sex",
    "assay": "Assay",
    "sample_preservation_method": "Preservation",
    "sequenced_fragment": "Sequenced fragment",
    "log_total_cells": "log(n_cells/sample)",
}

LEVEL_MAP = {
    "sampled_site_condition": {"healthy": "Healthy", "adjacent": "Disease-adjacent"},
    "sample_collection_method": {
        "biopsy": "Biopsy",
        "surgical resection": "Resection",
    },
    "chemical_fractionation": {
        "unfractionated": "Unfractionated",
        "fractionated": "Fractionated",
    },
    "tissue_level_1": {"ileum": "ileum", "colon": "colon"},
}


def prep_cat(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    return s.where(~s.isin(BAD) & series.notna())


def load_m(k: int) -> pd.DataFrame:
    clr = pd.read_csv(CLR)
    cols = ["sample_id"] + [c for c in COVS if c != "log_total_cells"]
    meta = clr.drop_duplicates("sample_id")[cols].copy()
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
    tot = clr.groupby("sample_id")["n_cells"].sum().rename("total_cells")
    m = (
        meta.set_index("sample_id")
        .join(piv, how="inner")
        .join(tot, how="left")
        .reset_index()
    )
    m["gc"] = ((m[GC[0]] >= k) | (m[GC[1]] >= k)).astype(int)
    m["log_total_cells"] = np.log1p(m["total_cells"].astype(float))
    # pretty categorical levels
    for col, mapping in LEVEL_MAP.items():
        if col in m.columns:
            m[col] = m[col].astype(str).str.lower().map(mapping).fillna(m[col])
    return m


def _ll_bernoulli(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def mcfadden_r2_categorical(y: np.ndarray, groups: pd.Series) -> float:
    """Exact McFadden R² for saturated categorical logistic (no optimizer)."""
    y = np.asarray(y, dtype=float)
    if y.sum() < 3 or (len(y) - y.sum()) < 3:
        return np.nan
    p0 = y.mean()
    ll_null = _ll_bernoulli(y, np.full(len(y), p0))
    # within-group rates
    g = pd.Series(groups).astype(str).to_numpy()
    p_hat = np.empty(len(y))
    for lev in np.unique(g):
        m = g == lev
        p_hat[m] = y[m].mean()
    ll_model = _ll_bernoulli(y, p_hat)
    if ll_null >= 0:  # degenerate
        return np.nan
    return float(1.0 - ll_model / ll_null)


def mcfadden_r2_continuous(y: np.ndarray, x: np.ndarray) -> float:
    """McFadden R² for a single continuous predictor via sklearn logistic."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    y, x = y[ok], x[ok]
    if y.sum() < 3 or (len(y) - y.sum()) < 3:
        return np.nan
    p0 = np.full(len(y), y.mean())
    ll_null = -log_loss(y, p0, normalize=False)
    try:
        clf = LogisticRegression(
            penalty=None, solver="lbfgs", max_iter=500
        )
        clf.fit(x.reshape(-1, 1), y)
        p = clf.predict_proba(x.reshape(-1, 1))[:, 1]
        ll_model = -log_loss(y, p, normalize=False)
        return float(1.0 - ll_model / ll_null)
    except Exception:
        return np.nan


def mcfadden_r2_multi(y: np.ndarray, X: pd.DataFrame) -> float:
    """McFadden R² for a multi-column design (study + cov)."""
    y = np.asarray(y, dtype=float)
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    y = y[X.index] if hasattr(X, "index") else y
    # realign
    X = X.reset_index(drop=True)
    y = np.asarray(y, dtype=float)[: len(X)]
    if len(y) != len(X) or y.sum() < 3 or (len(y) - y.sum()) < 3:
        return np.nan
    if X.shape[1] == 0:
        return np.nan
    p0 = np.full(len(y), y.mean())
    ll_null = -log_loss(y, p0, normalize=False)
    try:
        clf = LogisticRegression(
            penalty="l2", C=1e6, solver="lbfgs", max_iter=800
        )
        clf.fit(X.to_numpy(dtype=float), y)
        p = clf.predict_proba(X.to_numpy(dtype=float))[:, 1]
        ll_model = -log_loss(y, p, normalize=False)
        r2 = 1.0 - ll_model / ll_null
        return float(max(0.0, r2))
    except Exception:
        return np.nan


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    tab = pd.crosstab(a, b)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan
    chi2, _, _, _ = chi2_contingency(tab)
    n = tab.to_numpy().sum()
    r, k = tab.shape
    return float(np.sqrt(chi2 / (n * (min(r - 1, k - 1)))))


def main() -> None:
    best = pd.read_csv(DATA / "follicle_threshold_best.csv")
    k = int(best["best_cutoff"].iloc[0])
    m = load_m(k)
    print(f"Universe n={len(m)} k={k} rate={m.gc.mean():.3f}")

    # complete cases for study-adjusted models
    study = prep_cat(m["dataset_id"])

    var_rows = []
    for col, label in COVS.items():
        kind = "continuous" if col == "log_total_cells" else "categorical"
        if kind == "categorical":
            xraw = prep_cat(m[col])
            vc = xraw.value_counts()
            keep = vc[vc >= 8].index
            xraw = xraw.where(xraw.isin(keep))
        else:
            xraw = pd.to_numeric(m[col], errors="coerce")

        ok = xraw.notna() & m["gc"].notna()
        if kind == "categorical" and xraw[ok].nunique() < 2:
            continue
        y = m.loc[ok, "gc"].to_numpy()

        if kind == "categorical":
            r2_uni = mcfadden_r2_categorical(y, xraw[ok])
        else:
            r2_uni = mcfadden_r2_continuous(y, xraw[ok].to_numpy())

        # study-adjusted unique contribution (additive model, no study×cov
        # interactions — saturated joint overfits sparse cells)
        ok2 = ok & study.notna()
        r2_unique = np.nan
        r2_study = np.nan
        r2_joint = np.nan
        if ok2.sum() > 40 and col != "dataset_id":
            y2 = m.loc[ok2, "gc"].to_numpy()
            X_study = pd.get_dummies(
                prep_cat(study[ok2]), drop_first=True, dtype=float
            ).reset_index(drop=True)
            if kind == "categorical":
                X_cov = pd.get_dummies(
                    prep_cat(xraw[ok2]), drop_first=True, dtype=float
                ).reset_index(drop=True)
                X_cov.columns = [f"cov_{c}" for c in X_cov.columns]
            else:
                x_std = (
                    (xraw[ok2] - xraw[ok2].mean())
                    / (float(xraw[ok2].std(ddof=0)) or 1.0)
                ).to_numpy()
                X_cov = pd.DataFrame({"x": x_std})
            X_both = pd.concat([X_study, X_cov], axis=1)
            X_both = X_both.loc[:, ~X_both.columns.duplicated()]
            r2_study = mcfadden_r2_multi(y2, X_study)
            r2_joint = mcfadden_r2_multi(y2, X_both)
            if np.isfinite(r2_joint) and np.isfinite(r2_study):
                r2_unique = max(0.0, float(r2_joint - r2_study))

        n_lev = int(xraw[ok].nunique()) if kind == "categorical" else 1
        var_rows.append(
            dict(
                covariate=col,
                label=label,
                kind=kind,
                n=int(ok.sum()),
                n_levels=n_lev,
                r2_univariate=r2_uni,
                r2_study_alone=r2_study,
                r2_study_plus_cov=r2_joint,
                r2_unique_after_study=r2_unique,
                cutoff=k,
            )
        )

    var = pd.DataFrame(var_rows).sort_values(
        "r2_univariate", ascending=False, na_position="last"
    )
    var.to_csv(DATA / "follicle_var_explained.csv", index=False)
    print(var.to_string(index=False))

    # Pairwise Cramér's V among categorical covariates (confounding structure)
    cat_cols = [c for c in COVS if c != "log_total_cells"]
    labels = {c: COVS[c] for c in cat_cols}
    pairs = []
    for i, a in enumerate(cat_cols):
        for b in cat_cols[i:]:
            aa = prep_cat(m[a])
            bb = prep_cat(m[b])
            if a in LEVEL_MAP:
                # already remapped in load for some
                pass
            ok = aa.notna() & bb.notna()
            # drop rare levels
            for s in (aa, bb):
                vc = s[ok].value_counts()
                rare = vc[vc < 5].index
                ok = ok & ~s.isin(rare)
            if a == b:
                v = 1.0
            else:
                v = cramers_v(aa[ok], bb[ok])
            pairs.append(
                dict(
                    cov_a=a,
                    cov_b=b,
                    label_a=labels[a],
                    label_b=labels[b],
                    cramers_v=v,
                    n=int(ok.sum()),
                    cutoff=k,
                )
            )
            if a != b:
                pairs.append(
                    dict(
                        cov_a=b,
                        cov_b=a,
                        label_a=labels[b],
                        label_b=labels[a],
                        cramers_v=v,
                        n=int(ok.sum()),
                        cutoff=k,
                    )
                )

    cv = pd.DataFrame(pairs)
    cv.to_csv(DATA / "follicle_cov_cramers_v.csv", index=False)
    print("Wrote", DATA / "follicle_var_explained.csv")
    print("Wrote", DATA / "follicle_cov_cramers_v.csv")


if __name__ == "__main__":
    main()
