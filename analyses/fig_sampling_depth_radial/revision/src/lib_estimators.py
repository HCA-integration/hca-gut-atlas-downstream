"""Shared estimators for Fig 3 covariate-importance revision.

All functions are intentionally explicit about truncation and whether they
compute one-way R2 / partial R2, omega2, or order-averaged LMG shares.
"""
from __future__ import annotations

from itertools import permutations
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

UNKNOWN = {"", "unknown", "nan", "none", "n/a", "na", "not applicable"}

BIOLOGICAL = [
    "sampled_site_condition",
    "radial_tissue_term",
    "sample_preservation_method",
    "sex_ontology_term",
    "age_range",
]
TECHNICAL = [
    "dataset_id",
    "assay",
    "sample_collection_method",
    "sequenced_fragment",
    "gene_annotation_version",
]
REFERENCE = ["tissue_level_1"]
ALL_COVARIATES = BIOLOGICAL + TECHNICAL + REFERENCE

PRETTY = {
    "sampled_site_condition": "Sample condition",
    "radial_tissue_term": "Radial layer",
    "sample_preservation_method": "Preservation",
    "sex_ontology_term": "Sex",
    "age_range": "Age",
    "dataset_id": "Study / batch",
    "assay": "Assay",
    "sample_collection_method": "Biopsy vs resection",
    "sequenced_fragment": "Sequenced fragment",
    "gene_annotation_version": "Gene annotation",
    "tissue_level_1": "Gut segment",
}


def is_unknown(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.strip().str.lower()
    return s.isna() | t.isin(UNKNOWN)


def block_of(cov: str) -> str:
    if cov in BIOLOGICAL:
        return "biological"
    if cov in TECHNICAL:
        return "technical"
    return "reference"


def one_way_r2(y: np.ndarray, groups: np.ndarray) -> float:
    """One-way ANOVA R2 / eta2 (= incremental partial R2 vs intercept-only)."""
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    ok = np.isfinite(y)
    y, groups = y[ok], groups[ok]
    n = y.size
    if n < 3:
        return np.nan
    codes, inv = np.unique(groups, return_inverse=True)
    g = codes.size
    if g < 2 or n <= g:
        return np.nan
    grand = y.mean()
    ss_total = ((y - grand) ** 2).sum()
    if ss_total <= 0:
        return 0.0
    ss_between = 0.0
    for gi in range(g):
        m = inv == gi
        ss_between += m.sum() * (y[m].mean() - grand) ** 2
    return float(ss_between / ss_total)


def one_way_omega2(y: np.ndarray, groups: np.ndarray, truncate: bool = True) -> float:
    """One-way omega-squared. If truncate=False, negative / >1 values are kept."""
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    ok = np.isfinite(y)
    y, groups = y[ok], groups[ok]
    n = y.size
    if n < 3:
        return np.nan
    codes, inv = np.unique(groups, return_inverse=True)
    g = codes.size
    if g < 2 or n <= g:
        return np.nan
    grand = y.mean()
    ss_total = ((y - grand) ** 2).sum()
    if ss_total <= 0:
        return 0.0
    ss_between = 0.0
    for gi in range(g):
        m = inv == gi
        ss_between += m.sum() * (y[m].mean() - grand) ** 2
    ss_within = max(ss_total - ss_between, 0.0)
    ms_within = ss_within / (n - g)
    omega = (ss_between - (g - 1) * ms_within) / (ss_total + ms_within)
    if truncate:
        return float(np.clip(omega, 0.0, 1.0))
    return float(omega)


def omega2_all_pcs(scores: np.ndarray, groups: np.ndarray, truncate: bool = True) -> np.ndarray:
    """Per-PC one-way omega2 (vectorised). Matches composition_vs_expression_pcr.py."""
    scores = np.asarray(scores, dtype=float)
    n, k = scores.shape
    codes, inv = np.unique(groups, return_inverse=True)
    g = len(codes)
    if g < 2 or n - g < 1:
        return np.zeros(k)
    grand = scores.mean(axis=0)
    ss_total = ((scores - grand) ** 2).sum(axis=0)
    ss_between = np.zeros(k)
    for gi in range(g):
        m = inv == gi
        n_g = m.sum()
        if n_g == 0:
            continue
        ss_between += n_g * (scores[m].mean(axis=0) - grand) ** 2
    ss_within = np.clip(ss_total - ss_between, 0.0, None)
    ms_within = ss_within / (n - g)
    num = ss_between - (g - 1) * ms_within
    den = ss_total + ms_within
    with np.errstate(divide="ignore", invalid="ignore"):
        omega = np.where(den > 0, num / den, 0.0)
    omega = np.nan_to_num(omega, nan=0.0)
    if truncate:
        return np.clip(omega, 0.0, 1.0)
    return omega


def pcr_omega2(
    scores: np.ndarray,
    var_weights: np.ndarray,
    groups: np.ndarray,
    truncate: bool = True,
) -> float:
    w = np.asarray(var_weights, dtype=float)
    w_sum = w.sum()
    if w_sum <= 0:
        return np.nan
    r2 = omega2_all_pcs(scores, groups, truncate=truncate)
    return float((w * r2).sum() / w_sum)


def sequential_r2(y: np.ndarray, design: pd.DataFrame, order: list[str]) -> dict[str, float]:
    """Type-I sequential R2 increments for covariates in `order` (one-hot via pandas)."""
    from sklearn.linear_model import LinearRegression

    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    for c in order:
        ok &= ~is_unknown(design[c]).to_numpy()
    y = y[ok]
    if y.size < 10:
        return {c: np.nan for c in order}
    # build cumulative one-hot design
    base_ss = ((y - y.mean()) ** 2).sum()
    if base_ss <= 0:
        return {c: 0.0 for c in order}
    prev_ss = base_ss
    out = {}
    cols_so_far = []
    for cov in order:
        cols_so_far.append(cov)
        X = pd.get_dummies(design.loc[ok, cols_so_far].astype(str), drop_first=True)
        if X.shape[1] == 0:
            out[cov] = 0.0
            continue
        model = LinearRegression(fit_intercept=True)
        model.fit(X.values, y)
        resid = y - model.predict(X.values)
        ss = (resid ** 2).sum()
        out[cov] = float(max(0.0, (prev_ss - ss) / base_ss))
        prev_ss = ss
    return out


def lmg_shares(
    y: np.ndarray,
    design: pd.DataFrame,
    covariates: list[str],
    n_orderings: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """LMG / Shapley-style relative importance: mean sequential R2 over orderings."""
    rng = np.random.default_rng(seed)
    covs = [c for c in covariates if c in design.columns]
    if len(covs) == 0:
        return {}
    # exact if small, else random sample of orderings
    if len(covs) <= 6:
        orders = list(permutations(covs))
    else:
        orders = [tuple(rng.permutation(covs)) for _ in range(n_orderings)]
    acc = {c: 0.0 for c in covs}
    n_ok = 0
    for order in orders:
        shares = sequential_r2(y, design, list(order))
        if any(np.isfinite(v) for v in shares.values()):
            n_ok += 1
            for c, v in shares.items():
                if np.isfinite(v):
                    acc[c] += v
    if n_ok == 0:
        return {c: np.nan for c in covs}
    return {c: acc[c] / n_ok for c in covs}


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    tab = pd.crosstab(x, y)
    if tab.size == 0 or tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan
    chi2 = chi2_contingency(tab.values, correction=False)[0]
    n = tab.values.sum()
    r, k = tab.shape
    return float(np.sqrt(chi2 / (n * min(r - 1, k - 1))))


def theils_u(x: pd.Series, y: pd.Series) -> float:
    """Uncertainty coefficient U(x|y): fraction of X information explained by Y."""
    tab = pd.crosstab(x, y).values.astype(float)
    if tab.size == 0:
        return np.nan
    n = tab.sum()
    if n <= 0:
        return np.nan
    px = tab.sum(axis=1) / n
    py = tab.sum(axis=0) / n
    pxy = tab / n
    hx = -(px[px > 0] * np.log(px[px > 0])).sum()
    if hx <= 0:
        return np.nan
    hy_x = 0.0
    for j in range(tab.shape[1]):
        col = tab[:, j]
        s = col.sum()
        if s <= 0:
            continue
        p = col[col > 0] / s
        hy_x += (s / n) * (-(p * np.log(p)).sum())
    # U(x|y) = (H(x) - H(x|y)) / H(x); H(x|y) from joint
    hxy = -(pxy[pxy > 0] * np.log(pxy[pxy > 0])).sum()
    hy = -(py[py > 0] * np.log(py[py > 0])).sum()
    hx_given_y = hxy - hy
    return float((hx - hx_given_y) / hx)


def within_study_support(
    meta: pd.DataFrame,
    cov: str,
    study_col: str = "dataset_id",
    min_per_level: int = 5,
) -> tuple[int, list[str]]:
    """n datasets with >=2 levels each having >= min_per_level samples."""
    if cov not in meta.columns:
        return 0, []
    supported = []
    for study, sub in meta.groupby(study_col, observed=True):
        vals = sub.loc[~is_unknown(sub[cov]), cov].astype(str)
        counts = vals.value_counts()
        if (counts >= min_per_level).sum() >= 2:
            supported.append(str(study))
    return len(supported), supported


def empirical_p(obs: float, null: np.ndarray, alternative: str = "greater") -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or null.size == 0:
        return np.nan
    if alternative == "greater":
        return float((np.sum(null >= obs) + 1) / (null.size + 1))
    return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (null.size + 1))


def fdr_bh(pvals: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvals), dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out


def permutation_null_oneway(
    y: np.ndarray,
    groups: np.ndarray,
    n_perm: int,
    seed: int,
    statistic: str = "omega2",
    truncate: bool = True,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    ok = np.isfinite(y)
    y, groups = y[ok], groups[ok]
    out = np.empty(n_perm, dtype=float)
    fn = one_way_omega2 if statistic == "omega2" else one_way_r2
    for i in range(n_perm):
        g = rng.permutation(groups)
        if statistic == "omega2":
            out[i] = fn(y, g, truncate=truncate)
        else:
            out[i] = fn(y, g)
    return out
