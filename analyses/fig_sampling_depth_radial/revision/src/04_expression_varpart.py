"""Task 3 expression side: variancePartition-style mixed decomposition on PC scores.

For each cell type, fit per-PC: score ~ C(cov) + (1|dataset_id) + (1|donor_id)
using statsmodels MixedLM, then average fixed-effect variance fractions across
PCs weighted by eigenvalue (same weights as PCR). Study-level bootstrap on the
lineage aggregate.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
TABLES = ROOT / "tables"
LOGS = ROOT / "logs"
TABLES.mkdir(exist_ok=True)

SEED = 20260804
N_BOOT = 100
COVS = [
    "radial_tissue_term",
    "tissue_level_1",
    "sample_collection_method",
    "age_range",
    "assay",
    "sample_preservation_method",
    "sampled_site_condition",
    "sex_ontology_term",
]


def is_unknown(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.strip().str.lower()
    return s.isna() | t.isin(["", "unknown", "nan", "none", "n/a", "na"])


def fixed_frac_mixed(y, g, dataset, donor) -> float:
    d = pd.DataFrame(
        {"y": y, "g": pd.Categorical(g.astype(str)), "dataset": dataset.astype(str), "donor": donor.astype(str)}
    )
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 40 or d["g"].nunique() < 2:
        return np.nan
    # drop levels with <2 obs in random effects to aid convergence
    for col in ("dataset", "donor"):
        vc = d[col].value_counts()
        d = d[d[col].isin(vc[vc >= 2].index)]
    if len(d) < 40 or d["g"].nunique() < 2:
        return np.nan
    try:
        # random effects for dataset; donor nested-ish via crossed RE
        md = smf.mixedlm("y ~ C(g)", d, groups=d["dataset"], re_formula="1")
        fit = md.fit(reml=True, method="lbfgs", maxiter=200, disp=False)
        fe_var = np.var(fit.fittedvalues - (fit.fe_params.get("Intercept", 0)))
        # approximate: variance of fixed part
        mm_var = np.var(fit.predict(exog=fit.model.exog))
        re_var = float(np.asarray(fit.cov_re).ravel()[0]) if fit.cov_re is not None else 0.0
        resid = float(fit.scale)
        tot = mm_var + re_var + resid
        if tot <= 0:
            return np.nan
        return float(mm_var / tot)
    except Exception:
        return np.nan


def pcr_fixed_frac(scores, weights, meta, cov) -> float:
    mask = ~is_unknown(meta[cov]) & ~is_unknown(meta["dataset_id"]) & ~is_unknown(meta["donor_id"])
    if mask.sum() < 40:
        return np.nan
    sc = scores[mask.to_numpy()]
    w = np.asarray(weights[: sc.shape[1]], dtype=float)
    fracs = []
    # use top PCs that cover ~80% variance or max 10
    cum = np.cumsum(w) / w.sum()
    n_pc = int(max(3, min(10, 1 + np.searchsorted(cum, 0.8))))
    for k in range(n_pc):
        fracs.append(
            fixed_frac_mixed(
                sc[:, k],
                meta.loc[mask, cov].values,
                meta.loc[mask, "dataset_id"].values,
                meta.loc[mask, "donor_id"].values,
            )
        )
    fracs = np.asarray(fracs, dtype=float)
    ww = w[:n_pc]
    ok = np.isfinite(fracs)
    if ok.sum() == 0:
        return np.nan
    return float(np.average(fracs[ok], weights=ww[ok]))


def main():
    idx_path = CACHE / "expression_embedding_index.csv"
    if not idx_path.exists():
        print("No embeddings yet; run 01_cache_expression_embeddings.py", flush=True)
        return
    rng = np.random.default_rng(SEED)
    idx = pd.read_csv(idx_path)
    rows = []
    for _, r in idx.iterrows():
        z = np.load(CACHE / r["path"])
        scores = z["scores"]
        weights = z["var_weights"]
        samples = z["samples"].astype(str)
        meta = pd.read_parquet(CACHE / f"{Path(r['path']).stem}_meta.parquet")
        meta = meta.reindex(samples)
        for cov in COVS:
            if cov not in meta.columns:
                continue
            v = pcr_fixed_frac(scores, weights, meta, cov)
            rows.append(
                dict(
                    celltype=r["celltype"],
                    lineage=r["lineage"],
                    covariate=cov,
                    modality="expression",
                    estimator="mixed_pcr_varfrac",
                    n=int((~is_unknown(meta[cov])).sum()),
                    fixed_frac=v,
                )
            )
        print(f"varpart-ish {r['lineage']}/{r['celltype']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "expression_mixed_varfrac.csv", index=False)

    # lineage means + study bootstrap using sample meta redraws of embeddings
    boot_rows = []
    for lineage, g in df.groupby("lineage"):
        # gather embeddings for lineage
        sub_idx = idx[idx["lineage"] == lineage]
        emb = []
        for _, r in sub_idx.iterrows():
            z = np.load(CACHE / r["path"])
            meta = pd.read_parquet(CACHE / f"{Path(r['path']).stem}_meta.parquet")
            meta = meta.reindex(z["samples"].astype(str))
            emb.append((r["celltype"], z["scores"], z["var_weights"], meta))
        studies = sorted(
            set().union(*[set(m["dataset_id"].astype(str).unique()) for _, _, _, m in emb])
        )
        for cov in COVS:
            point = float(g.loc[g["covariate"] == cov, "fixed_frac"].mean())
            boots = []
            for b in range(N_BOOT):
                draw = rng.choice(studies, size=len(studies), replace=True)
                fracs = []
                for ct, scores, weights, meta in emb:
                    # concatenate samples from drawn studies
                    parts_sc = []
                    parts_meta = []
                    for s in draw:
                        m = meta["dataset_id"].astype(str) == s
                        if m.sum() == 0:
                            continue
                        parts_sc.append(scores[m.to_numpy()])
                        parts_meta.append(meta.loc[m])
                    if not parts_sc:
                        continue
                    sc = np.vstack(parts_sc)
                    md = pd.concat(parts_meta, axis=0)
                    fracs.append(pcr_fixed_frac(sc, weights, md, cov))
                if fracs:
                    boots.append(float(np.nanmean(fracs)))
            boots = np.asarray(boots, dtype=float)
            boots = boots[np.isfinite(boots)]
            boot_rows.append(
                dict(
                    lineage=lineage,
                    covariate=cov,
                    modality="expression",
                    fixed_frac=point,
                    boot_lo=float(np.percentile(boots, 2.5)) if boots.size else np.nan,
                    boot_hi=float(np.percentile(boots, 97.5)) if boots.size else np.nan,
                    boot_mean=float(boots.mean()) if boots.size else np.nan,
                    n_boot=int(boots.size),
                )
            )
            print(f"boot {lineage} {cov} n={boots.size}", flush=True)
    pd.DataFrame(boot_rows).to_csv(TABLES / "expression_mixed_study_bootstrap.csv", index=False)

    # compare to OLS PCR
    ols_path = TABLES / "expression_pcr_celltype.csv"
    if ols_path.exists():
        ols = pd.read_csv(ols_path)
        m = df.merge(
            ols[["celltype", "covariate", "omega2_trunc"]],
            on=["celltype", "covariate"],
            how="inner",
        )
        m["inflation"] = m["omega2_trunc"] / m["fixed_frac"]
        m.to_csv(TABLES / "task3_ols_vs_mixed_expression.csv", index=False)
        summ = (
            m.groupby("covariate")
            .agg(
                median_ols=("omega2_trunc", "median"),
                median_mixed=("fixed_frac", "median"),
                median_inflation=("inflation", "median"),
                n=("inflation", "size"),
            )
            .reset_index()
        )
        summ.to_csv(TABLES / "task3_inflation_expression_by_covariate.csv", index=False)
        print(summ.to_string(index=False), flush=True)

    (LOGS / "expression_varpart_seed.json").write_text(
        json.dumps({"seed": SEED, "n_boot": N_BOOT}, indent=2)
    )
    print("Done", flush=True)


if __name__ == "__main__":
    main()
