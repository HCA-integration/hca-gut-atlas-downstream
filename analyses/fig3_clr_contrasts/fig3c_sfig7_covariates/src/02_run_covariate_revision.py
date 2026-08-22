"""Fig 3 covariate-importance revision: Tasks 1–2, 4–6, 8 (composition + PCR).

Writes intermediates under revision/tables/ and logs seeds + package versions.
Mixed-model / variancePartition (Task 3) and figure rendering are separate.
"""
from __future__ import annotations

import json
import platform
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT.parent
DATA = FIG / "data"
TABLES = ROOT / "tables"
LOGS = ROOT / "logs"
CACHE = ROOT / "cache"
for d in (TABLES, LOGS, CACHE):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_estimators import (  # noqa: E402
    ALL_COVARIATES,
    PRETTY,
    block_of,
    cramers_v,
    empirical_p,
    fdr_bh,
    is_unknown,
    lmg_shares,
    one_way_omega2,
    one_way_r2,
    pcr_omega2,
    theils_u,
    within_study_support,
)

SEED = 20260804
N_PERM = 1000
N_BOOT = 200
N_LMG_ORDERINGS = 200
MAIN_SEGMENTS = ["duodenum", "jejunum", "ileum", "colon"]
MIN_SAMPLES_CT = 40
MIN_TISSUES = 4
MIN_PER_TISSUE = 3
RANK_COVS = [
    "dataset_id",
    "radial_tissue_term",
    "tissue_level_1",
    "sample_collection_method",
    "age_range",
    "assay",
    "sample_preservation_method",
    "sampled_site_condition",
    "sex_ontology_term",
    "sequenced_fragment",
    "gene_annotation_version",
]


def log_versions():
    import sklearn
    import scipy
    import statsmodels

    info = {
        "seed": SEED,
        "n_perm": N_PERM,
        "n_boot_study": N_BOOT,
        "n_lmg_orderings": N_LMG_ORDERINGS,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (LOGS / "package_versions.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2), flush=True)
    return info


def load_clr_sample_table():
    """Wide-ish long CLR with one row per sample × cell type."""
    clr = pd.read_csv(DATA / "clr_long.csv")
    # sample-level metadata (first non-null across lineage rows)
    meta_cols = [
        "donor_id", "dataset_id", "tissue_level_1", "sampled_site_condition",
        "radial_tissue_term", "sample_preservation_method", "sex_ontology_term",
        "age_range", "assay", "sample_collection_method", "sequenced_fragment",
        "gene_annotation_version",
    ]

    def first_nonnull(s):
        s = s.dropna()
        return s.iloc[0] if len(s) else np.nan

    sample_meta = (
        clr.groupby("sample_id", sort=False)[meta_cols]
        .agg(first_nonnull)
        .reset_index()
    )
    return clr, sample_meta


def composition_support_filter(ct_df: pd.DataFrame) -> bool:
    """Match vignette support: main segments, >=40 samples, >=4 tissues, >=3/tissue."""
    sub = ct_df[ct_df["tissue_level_1"].astype(str).isin(MAIN_SEGMENTS)]
    if len(sub) < MIN_SAMPLES_CT:
        return False
    tc = sub.groupby("tissue_level_1").size()
    return int((tc > 0).sum()) >= MIN_TISSUES and int(tc.min()) >= MIN_PER_TISSUE


def task1_characterize(clr, sample_meta):
    """Write Task-1 characterization tables + markdown fragment."""
    # n per covariate at sample level
    rows = []
    masks = {}
    for cov in ALL_COVARIATES:
        m = ~is_unknown(sample_meta[cov])
        masks[cov] = m
        rows.append(
            dict(
                covariate=cov,
                pretty=PRETTY[cov],
                n_known=int(m.sum()),
                n_missing=int((~m).sum()),
                n_levels=int(sample_meta.loc[m, cov].astype(str).nunique()),
                block=block_of(cov),
            )
        )
    ntab = pd.DataFrame(rows)
    ntab.to_csv(TABLES / "task1_n_per_covariate.csv", index=False)

    # pairwise Jaccard of known-value sample subsets
    jac_rows = []
    for i, a in enumerate(ALL_COVARIATES):
        for b in ALL_COVARIATES[i + 1 :]:
            inter = int((masks[a] & masks[b]).sum())
            union = int((masks[a] | masks[b]).sum())
            jac_rows.append(
                dict(
                    cov_a=a,
                    cov_b=b,
                    n_a=int(masks[a].sum()),
                    n_b=int(masks[b].sum()),
                    n_both=inter,
                    jaccard=inter / union if union else np.nan,
                )
            )
    pd.DataFrame(jac_rows).to_csv(TABLES / "task1_subset_jaccard.csv", index=False)

    # published PCR n_samples are summed cell-type weights, not unique samples
    pcr_long = pd.read_csv(DATA / "composition_vs_expression_pcr_long.csv")
    sens = pd.read_csv(DATA / "celltype_metadata_sensitivity_top2.csv")

    md = []
    md.append("# Task 1 — Current estimator characterization\n")
    md.append("## a. What quantity is reported?\n")
    md.append(
        "**Fig 3b (top-2 cell-type heatmap; `celltype_metadata_sensitivity_top2.csv`)** "
        "computes **incremental partial R²** of `y ~ C(covariate)` vs `y ~ 1` on the "
        "within-lineage CLR column for each cell type "
        "(`Composition_patpy_variance.ipynb` → `partial_r2_y_vs_covariate`). "
        "Because the reduced model is intercept-only, this equals one-way ANOVA "
        "**R² / η²**, not ω², and not multi-term Type-II/III SS.\n"
    )
    md.append(
        "**Fig 3c (PCR lollipop + anatomy:study ratio; `composition_vs_expression_pcr_*.csv`)** "
        "computes **one-way ω² per PC**, then a **variance-weighted average across PCs** "
        "(scIB-style PCR; `composition_vs_expression_pcr.py` → `_anova_r2_all_pcs`). "
        "Each covariate is fit **marginally** (one at a time). Labels saying "
        "\"incremental partial R²\" vs \"Variance Explained ω²\" are therefore mixed "
        "across panels: heatmap = partial R²; PCR = ω².\n"
    )
    md.append("## b. Sequential term order?\n")
    md.append(
        "**No multi-term sequential (Type-I) SS is used in the published panels.** "
        "Each covariate enters alone against an intercept-only (heatmap) or as a "
        "one-way ANOVA on PC scores (PCR). Term entry order is therefore **not "
        "applicable** to the current estimates. Order dependence is tested in Task 4 "
        "via LMG / Shapley averaging of multi-covariate sequential R².\n"
    )
    md.append("## c. Truncation rule\n")
    md.append(
        "- Heatmap partial R²: `max(0.0, (SSR_red - SSR_full) / SSR_red)` "
        "(negative clipped to 0; upper clip not applied; R² already ∈ [0,1]).\n"
        "- PCR ω²: `np.clip(omega2, 0.0, 1.0)` after `nan_to_num` — applied "
        "**per PC before** variance-weighted averaging in `_anova_r2_all_pcs`.\n"
        "- Audit one-way ω² in `recompute_clr_tables.py`: `max(0.0, min(1.0, omega))`.\n"
    )
    md.append("## d. n per covariate / subset differences\n")
    md.append(
        f"Sample-level metadata from `clr_long.csv`: **{sample_meta.shape[0]} samples**, "
        f"**{sample_meta['donor_id'].nunique()} donors**, "
        f"**{sample_meta['dataset_id'].nunique()} dataset_id levels**.\n"
    )
    md.append(
        "Unknown/unreported values are excluded **per covariate**. In the current "
        "`clr_long` export, only `sex_ontology_term` (n=491) and `age_range` (n=485) "
        "have missingness; all other ranked covariates are complete (n=502). "
        "Complete-case across all 11 covariates: "
        f"**{(~pd.concat({c: is_unknown(sample_meta[c]) for c in ALL_COVARIATES}, axis=1).any(axis=1)).sum()}** samples.\n"
    )
    md.append(
        "PCR `n_samples` in `composition_vs_expression_pcr_long.csv` is the "
        "**sum of per-cell-type sample counts used as aggregation weights**, not "
        "unique atlas samples (e.g. epithelial composition rows show n≈2309).\n"
    )
    md.append("\n```\n")
    md.append(ntab.to_string(index=False))
    md.append("\n```\n")
    (LOGS / "TASK1_estimator_characterization.md").write_text("".join(md))
    print("Wrote Task 1 characterization", flush=True)
    return ntab, pcr_long, sens


def run_composition_celltype(clr, sample_meta, rng):
    """Per cell-type × covariate: R2, omega2±trunc, permutations, within-study."""
    rows = []
    # precompute study support on sample meta
    support = {
        cov: within_study_support(sample_meta, cov) for cov in ALL_COVARIATES
    }
    celltypes = (
        clr[["celltype", "lineage"]]
        .drop_duplicates()
        .sort_values(["lineage", "celltype"])
    )
    for _, ctrow in celltypes.iterrows():
        ct, lineage = ctrow["celltype"], ctrow["lineage"]
        cdf = clr[clr["celltype"] == ct].drop_duplicates("sample_id").copy()
        # merge freshest sample meta
        cdf = cdf.drop(columns=[c for c in ALL_COVARIATES + ["donor_id"] if c in cdf.columns], errors="ignore")
        cdf = cdf.merge(sample_meta, on="sample_id", how="left")
        if not composition_support_filter(cdf):
            continue
        # vignette-like analysis set: main segments
        cdf = cdf[cdf["tissue_level_1"].astype(str).isin(MAIN_SEGMENTS)].copy()
        y = cdf["clr"].to_numpy(dtype=float)
        for cov in ALL_COVARIATES:
            mask = ~is_unknown(cdf[cov])
            if mask.sum() < 10 or cdf.loc[mask, cov].astype(str).nunique() < 2:
                continue
            yy = y[mask.to_numpy()]
            gg = cdf.loc[mask, cov].astype(str).to_numpy()
            r2 = one_way_r2(yy, gg)
            om_raw = one_way_omega2(yy, gg, truncate=False)
            om_trunc = one_way_omega2(yy, gg, truncate=True)
            # permutations on truncated omega2 (display scale) and raw
            null_trunc = np.empty(N_PERM)
            null_raw = np.empty(N_PERM)
            for i in range(N_PERM):
                gperm = rng.permutation(gg)
                null_trunc[i] = one_way_omega2(yy, gperm, truncate=True)
                null_raw[i] = one_way_omega2(yy, gperm, truncate=False)
            p_emp = empirical_p(om_trunc, null_trunc, "greater")
            # within-study: restrict to datasets with support, residualize study
            n_sup, studies = support[cov]
            om_within = np.nan
            n_within = 0
            if n_sup >= 1 and cov != "dataset_id":
                sub = cdf.loc[mask & cdf["dataset_id"].astype(str).isin(studies)].copy()
                if len(sub) >= 10 and sub[cov].astype(str).nunique() >= 2:
                    # residualize CLR on dataset_id, then omega2 of covariate
                    from sklearn.linear_model import LinearRegression

                    n_studies_sub = sub["dataset_id"].astype(str).nunique()
                    if n_studies_sub >= 2:
                        Xd = pd.get_dummies(sub["dataset_id"].astype(str), drop_first=True)
                        lr = LinearRegression().fit(Xd.values, sub["clr"].values)
                        resid = sub["clr"].values - lr.predict(Xd.values)
                    else:
                        resid = sub["clr"].values - sub["clr"].values.mean()
                    om_within = one_way_omega2(
                        resid, sub[cov].astype(str).values, truncate=True
                    )
                    n_within = len(sub)
            z_null = (om_trunc - np.nanmean(null_trunc)) / (np.nanstd(null_trunc) + 1e-12)
            rows.append(
                dict(
                    celltype=ct,
                    lineage=lineage,
                    covariate=cov,
                    pretty=PRETTY[cov],
                    block=block_of(cov),
                    modality="composition",
                    estimator="oneway_clr",
                    n=int(mask.sum()),
                    partial_r2=r2,
                    omega2_raw=om_raw,
                    omega2_trunc=om_trunc,
                    omega2_negative_pretrunc=bool(np.isfinite(om_raw) and om_raw < 0),
                    null_mean=float(np.nanmean(null_trunc)),
                    null_p50=float(np.nanpercentile(null_trunc, 50)),
                    null_p95=float(np.nanpercentile(null_trunc, 95)),
                    null_p99=float(np.nanpercentile(null_trunc, 99)),
                    empirical_p=p_emp,
                    null_z=float(z_null),
                    identifiable_within_study=n_sup >= 2,
                    n_datasets_with_support=n_sup,
                    omega2_within_study=om_within,
                    n_within_study=n_within,
                )
            )
        print(f"  composition {lineage}/{ct}", flush=True)
    df = pd.DataFrame(rows)
    # FDR within covariate across cell types
    df["fdr_q"] = np.nan
    for cov, idx in df.groupby("covariate").groups.items():
        df.loc[idx, "fdr_q"] = fdr_bh(df.loc[idx, "empirical_p"])
    df.to_csv(TABLES / "composition_celltype_estimates.csv", index=False)
    return df


def run_composition_lmg(clr, sample_meta, rng):
    """LMG shares on complete-case main-segment samples, per cell type."""
    rows = []
    celltypes = clr[["celltype", "lineage"]].drop_duplicates()
    covs = [c for c in RANK_COVS if c != "dataset_id"]  # dataset handled separately
    # include dataset_id in LMG set
    covs_lmg = RANK_COVS
    for _, ctrow in celltypes.iterrows():
        ct, lineage = ctrow["celltype"], ctrow["lineage"]
        cdf = clr[clr["celltype"] == ct].drop_duplicates("sample_id").copy()
        cdf = cdf.drop(columns=[c for c in ALL_COVARIATES + ["donor_id"] if c in cdf.columns], errors="ignore")
        cdf = cdf.merge(sample_meta, on="sample_id", how="left")
        if not composition_support_filter(cdf):
            continue
        cdf = cdf[cdf["tissue_level_1"].astype(str).isin(MAIN_SEGMENTS)].copy()
        unk = pd.concat({c: is_unknown(cdf[c]) for c in covs_lmg}, axis=1)
        cdf = cdf.loc[~unk.any(axis=1)]
        if len(cdf) < 40:
            continue
        shares = lmg_shares(
            cdf["clr"].values,
            cdf[covs_lmg],
            covs_lmg,
            n_orderings=N_LMG_ORDERINGS,
            seed=int(rng.integers(1e9)),
        )
        # also compute sequential R2 for a few fixed orders to measure instability
        from lib_estimators import sequential_r2

        orders = {
            "published_bio_then_tech": [
                "tissue_level_1", "radial_tissue_term", "sampled_site_condition",
                "age_range", "sex_ontology_term", "sample_preservation_method",
                "dataset_id", "assay", "sample_collection_method",
                "sequenced_fragment", "gene_annotation_version",
            ],
            "tech_first": [
                "dataset_id", "assay", "sample_collection_method",
                "sequenced_fragment", "gene_annotation_version",
                "tissue_level_1", "radial_tissue_term", "sampled_site_condition",
                "age_range", "sex_ontology_term", "sample_preservation_method",
            ],
            "anatomy_last": [
                "dataset_id", "age_range", "sex_ontology_term", "assay",
                "sample_preservation_method", "sampled_site_condition",
                "sample_collection_method", "sequenced_fragment",
                "gene_annotation_version", "radial_tissue_term", "tissue_level_1",
            ],
        }
        seq = {}
        for name, order in orders.items():
            order = [c for c in order if c in covs_lmg]
            seq[name] = sequential_r2(cdf["clr"].values, cdf[covs_lmg], order)
        for cov, v in shares.items():
            rows.append(
                dict(
                    celltype=ct,
                    lineage=lineage,
                    covariate=cov,
                    modality="composition",
                    estimator="lmg",
                    n=len(cdf),
                    point=v,
                    seq_bio_then_tech=seq["published_bio_then_tech"].get(cov, np.nan),
                    seq_tech_first=seq["tech_first"].get(cov, np.nan),
                    seq_anatomy_last=seq["anatomy_last"].get(cov, np.nan),
                )
            )
        print(f"  LMG {lineage}/{ct} n={len(cdf)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "composition_lmg_celltype.csv", index=False)
    return df


def run_expression_pcr(rng):
    """Reproduce PCR omega2 on cached embeddings + permutation nulls."""
    idx_path = CACHE / "expression_embedding_index.csv"
    if not idx_path.exists():
        raise FileNotFoundError(
            "Missing expression embeddings. Run 01_cache_expression_embeddings.py first."
        )
    idx = pd.read_csv(idx_path)
    rows = []
    for _, r in idx.iterrows():
        z = np.load(CACHE / r["path"])
        scores = z["scores"]
        weights = z["var_weights"]
        samples = z["samples"].astype(str)
        meta = pd.read_parquet(CACHE / f"{Path(r['path']).stem}_meta.parquet")
        meta = meta.reindex(samples)
        for cov in ALL_COVARIATES:
            if cov not in meta.columns:
                continue
            mask = ~is_unknown(meta[cov])
            if mask.sum() < 10 or meta.loc[mask, cov].astype(str).nunique() < 2:
                continue
            gg = meta.loc[mask, cov].astype(str).to_numpy()
            sc = scores[mask.to_numpy()]
            om_raw = pcr_omega2(sc, weights, gg, truncate=False)
            om_trunc = pcr_omega2(sc, weights, gg, truncate=True)
            null = np.empty(N_PERM)
            for i in range(N_PERM):
                null[i] = pcr_omega2(sc, weights, rng.permutation(gg), truncate=True)
            p_emp = empirical_p(om_trunc, null, "greater")
            n_sup, studies = within_study_support(meta.reset_index(), cov)
            # within-study: residualize scores on dataset then PCR
            om_within = np.nan
            if n_sup >= 1 and cov != "dataset_id":
                m2 = mask & meta["dataset_id"].astype(str).isin(studies)
                if m2.sum() >= 10 and meta.loc[m2, cov].astype(str).nunique() >= 2:
                    from sklearn.linear_model import LinearRegression

                    resid = scores[m2.to_numpy()].copy()
                    n_st = meta.loc[m2, "dataset_id"].astype(str).nunique()
                    if n_st >= 2:
                        Xd = pd.get_dummies(
                            meta.loc[m2, "dataset_id"].astype(str), drop_first=True
                        )
                        for k in range(resid.shape[1]):
                            lr = LinearRegression().fit(Xd.values, resid[:, k])
                            resid[:, k] = resid[:, k] - lr.predict(Xd.values)
                    else:
                        resid = resid - resid.mean(axis=0, keepdims=True)
                    om_within = pcr_omega2(
                        resid,
                        weights[: resid.shape[1]],
                        meta.loc[m2, cov].astype(str).to_numpy(),
                        truncate=True,
                    )
            rows.append(
                dict(
                    celltype=r["celltype"],
                    lineage=r["lineage"],
                    covariate=cov,
                    pretty=PRETTY[cov],
                    block=block_of(cov),
                    modality="expression",
                    estimator="pcr_omega2",
                    n=int(mask.sum()),
                    omega2_raw=om_raw,
                    omega2_trunc=om_trunc,
                    omega2_negative_pretrunc=bool(np.isfinite(om_raw) and om_raw < 0),
                    null_mean=float(np.nanmean(null)),
                    null_p50=float(np.nanpercentile(null, 50)),
                    null_p95=float(np.nanpercentile(null, 95)),
                    null_p99=float(np.nanpercentile(null, 99)),
                    empirical_p=p_emp,
                    null_z=float(
                        (om_trunc - np.nanmean(null)) / (np.nanstd(null) + 1e-12)
                    ),
                    identifiable_within_study=n_sup >= 2,
                    n_datasets_with_support=n_sup,
                    omega2_within_study=om_within,
                )
            )
        print(f"  expression PCR {r['lineage']}/{r['celltype']}", flush=True)
    df = pd.DataFrame(rows)
    df["fdr_q"] = np.nan
    for cov, ix in df.groupby("covariate").groups.items():
        df.loc[ix, "fdr_q"] = fdr_bh(df.loc[ix, "empirical_p"])
    df.to_csv(TABLES / "expression_pcr_celltype.csv", index=False)

    # lineage aggregation with sample-count weights (published) + equal weights
    agg_rows = []
    for (lineage, cov, modality), g in df.groupby(["lineage", "covariate", "modality"]):
        w = g["n"].to_numpy(dtype=float)
        v = g["omega2_trunc"].to_numpy(dtype=float)
        ok = np.isfinite(v)
        if ok.sum() == 0:
            continue
        agg_rows.append(
            dict(
                lineage=lineage,
                covariate=cov,
                modality=modality,
                pcr_weighted=float(np.average(v[ok], weights=w[ok])),
                pcr_unweighted=float(np.mean(v[ok])),
                n_celltypes=int(ok.sum()),
                n_sample_weights=int(w[ok].sum()),
            )
        )
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(TABLES / "expression_pcr_lineage.csv", index=False)
    return df, agg


def run_composition_pcr(clr, sample_meta):
    """Lineage CLR → PCA → PCR omega2, paired to expression sample supports when possible."""
    rows = []
    for lineage, ldf in clr.groupby("lineage"):
        wide = ldf.pivot_table(index="sample_id", columns="celltype", values="clr")
        wide = wide.dropna(axis=0, how="any")
        meta = sample_meta.set_index("sample_id").reindex(wide.index)
        # z-score + PCA
        X = wide.values.astype(float)
        sd = X.std(axis=0)
        X = X[:, sd > 1e-12]
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        n_comp = min(X.shape[1], X.shape[0] - 1, 50)
        pca = PCA(n_components=n_comp, svd_solver="full")
        scores = pca.fit_transform(X)
        for cov in ALL_COVARIATES:
            mask = ~is_unknown(meta[cov])
            if mask.sum() < 10 or meta.loc[mask, cov].astype(str).nunique() < 2:
                continue
            om = pcr_omega2(
                scores[mask.to_numpy()],
                pca.explained_variance_,
                meta.loc[mask, cov].astype(str).to_numpy(),
                truncate=True,
            )
            om_raw = pcr_omega2(
                scores[mask.to_numpy()],
                pca.explained_variance_,
                meta.loc[mask, cov].astype(str).to_numpy(),
                truncate=False,
            )
            rows.append(
                dict(
                    lineage=lineage,
                    covariate=cov,
                    modality="composition",
                    estimator="pcr_omega2_lineage",
                    n=int(mask.sum()),
                    omega2_trunc=om,
                    omega2_raw=om_raw,
                )
            )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "composition_pcr_lineage.csv", index=False)
    return df


def identifiability_matrix(sample_meta):
    rows = []
    for i, a in enumerate(ALL_COVARIATES):
        for b in ALL_COVARIATES:
            ma = ~is_unknown(sample_meta[a])
            mb = ~is_unknown(sample_meta[b])
            both = ma & mb
            if both.sum() < 10:
                v = u_ab = u_ba = np.nan
            else:
                xa = sample_meta.loc[both, a].astype(str)
                xb = sample_meta.loc[both, b].astype(str)
                v = cramers_v(xa, xb)
                u_ab = theils_u(xa, xb)  # U(a|b)
                u_ba = theils_u(xb, xa)
            n_sup_a, _ = within_study_support(sample_meta, a)
            rows.append(
                dict(
                    cov_a=a,
                    cov_b=b,
                    cramers_v=v,
                    theils_u_a_given_b=u_ab,
                    theils_u_b_given_a=u_ba,
                    n=int(both.sum()) if both is not None else 0,
                    nonseparable_v_ge_0_9=bool(np.isfinite(v) and v >= 0.9),
                    nonseparable_u_ge_0_9=bool(
                        np.isfinite(u_ab) and max(u_ab, u_ba if np.isfinite(u_ba) else 0) >= 0.9
                    ),
                    n_datasets_with_support_a=n_sup_a,
                )
            )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "identifiability_matrix.csv", index=False)

    # paper table skeleton
    paper = []
    for cov in ALL_COVARIATES:
        n_sup, _ = within_study_support(sample_meta, cov)
        # confounders with V>=0.9
        conf = df[
            (df["cov_a"] == cov)
            & (df["cov_b"] != cov)
            & (df["nonseparable_v_ge_0_9"])
        ]["cov_b"].tolist()
        paper.append(
            dict(
                covariate=cov,
                pretty=PRETTY[cov],
                identifiable_within_study="Y" if n_sup >= 2 else "N",
                n_datasets_with_support=n_sup,
                nonseparable_partners=";".join(conf) if conf else "",
            )
        )
    pd.DataFrame(paper).to_csv(TABLES / "identifiability_summary.csv", index=False)
    return df


def study_bootstrap_lineage_pcr(clr, sample_meta, rng):
    """Study-level bootstrap CIs for lineage PCR omega2 (composition)."""
    studies = sample_meta["dataset_id"].astype(str).unique()
    rows = []
    for lineage, ldf in clr.groupby("lineage"):
        wide = ldf.pivot_table(index="sample_id", columns="celltype", values="clr")
        meta = sample_meta.set_index("sample_id")
        boot = {cov: [] for cov in RANK_COVS}
        for b in range(N_BOOT):
            draw = rng.choice(studies, size=len(studies), replace=True)
            # samples from drawn studies (with multiplicity via concat)
            parts = []
            for s in draw:
                parts.append(wide.loc[wide.index.isin(meta.index[meta["dataset_id"].astype(str) == s])])
            w = pd.concat(parts)
            if w.shape[0] < 30:
                continue
            w = w.dropna(axis=0, how="any")
            if w.shape[0] < 30:
                continue
            m = meta.reindex(w.index)
            X = w.values.astype(float)
            sd = X.std(axis=0)
            keep = sd > 1e-12
            if keep.sum() < 2:
                continue
            X = X[:, keep]
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
            n_comp = min(X.shape[1], X.shape[0] - 1, 30)
            if n_comp < 2:
                continue
            pca = PCA(n_components=n_comp, svd_solver="full")
            try:
                scores = pca.fit_transform(X)
            except Exception:
                continue
            for cov in RANK_COVS:
                mask = ~is_unknown(m[cov])
                if mask.sum() < 10 or m.loc[mask, cov].astype(str).nunique() < 2:
                    boot[cov].append(np.nan)
                    continue
                boot[cov].append(
                    pcr_omega2(
                        scores[mask.to_numpy()],
                        pca.explained_variance_,
                        m.loc[mask, cov].astype(str).to_numpy(),
                        truncate=True,
                    )
                )
        for cov, vals in boot.items():
            v = np.asarray(vals, dtype=float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            rows.append(
                dict(
                    lineage=lineage,
                    covariate=cov,
                    modality="composition",
                    n_boot=int(v.size),
                    boot_mean=float(v.mean()),
                    boot_lo=float(np.percentile(v, 2.5)),
                    boot_hi=float(np.percentile(v, 97.5)),
                )
            )
        print(f"  study bootstrap composition {lineage}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "composition_pcr_study_bootstrap.csv", index=False)
    return df


def assemble_authoritative(comp_ct, expr_ct, lmg, id_sum, pub_sens):
    """Build the paper-facing authoritative long table."""
    # published top-2 from heatmap
    pub = pub_sens.copy()
    pub.columns = ["celltype"] + [c for c in pub.columns[1:]]
    # melt published
    # columns are pretty names with spaces
    rows = []
    # composition oneway
    for _, r in comp_ct.iterrows():
        rows.append(
            dict(
                celltype=r["celltype"],
                lineage=r["lineage"],
                covariate=r["covariate"],
                estimator="oneway_omega2_trunc",
                n=r["n"],
                point_estimate=r["omega2_trunc"],
                point_estimate_raw=r["omega2_raw"],
                partial_r2=r["partial_r2"],
                bootstrap_lo=np.nan,
                bootstrap_hi=np.nan,
                null_mean=r["null_mean"],
                null_p50=r["null_p50"],
                null_p95=r["null_p95"],
                null_p99=r["null_p99"],
                empirical_p=r["empirical_p"],
                fdr_q=r["fdr_q"],
                null_z=r["null_z"],
                identifiable_within_study=r["identifiable_within_study"],
                n_datasets_with_support=r["n_datasets_with_support"],
                within_study_estimate=r["omega2_within_study"],
                modality="composition",
            )
        )
    for _, r in expr_ct.iterrows():
        rows.append(
            dict(
                celltype=r["celltype"],
                lineage=r["lineage"],
                covariate=r["covariate"],
                estimator="pcr_omega2_trunc",
                n=r["n"],
                point_estimate=r["omega2_trunc"],
                point_estimate_raw=r["omega2_raw"],
                partial_r2=np.nan,
                bootstrap_lo=np.nan,
                bootstrap_hi=np.nan,
                null_mean=r["null_mean"],
                null_p50=r["null_p50"],
                null_p95=r["null_p95"],
                null_p99=r["null_p99"],
                empirical_p=r["empirical_p"],
                fdr_q=r["fdr_q"],
                null_z=r["null_z"],
                identifiable_within_study=r["identifiable_within_study"],
                n_datasets_with_support=r["n_datasets_with_support"],
                within_study_estimate=r["omega2_within_study"],
                modality="expression",
            )
        )
    if lmg is not None and len(lmg):
        for _, r in lmg.iterrows():
            rows.append(
                dict(
                    celltype=r["celltype"],
                    lineage=r["lineage"],
                    covariate=r["covariate"],
                    estimator="lmg_r2",
                    n=r["n"],
                    point_estimate=r["point"],
                    point_estimate_raw=r["point"],
                    partial_r2=np.nan,
                    bootstrap_lo=np.nan,
                    bootstrap_hi=np.nan,
                    null_mean=np.nan,
                    null_p50=np.nan,
                    null_p95=np.nan,
                    null_p99=np.nan,
                    empirical_p=np.nan,
                    fdr_q=np.nan,
                    null_z=np.nan,
                    identifiable_within_study=np.nan,
                    n_datasets_with_support=np.nan,
                    within_study_estimate=np.nan,
                    modality="composition",
                )
            )
    auth = pd.DataFrame(rows)
    auth.to_csv(TABLES / "covariate_variance_authoritative.csv", index=False)

    # truncation bias summary
    trunc = (
        comp_ct.groupby("covariate")
        .agg(
            n_estimates=("omega2_raw", "size"),
            n_negative_pretrunc=("omega2_negative_pretrunc", "sum"),
            mean_raw=("omega2_raw", "mean"),
            mean_trunc=("omega2_trunc", "mean"),
        )
        .reset_index()
    )
    trunc.to_csv(TABLES / "task2_truncation_bias_composition.csv", index=False)
    trunc_e = (
        expr_ct.groupby("covariate")
        .agg(
            n_estimates=("omega2_raw", "size"),
            n_negative_pretrunc=("omega2_negative_pretrunc", "sum"),
            mean_raw=("omega2_raw", "mean"),
            mean_trunc=("omega2_trunc", "mean"),
        )
        .reset_index()
    )
    trunc_e.to_csv(TABLES / "task2_truncation_bias_expression.csv", index=False)

    # top-2 comparison vs published heatmap (partial R2)
    top_rows = []
    for cov in [
        "sampled_site_condition",
        "radial_tissue_term",
        "sample_preservation_method",
        "sex_ontology_term",
        "age_range",
        "assay",
        "sample_collection_method",
        "sequenced_fragment",
        "gene_annotation_version",
    ]:
        sub = comp_ct[comp_ct["covariate"] == cov].sort_values("partial_r2", ascending=False)
        top2 = sub.head(2)
        for rank, (_, rr) in enumerate(top2.iterrows(), 1):
            top_rows.append(
                dict(
                    covariate=cov,
                    rank=rank,
                    celltype=rr["celltype"],
                    partial_r2=rr["partial_r2"],
                    omega2_trunc=rr["omega2_trunc"],
                    null_z=rr["null_z"],
                    empirical_p=rr["empirical_p"],
                    fdr_q=rr["fdr_q"],
                    metric="partial_r2",
                )
            )
        subz = comp_ct[comp_ct["covariate"] == cov].sort_values("null_z", ascending=False)
        for rank, (_, rr) in enumerate(subz.head(2).iterrows(), 1):
            top_rows.append(
                dict(
                    covariate=cov,
                    rank=rank,
                    celltype=rr["celltype"],
                    partial_r2=rr["partial_r2"],
                    omega2_trunc=rr["omega2_trunc"],
                    null_z=rr["null_z"],
                    empirical_p=rr["empirical_p"],
                    fdr_q=rr["fdr_q"],
                    metric="null_z",
                )
            )
    pd.DataFrame(top_rows).to_csv(TABLES / "task6_top2_comparison.csv", index=False)
    return auth


def complete_case_stability(comp_ct, clr, sample_meta):
    unk = pd.concat({c: is_unknown(sample_meta[c]) for c in ALL_COVARIATES}, axis=1)
    ok = ~unk.any(axis=1)
    keep = set(sample_meta.loc[ok, "sample_id"])
    # recompute mean omega2 by covariate on complete-case subset only
    rows = []
    for cov in RANK_COVS:
        sub = comp_ct[comp_ct["covariate"] == cov]
        # approximate: restrict using n from complete case recomputed quickly
        vals = []
        for ct in sub["celltype"].unique():
            cdf = clr[clr["celltype"] == ct].drop_duplicates("sample_id")
            cdf = cdf[cdf["sample_id"].isin(keep)]
            cdf = cdf[cdf["tissue_level_1"].astype(str).isin(MAIN_SEGMENTS)]
            if len(cdf) < 20:
                continue
            vals.append(
                one_way_omega2(
                    cdf["clr"].values,
                    cdf[cov].astype(str).values,
                    truncate=True,
                )
            )
        rows.append(
            dict(
                covariate=cov,
                mean_omega2_complete_case=float(np.nanmean(vals)) if vals else np.nan,
                n_celltypes=len(vals),
                mean_omega2_primary=float(sub["omega2_trunc"].mean()),
            )
        )
    df = pd.DataFrame(rows)
    df["rank_primary"] = df["mean_omega2_primary"].rank(ascending=False)
    df["rank_complete"] = df["mean_omega2_complete_case"].rank(ascending=False)
    df.to_csv(TABLES / "task8_complete_case_ranking.csv", index=False)
    return df


def main():
    info = log_versions()
    rng = np.random.default_rng(SEED)
    print("Loading CLR…", flush=True)
    clr, sample_meta = load_clr_sample_table()
    sample_meta.to_csv(TABLES / "sample_metadata_502.csv", index=False)
    print(
        f"samples={len(sample_meta)} donors={sample_meta.donor_id.nunique()} "
        f"datasets={sample_meta.dataset_id.nunique()}",
        flush=True,
    )

    ntab, pcr_long, sens = task1_characterize(clr, sample_meta)
    print("Identifiability…", flush=True)
    id_mat = identifiability_matrix(sample_meta)
    id_sum = pd.read_csv(TABLES / "identifiability_summary.csv")

    print("Composition cell-type estimates + permutations…", flush=True)
    t0 = time.time()
    comp_ct = run_composition_celltype(clr, sample_meta, rng)
    print(f"  done in {time.time()-t0:.1f}s  rows={len(comp_ct)}", flush=True)

    print("Composition LMG…", flush=True)
    t0 = time.time()
    lmg = run_composition_lmg(clr, sample_meta, rng)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    print("Composition lineage PCR + study bootstrap…", flush=True)
    comp_pcr = run_composition_pcr(clr, sample_meta)
    boot = study_bootstrap_lineage_pcr(clr, sample_meta, rng)

    print("Expression PCR + permutations…", flush=True)
    t0 = time.time()
    try:
        expr_ct, expr_agg = run_expression_pcr(rng)
    except FileNotFoundError as e:
        print(f"WARNING: {e}", flush=True)
        expr_ct = pd.DataFrame()
        expr_agg = pd.DataFrame()
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    print("Assemble authoritative + stability…", flush=True)
    auth = assemble_authoritative(comp_ct, expr_ct, lmg, id_sum, sens)
    complete_case_stability(comp_ct, clr, sample_meta)

    # headline ranking tables
    rank_jobs = [
        ("composition", comp_ct, "omega2_trunc"),
        ("composition_nullz", comp_ct, "null_z"),
    ]
    if lmg is not None and len(lmg):
        rank_jobs.append(("composition_lmg", lmg, "point"))
    for modality, df, col in rank_jobs:
        if df is None or not len(df):
            continue
        g = (
            df.groupby("covariate")[col]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        g.to_csv(TABLES / f"ranking_{modality}.csv", index=False)
        print(f"\nRanking {modality}:\n", g.to_string(index=False), flush=True)

    if len(expr_ct):
        g = (
            expr_ct.groupby("covariate")["omega2_trunc"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        g.to_csv(TABLES / "ranking_expression_pcr.csv", index=False)
        gz = (
            expr_ct.groupby("covariate")["null_z"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        gz.to_csv(TABLES / "ranking_expression_nullz.csv", index=False)
        print(f"\nRanking expression PCR:\n", g.to_string(index=False), flush=True)
        print(f"\nRanking expression null-z:\n", gz.to_string(index=False), flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
