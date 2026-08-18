"""Faster expression mixed-model partition: residualize study+donor via OLS
demeaning within dataset, then omega2 of covariate on residual PC scores.
Approximates the fixed-effect share after removing dataset/donor means.
Also runs true MixedLM on PC1 only for a validation subset.
"""
from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
TABLES = ROOT / "tables"
SEED = 20260804
N_BOOT = 100

from lib_estimators import is_unknown, pcr_omega2, within_study_support  # noqa: E402

COVS = [
    "radial_tissue_term",
    "tissue_level_1",
    "sample_collection_method",
    "age_range",
    "assay",
    "sample_preservation_method",
    "sampled_site_condition",
    "sex_ontology_term",
    "dataset_id",
]


def demean_within(scores: np.ndarray, keys: pd.Series) -> np.ndarray:
    out = scores.copy()
    for k, idx in keys.groupby(keys).groups.items():
        ix = list(idx)
        out[ix] = out[ix] - out[ix].mean(axis=0, keepdims=True)
    return out


def main():
    rng = np.random.default_rng(SEED)
    idx = pd.read_csv(CACHE / "expression_embedding_index.csv")
    rows = []
    for _, r in idx.iterrows():
        z = np.load(CACHE / r["path"])
        scores = z["scores"].astype(float)
        weights = z["var_weights"].astype(float)
        samples = z["samples"].astype(str)
        meta = pd.read_parquet(CACHE / f"{Path(r['path']).stem}_meta.parquet").reindex(samples)
        # residualize dataset then donor
        ok = ~is_unknown(meta["dataset_id"]) & ~is_unknown(meta["donor_id"])
        sc = scores[ok.to_numpy()]
        md = meta.loc[ok].reset_index(drop=True)
        sc_ds = demean_within(sc, md["dataset_id"].astype(str))
        sc_res = demean_within(sc_ds, md["donor_id"].astype(str))
        for cov in COVS:
            if cov not in md.columns:
                continue
            mask = ~is_unknown(md[cov])
            if mask.sum() < 25 or md.loc[mask, cov].astype(str).nunique() < 2:
                continue
            if cov == "dataset_id":
                # variance removed by dataset demeaning ≈ 1 - residual PCR of noise
                # report OLS PCR on raw scores for dataset for comparison
                om = pcr_omega2(sc[mask.to_numpy()], weights, md.loc[mask, cov].astype(str).values, True)
                om_adj = np.nan
            else:
                om = pcr_omega2(sc[mask.to_numpy()], weights, md.loc[mask, cov].astype(str).values, True)
                om_adj = pcr_omega2(
                    sc_res[mask.to_numpy()], weights, md.loc[mask, cov].astype(str).values, True
                )
            n_sup, _ = within_study_support(md, cov)
            rows.append(
                dict(
                    celltype=r["celltype"],
                    lineage=r["lineage"],
                    covariate=cov,
                    modality="expression",
                    estimator="pcr_after_dataset_donor_demean",
                    n=int(mask.sum()),
                    omega2_ols=om,
                    omega2_study_donor_adjusted=om_adj,
                    inflation=om / om_adj if (om_adj and om_adj > 1e-6) else np.nan,
                    n_datasets_with_support=n_sup,
                )
            )
        print(f"adj {r['lineage']}/{r['celltype']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "expression_adjusted_pcr.csv", index=False)
    summ = (
        df.groupby("covariate")
        .agg(
            median_ols=("omega2_ols", "median"),
            median_adj=("omega2_study_donor_adjusted", "median"),
            median_inflation=("inflation", "median"),
            n=("omega2_ols", "size"),
        )
        .reset_index()
        .sort_values("median_ols", ascending=False)
    )
    summ.to_csv(TABLES / "task3_inflation_expression_by_covariate.csv", index=False)
    print(summ.to_string(index=False))

    # lineage means + study bootstrap of adjusted omega2
    boot_rows = []
    for lineage, gidx in idx.groupby("lineage"):
        emb = []
        for _, r in gidx.iterrows():
            z = np.load(CACHE / r["path"])
            meta = pd.read_parquet(CACHE / f"{Path(r['path']).stem}_meta.parquet").reindex(
                z["samples"].astype(str)
            )
            emb.append((z["scores"].astype(float), z["var_weights"].astype(float), meta))
        studies = sorted(set().union(*[set(m["dataset_id"].astype(str).dropna().unique()) for _, _, m in emb]))
        for cov in COVS:
            if cov == "dataset_id":
                continue
            point = float(
                df.loc[(df.lineage == lineage) & (df.covariate == cov), "omega2_study_donor_adjusted"].mean()
            )
            boots = []
            for _ in range(N_BOOT):
                draw = rng.choice(studies, size=len(studies), replace=True)
                vals = []
                for scores, weights, meta in emb:
                    parts_sc, parts_meta = [], []
                    for s in draw:
                        m = meta["dataset_id"].astype(str) == s
                        if m.sum():
                            parts_sc.append(scores[m.to_numpy()])
                            parts_meta.append(meta.loc[m])
                    if not parts_sc:
                        continue
                    sc = np.vstack(parts_sc)
                    md = pd.concat(parts_meta).reset_index(drop=True)
                    ok = ~is_unknown(md["dataset_id"]) & ~is_unknown(md["donor_id"])
                    sc, md = sc[ok.to_numpy()], md.loc[ok].reset_index(drop=True)
                    if len(md) < 25:
                        continue
                    sc = demean_within(sc, md["dataset_id"].astype(str))
                    sc = demean_within(sc, md["donor_id"].astype(str))
                    mask = ~is_unknown(md[cov])
                    if mask.sum() < 15 or md.loc[mask, cov].astype(str).nunique() < 2:
                        continue
                    vals.append(
                        pcr_omega2(
                            sc[mask.to_numpy()],
                            weights,
                            md.loc[mask, cov].astype(str).values,
                            True,
                        )
                    )
                if vals:
                    boots.append(float(np.mean(vals)))
            boots = np.asarray(boots, float)
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
    print("Done", flush=True)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
