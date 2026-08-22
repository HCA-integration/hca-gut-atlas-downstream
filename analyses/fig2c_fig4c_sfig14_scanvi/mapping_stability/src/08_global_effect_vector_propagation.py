#!/usr/bin/env python3
"""Global compositional effect-vector stability across stroma reference realizations.

Uses EXISTING stroma predictions. Predefined TAURUS contrasts only.
Compares each counterfactual beta vector to full/seed0 beta vector.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paths import ANNOT_PARQUET, PREDICTIONS, TABLES  # noqa: E402

LOGGER = logging.getLogger("effect_vectors")
PSEUDO = 0.5
MIN_PATIENTS_PER_ARM = 3  # TAURUS Healthy n_patients=3; other arms still >>3
MIN_CELLS_PER_SAMPLE = 20
MIN_FRAC_SAMPLES_WITH_TYPE = 0.05  # prevalence filter frozen before atlas compare


def _clr_row(counts: np.ndarray) -> np.ndarray:
    x = counts + PSEUDO
    logx = np.log(x)
    return logx - logx.mean()


def _sample_clr(pred: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    tab = pd.crosstab(pred["sample_id"], pred["leaf_prediction"])
    tab = tab.reindex(columns=labels, fill_value=0)
    mat = tab.to_numpy(dtype=float)
    clr = np.vstack([_clr_row(r) for r in mat])
    return pd.DataFrame(clr, index=tab.index.astype(str), columns=labels)


def _contrast_beta(
    clr: pd.DataFrame,
    meta: pd.DataFrame,
    arm_a: set[str],
    arm_b: set[str],
    sample_col: str = "sample_id",
) -> tuple[np.ndarray, dict]:
    """Simple patient-mean difference of sample CLR means: mean(A) - mean(B)."""
    m = meta.drop_duplicates(sample_col).set_index(sample_col)
    # keep samples present in clr
    samples = clr.index.intersection(m.index)
    m = m.loc[samples]
    clr = clr.loc[samples]

    def patient_means(arm_samples: set[str]) -> pd.DataFrame:
        sub_s = [s for s in samples if s in arm_samples]
        if not sub_s:
            return pd.DataFrame(columns=clr.columns)
        sub = clr.loc[sub_s].copy()
        sub["Patient"] = m.loc[sub_s, "Patient"].astype(str).to_numpy()
        return sub.groupby("Patient").mean(numeric_only=True)

    a = patient_means(arm_a)
    b = patient_means(arm_b)
    info = {"n_patients_a": len(a), "n_patients_b": len(b), "n_samples_a": len(arm_a & set(samples)), "n_samples_b": len(arm_b & set(samples))}
    if len(a) < MIN_PATIENTS_PER_ARM or len(b) < MIN_PATIENTS_PER_ARM:
        return np.full(clr.shape[1], np.nan), {**info, "status": "insufficient_replication"}
    beta = a.mean(axis=0).to_numpy() - b.mean(axis=0).to_numpy()
    return beta, {**info, "status": "ok"}


def _build_contrasts(meta: pd.DataFrame) -> dict[str, tuple[set[str], set[str]]]:
    """Return contrast_name -> (arm_a sample set, arm_b sample set)."""
    m = meta.drop_duplicates("sample_id").copy()
    out = {}

    # 1. CD vs Healthy
    cd = set(m.loc[m.Disease.astype(str) == "CD", "sample_id"].astype(str))
    healthy = set(m.loc[m.Disease.astype(str) == "Healthy", "sample_id"].astype(str))
    out["CD_vs_Healthy"] = (cd, healthy)

    # 2. inflamed vs noninflamed (if column exists)
    for col in ("Inflammation", "inflammation", "Inflamed"):
        if col in m.columns:
            vals = m[col].astype(str).str.lower()
            inf = set(m.loc[vals.str.contains("inflamed") & ~vals.str.contains("non"), "sample_id"].astype(str))
            non = set(m.loc[vals.str.contains("non"), "sample_id"].astype(str))
            if not inf:
                # try exact
                inf = set(m.loc[m[col].astype(str).isin(["Inflamed", "inflamed", "Yes", "yes"]), "sample_id"].astype(str))
                non = set(m.loc[m[col].astype(str).isin(["Non-inflamed", "noninflamed", "Noninflamed", "No", "no"]), "sample_id"].astype(str))
            out["inflamed_vs_noninflamed"] = (inf, non)
            break

    # 3. Pre vs Post treatment
    for col in ("Treatment", "treatment", "Timepoint", "timepoint"):
        if col in m.columns:
            vals = m[col].astype(str)
            pre = set(m.loc[vals.str.contains("Pre", case=False, na=False), "sample_id"].astype(str))
            post = set(m.loc[vals.str.contains("Post", case=False, na=False), "sample_id"].astype(str))
            out["Pre_vs_Post"] = (pre, post)
            break

    # 4. Remission vs Non-remission at baseline if possible
    for col in ("Remission_status", "remission", "Remission"):
        if col in m.columns:
            vals = m[col].astype(str)
            rem = set(m.loc[vals.str.contains("Remission", case=False, na=False) & ~vals.str.contains("Non", case=False, na=False), "sample_id"].astype(str))
            non = set(m.loc[vals.str.contains("Non", case=False, na=False), "sample_id"].astype(str))
            # prefer baseline/pre if Treatment available
            if "Treatment" in m.columns:
                pre_mask = m["Treatment"].astype(str).str.contains("Pre", case=False, na=False)
                rem = set(m.loc[pre_mask & vals.str.contains("Remission", case=False, na=False) & ~vals.str.contains("Non", case=False, na=False), "sample_id"].astype(str))
                non = set(m.loc[pre_mask & vals.str.contains("Non", case=False, na=False), "sample_id"].astype(str))
            out["Remission_vs_Nonremission_baseline"] = (rem, non)
            break
    return out


def compare_vectors(beta: np.ndarray, ref: np.ndarray) -> dict:
    mask = np.isfinite(beta) & np.isfinite(ref)
    if mask.sum() < 3:
        return {"status": "too_few"}
    b, r = beta[mask], ref[mask]
    pear = float(np.corrcoef(b, r)[0, 1]) if np.std(b) > 0 and np.std(r) > 0 else np.nan
    spear = float(stats.spearmanr(b, r).correlation) if len(b) > 2 else np.nan
    cos = float(np.dot(b, r) / (np.linalg.norm(b) * np.linalg.norm(r) + 1e-12))
    rmse = float(np.sqrt(np.mean((b - r) ** 2)))
    sign_change = float(np.mean(np.sign(b) != np.sign(r))) if len(b) else np.nan
    # rank stability of top-|ref| effects
    k = min(10, len(r))
    top_ref = set(np.argsort(-np.abs(r))[:k])
    top_b = set(np.argsort(-np.abs(b))[:k])
    rank_jaccard = float(len(top_ref & top_b) / len(top_ref | top_b)) if top_ref else np.nan
    return {
        "status": "ok",
        "pearson": pear,
        "spearman": spear,
        "cosine": cos,
        "rmse": rmse,
        "frac_sign_change": sign_change,
        "top10_rank_jaccard": rank_jaccard,
        "n_labels": int(mask.sum()),
    }


def atlas_run(atlas: str) -> pd.DataFrame:
    base = PREDICTIONS / atlas / "stroma"
    full0 = pd.read_parquet(base / "full" / "seed0" / "predictions.parquet")
    # attach richer meta from annotation if needed
    ann = pd.read_parquet(ANNOT_PARQUET)
    ann_s = ann[ann.assigned_lineage.astype(str) == "stroma"].drop_duplicates("sample_id")
    meta_cols = ["sample_id", "Patient", "Disease"]
    for c in ("Inflammation", "Treatment", "Remission_status", "segment"):
        if c in ann_s.columns:
            meta_cols.append(c)
        elif c in full0.columns:
            pass
    meta = full0[["sample_id", "Patient", "Disease"]].drop_duplicates("sample_id")
    for c in ("Inflammation", "Treatment", "Remission_status", "segment"):
        if c in ann_s.columns:
            meta = meta.merge(ann_s[["sample_id", c]], on="sample_id", how="left")
        elif c in full0.columns:
            meta = meta.merge(full0[["sample_id", c]].drop_duplicates("sample_id"), on="sample_id", how="left")

    # freeze label set from full seed0 with prevalence filter
    n_samples = full0["sample_id"].nunique()
    prev = (
        full0.groupby("leaf_prediction")["sample_id"]
        .nunique()
        / n_samples
    )
    labels = sorted(prev[prev >= MIN_FRAC_SAMPLES_WITH_TYPE].index.astype(str).tolist())
    LOGGER.info("%s labels after prevalence filter: %d", atlas, len(labels))

    # filter samples with enough cells
    vc = full0.groupby("sample_id").size()
    keep_samples = set(vc[vc >= MIN_CELLS_PER_SAMPLE].index.astype(str))
    full0 = full0[full0.sample_id.astype(str).isin(keep_samples)]
    meta = meta[meta.sample_id.astype(str).isin(keep_samples)]

    contrasts = _build_contrasts(meta)
    ref_clr = _sample_clr(full0, labels)
    ref_betas = {}
    contrast_info = {}
    for name, (a, b) in contrasts.items():
        beta, info = _contrast_beta(ref_clr, meta, a, b)
        ref_betas[name] = beta
        contrast_info[name] = info
        LOGGER.info("%s %s ref: %s", atlas, name, info)

    rows = []
    for d in sorted(list(base.glob("omit_*/seed*")) + list(base.glob("full/seed*"))):
        if d.parent.name == "full" and d.name == "seed0":
            continue
        pred = pd.read_parquet(d / "predictions.parquet")
        pred = pred[pred.sample_id.astype(str).isin(keep_samples)]
        clr = _sample_clr(pred, labels)
        omit = d.parent.name.replace("omit_", "") if d.parent.name.startswith("omit_") else "full"
        seed = int(d.name.replace("seed", ""))
        for name, (a, b) in contrasts.items():
            if contrast_info[name].get("status") != "ok":
                continue
            if not np.isfinite(ref_betas[name]).any():
                continue
            beta, info = _contrast_beta(clr, meta, a, b)
            if info.get("status") != "ok":
                continue
            cmp_ = compare_vectors(beta, ref_betas[name])
            rows.append(
                {
                    "atlas": atlas,
                    "contrast": name,
                    "omitted_study": omit,
                    "model_seed": seed,
                    "realization": f"{d.parent.name}/{d.name}",
                    **cmp_,
                    **{f"ref_{k}": v for k, v in contrast_info[name].items() if k.startswith("n_")},
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    frames = [atlas_run(a) for a in ("HGCA", "PanGI")]
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(TABLES / "effect_vector_propagation_realizations.csv", index=False)

    summary = (
        all_df[all_df.omitted_study != "full"]
        .groupby(["atlas", "contrast"])
        .agg(
            n=("cosine", "count"),
            median_cosine=("cosine", "median"),
            median_pearson=("pearson", "median"),
            median_spearman=("spearman", "median"),
            median_rmse=("rmse", "median"),
            median_frac_sign_change=("frac_sign_change", "median"),
            median_top10_jaccard=("top10_rank_jaccard", "median"),
        )
        .reset_index()
    )
    summary.to_csv(TABLES / "effect_vector_propagation_summary.csv", index=False)

    seed_summary = (
        all_df[all_df.omitted_study == "full"]
        .groupby(["atlas", "contrast"])
        .agg(
            n=("cosine", "count"),
            median_cosine=("cosine", "median"),
            median_rmse=("rmse", "median"),
        )
        .reset_index()
    )
    seed_summary.to_csv(TABLES / "effect_vector_seed_control_summary.csv", index=False)

    man = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "min_patients_per_arm": MIN_PATIENTS_PER_ARM,
        "min_cells_per_sample": MIN_CELLS_PER_SAMPLE,
        "min_frac_samples_with_type": MIN_FRAC_SAMPLES_WITH_TYPE,
        "summary": summary.to_dict(orient="records"),
    }
    (TABLES / "effect_vector_propagation_manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    LOGGER.info("\n%s", summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
