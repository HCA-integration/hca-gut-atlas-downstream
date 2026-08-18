#!/usr/bin/env python3
"""Stage 3: predict response from PRE-TREATMENT samples only.

Clinical question: given a patient's pre-treatment biopsy, can the sample-level
representation predict whether they go on to remission vs non-remission?
Because Remission_status is patient-level (constant across a patient's Pre/Post
samples), the pre-treatment sample carries the eventual-outcome label.

Design
------
* Samples: Treatment == "Pre", Disease in {CD, UC}, Remission_status in
  {Remission, Non_Remission}. Healthy / Not_avail dropped.
* Features: CLR-transformed cell-type composition (hgca_celltype_v1), computed
  directly from obs so the feature vocabulary is shared across tissues (lets us
  pool a disease across sites and still stratify).
* Models: regularised logistic regression (L2, balanced) and KNN, both
  evaluated with leave-one-PATIENT-out CV (cross_val_predict probabilities).
* Positive class = Non_Remission (the clinically important "non-responder").
* Stratification: out-of-fold predictions are sliced by Site and Inflammation
  to report per-stratum ROC-AUC / F1; disease (CD vs UC) is modelled separately.
* Supplementary: per-representation distance-KNN (using the Stage-1 distance
  matrices) per tissue x disease group, to see which embedding predicts best.
* Interpretability: logistic coefficients (which cell types drive non-remission)
  refit on the full per-disease cohort.

Outputs (data/):
  predict_samples.csv                 - the pre-treatment cohort used
  predict_metrics_overall.csv         - per disease x model, overall metrics
  predict_metrics_stratified.csv      - per disease x model x stratum metrics
  predict_oof_predictions.csv         - per-sample out-of-fold probabilities
  predict_logreg_celltype_coef.csv    - cell-type coefficients (interpretability)
  predict_representation_knn.csv      - supplementary per-representation KNN AUC
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import _common as C

warnings.filterwarnings("ignore")

POS_LABEL = "Non_Remission"
VALID_REMISSION = ["Remission", "Non_Remission"]
DROP_CELLTYPES = {"nan", "Not predicted", "Unknown", "Doublet", "Doublets", ""}


def log(m):
    print(f"[stage3] {m}", flush=True)


# --------------------------------------------------------------------------
def clr(counts: pd.DataFrame, pseudocount: float = 1.0) -> pd.DataFrame:
    """Centred log-ratio transform, matching patpy's native CellGroupComposition
    (`_compute_clr`): CLR on raw counts with an additive pseudocount, where each
    sample's log-counts are centred by their geometric mean (== mean of logs)."""
    log_counts = np.log(counts + pseudocount)
    gm = np.exp(log_counts.mean(axis=1))
    return log_counts.sub(np.log(gm), axis=0)


def build_cohort():
    """Per-sample CLR composition + metadata for the pre-treatment cohort."""
    usecols = [C.SAMPLE_KEY, C.PATIENT_KEY, C.DISEASE_KEY, "Site", C.TISSUE_KEY,
               C.TREATMENT_KEY, C.REMISSION_KEY, "Inflammation", "Age", "Gender",
               C.CELLTYPE_KEY]
    obs = pd.read_csv(C.TAURUS_OBS_CSV,
                      usecols=lambda c: c in usecols, low_memory=False)
    obs[C.CELLTYPE_KEY] = obs[C.CELLTYPE_KEY].astype(str)
    obs = obs[~obs[C.CELLTYPE_KEY].isin(DROP_CELLTYPES)]

    # Native-CLR cell-type composition (shared vocabulary across all samples).
    counts = pd.crosstab(obs[C.SAMPLE_KEY], obs[C.CELLTYPE_KEY])
    comp_clr = clr(counts)

    meta = obs.drop_duplicates(C.SAMPLE_KEY).set_index(C.SAMPLE_KEY)
    meta = meta.drop(columns=[C.CELLTYPE_KEY])
    meta["n_cells"] = counts.sum(axis=1)

    # Pre-treatment cohort with a usable response label.
    keep = (meta[C.TREATMENT_KEY].astype(str) == "Pre") & \
           (meta[C.DISEASE_KEY].astype(str).isin(C.DISEASES)) & \
           (meta[C.REMISSION_KEY].astype(str).isin(VALID_REMISSION))
    meta = meta[keep].copy()
    comp_clr = comp_clr.loc[meta.index]
    return comp_clr, meta


# --------------------------------------------------------------------------
def evaluate_oof(y_true, proba, pos_label=POS_LABEL):
    """Metrics from out-of-fold probabilities for the positive class.

    `y_true` is a Series/array of string labels; positive == pos_label.
    """
    y = (pd.Series(y_true).reset_index(drop=True) == pos_label).astype(int).values
    proba = np.asarray(proba)
    pred = (proba >= 0.5).astype(int)
    out = {"n": len(y), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
    out["roc_auc"] = roc_auc_score(y, proba) if len(np.unique(y)) == 2 else np.nan
    out["f1_pos"] = f1_score(y, pred, pos_label=1, zero_division=0)
    out["f1_macro"] = f1_score(y, pred, average="macro", zero_division=0)
    out["balanced_acc"] = balanced_accuracy_score(y, pred) if len(np.unique(y)) == 2 else np.nan
    out["accuracy"] = accuracy_score(y, pred)
    return out


def make_models():
    return {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty="l2", C=0.5, class_weight="balanced",
                               max_iter=2000, solver="liblinear"),
        ),
        "knn": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5, weights="distance"),
        ),
    }


def run_cv(X, meta, disease):
    """LOPO-CV out-of-fold P(Non_Remission) per model for one disease cohort."""
    sub = meta[meta[C.DISEASE_KEY].astype(str) == disease]
    if sub.shape[0] < 8 or sub[C.REMISSION_KEY].nunique() < 2:
        log(f"  {disease}: too few samples / single class (n={sub.shape[0]}), skip")
        return None
    Xd = X.loc[sub.index]
    Xd = Xd.loc[:, Xd.var() > 0]
    y = sub[C.REMISSION_KEY].astype(str)
    groups = sub[C.PATIENT_KEY].astype(str)
    cv = LeaveOneGroupOut()
    log(f"  {disease}: n={sub.shape[0]} samples, {groups.nunique()} patients, "
        f"{(y == POS_LABEL).sum()} non-remission / {(y != POS_LABEL).sum()} remission, "
        f"{Xd.shape[1]} features")

    oof = {}
    classes = np.unique(y.values)
    pos_idx = int(np.where(classes == POS_LABEL)[0][0])
    for name, model in make_models().items():
        proba = cross_val_predict(model, Xd.values, y.values, cv=cv,
                                  groups=groups.values, method="predict_proba")
        oof[name] = pd.Series(proba[:, pos_idx], index=sub.index)
    return sub, y, oof, Xd


def stratified_metrics(disease, model_name, y, proba, meta):
    """Overall + per-stratum metrics for one (disease, model)."""
    rows = [{"disease": disease, "model": model_name, "stratum_var": "overall",
             "stratum": "all", **evaluate_oof(y.values, proba.values)}]
    for var in ["Site", "Inflammation"]:
        if var not in meta.columns:
            continue
        s = meta.loc[y.index, var].astype(str)
        for level in sorted(s.unique()):
            idx = s.index[s == level]
            if len(idx) < 5:
                continue
            yi = y.loc[idx]
            if yi.nunique() < 2:
                continue
            rows.append({"disease": disease, "model": model_name,
                         "stratum_var": var, "stratum": level,
                         **evaluate_oof(yi.values, proba.loc[idx].values)})
    return rows


def logreg_coefficients(Xd, y, disease):
    """Refit L2 logistic on the full cohort -> cell-type coefficients."""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l2", C=0.5, class_weight="balanced",
                           max_iter=2000, solver="liblinear"),
    )
    yb = (y == POS_LABEL).astype(int)
    model.fit(Xd.values, yb.values)
    coef = model.named_steps["logisticregression"].coef_.ravel()
    return pd.DataFrame({"disease": disease, "celltype": Xd.columns,
                         "coef_nonremission": coef}).sort_values(
        "coef_nonremission", ascending=False)


# --------------------------------------------------------------------------
def loo_distance_knn(D, y, k=5):
    """Leave-one-out KNN on a precomputed distance matrix -> P(pos)."""
    n = len(y)
    if n < 6:
        return None
    proba = np.zeros(n)
    kk = int(min(k, n - 1))
    for i in range(n):
        order = np.argsort(D[i])
        order = order[order != i][:kk]
        w = 1.0 / (D[i, order] + 1e-9)
        proba[i] = np.average(y[order], weights=w)
    return proba


def representation_knn_supplement():
    """Per tissue x disease group, LOO distance-KNN AUC for each representation."""
    rows = []
    for tissue, disease, label in C.GROUPS:
        meta_path = C.group_meta_path(label)
        if not meta_path.exists():
            continue
        meta = pd.read_csv(meta_path, index_col=0)
        meta.index = meta.index.astype(str)
        pre = meta[(meta[C.TREATMENT_KEY].astype(str) == "Pre")
                   & (meta[C.REMISSION_KEY].astype(str).isin(VALID_REMISSION))]
        if pre.shape[0] < 8 or pre[C.REMISSION_KEY].nunique() < 2:
            continue
        y = (pre[C.REMISSION_KEY].astype(str) == POS_LABEL).astype(int).values
        for m in C.REPR_ORDER:
            dpath = C.repr_distance_path(label, m)
            if not dpath.exists():
                continue
            D = pd.read_csv(dpath, index_col=0)
            D.index = D.index.astype(str); D.columns = D.columns.astype(str)
            D = D.loc[pre.index, pre.index].values
            proba = loo_distance_knn(D, y, k=5)
            if proba is None:
                continue
            try:
                auc = roc_auc_score(y, proba)
            except ValueError:
                auc = np.nan
            rows.append({"group": label, "tissue": tissue, "disease": disease,
                         "representation": m, "n": int(pre.shape[0]), "roc_auc": auc})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def main():
    comp_clr, meta = build_cohort()
    log(f"pre-treatment cohort: {meta.shape[0]} samples, "
        f"{meta[C.PATIENT_KEY].nunique()} patients "
        f"(CD={(meta[C.DISEASE_KEY] == 'CD').sum()}, UC={(meta[C.DISEASE_KEY] == 'UC').sum()})")
    meta.to_csv(C.DATA / "predict_samples.csv")

    overall_rows, strat_rows, oof_rows, coef_frames = [], [], [], []
    for disease in C.DISEASES:
        res = run_cv(comp_clr, meta, disease)
        if res is None:
            continue
        sub, y, oof, Xd = res
        for model_name, proba in oof.items():
            strat_rows += stratified_metrics(disease, model_name, y, proba, meta)
            overall_rows.append({"disease": disease, "model": model_name,
                                 **evaluate_oof(y.values, proba.values)})
            for sid in sub.index:
                oof_rows.append({
                    "sample_id": sid, "disease": disease, "model": model_name,
                    "Site": meta.loc[sid, "Site"],
                    "Inflammation": meta.loc[sid, "Inflammation"],
                    "y_true": y.loc[sid], "p_nonremission": proba.loc[sid],
                })
        coef_frames.append(logreg_coefficients(Xd, y, disease))

    pd.DataFrame(overall_rows).to_csv(C.DATA / "predict_metrics_overall.csv", index=False)
    pd.DataFrame(strat_rows).to_csv(C.DATA / "predict_metrics_stratified.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(C.DATA / "predict_oof_predictions.csv", index=False)
    if coef_frames:
        pd.concat(coef_frames, ignore_index=True).to_csv(
            C.DATA / "predict_logreg_celltype_coef.csv", index=False)

    log("computing per-representation distance-KNN supplement ...")
    representation_knn_supplement().to_csv(
        C.DATA / "predict_representation_knn.csv", index=False)

    log("overall metrics:")
    if overall_rows:
        print(pd.DataFrame(overall_rows).round(3).to_string(index=False))
    log("done.")


if __name__ == "__main__":
    main()
