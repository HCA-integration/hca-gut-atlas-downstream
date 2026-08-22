#!/usr/bin/env python3
"""Stage 1: compute patpy sample representations per tissue x disease group.

For each (tissue, disease) group (Ileum/Colon/Rectum x CD/UC, with same-tissue
Healthy samples added as reference anchors) we:

  1. subset cells, normalise + log1p, pick HVGs, run PCA (X_pca);
  2. run the CPU-feasible patpy representation methods:
       cell-type composition, pseudobulk, CT pseudobulk, MOFA, GloScope, PILOT,
       random-vector baseline;
  3. export, per group x method, a labelled sample x sample distance matrix and
     a 2-D MDS embedding, plus per-group sample metadata and the CLR cell-type
     composition matrix used downstream by the predictor.

Every method is wrapped so one failing tool does not abort the whole run; a
run log records which method x group combinations succeeded.

Usage:
    python stage1_run_representations.py            # all groups
    python stage1_run_representations.py Ileum_CD   # a single group (debug)
"""

from __future__ import annotations

import os
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.manifold import MDS

import _common as C

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

N_HVG = 2000
N_PCS = 30
GLOSCOPE_K = 15
MOFA_FACTORS = 10
MIXMIL_EPOCHS = 300   # reduced from the 2000 default for CPU feasibility on ~10^5-cell bags
RANDOM_SEED = 0


# --------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[stage1] {msg}", flush=True)


def mds_embed(dist: np.ndarray, seed: int = RANDOM_SEED) -> np.ndarray:
    """Classical-ish 2-D embedding of a precomputed distance matrix."""
    dist = np.asarray(dist, dtype=float)
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    if not np.isfinite(dist).all():
        dist = np.nan_to_num(dist, nan=np.nanmax(dist[np.isfinite(dist)]))
    n = dist.shape[0]
    if n < 3:
        return np.zeros((n, 2))
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=seed,
              normalized_stress="auto", n_init=4, max_iter=300)
    return mds.fit_transform(dist)


def labelled_distance_df(dist, samples) -> pd.DataFrame:
    """Return a sample x sample distance DataFrame from a distance array.

    Uses the value returned by `calculate_distance_matrix()` (works across
    patpy versions, which store it either in adata.uns or self._distances).
    """
    dist = np.asarray(dist, dtype=float)
    samples = [str(s) for s in samples]
    return pd.DataFrame(dist, index=samples, columns=samples)


# --------------------------------------------------------------------------
def build_group_adata(adata_norm, tissue: str, disease: str):
    """Subset to a tissue x disease group (+ same-tissue Healthy), HVG + PCA."""
    obs = adata_norm.obs
    keep = (obs[C.TISSUE_KEY].astype(str) == tissue) & (
        obs[C.DISEASE_KEY].astype(str).isin([disease, "Healthy"])
    )
    sub = adata_norm[keep].copy()

    # Drop samples with too few cells.
    vc = sub.obs[C.SAMPLE_KEY].value_counts()
    good_samples = vc[vc >= C.MIN_CELLS_PER_SAMPLE].index
    sub = sub[sub.obs[C.SAMPLE_KEY].isin(good_samples)].copy()
    if sub.n_obs == 0 or sub.obs[C.SAMPLE_KEY].nunique() < 3:
        return None

    sub.obs[C.SAMPLE_KEY] = sub.obs[C.SAMPLE_KEY].astype(str)
    sub.obs[C.CELLTYPE_KEY] = sub.obs[C.CELLTYPE_KEY].astype(str)

    # Cell-level disease state (needed by PILOT's `status` argument).
    sub.obs["state"] = [
        C.assign_state(d, t, r)
        for d, t, r in zip(
            sub.obs[C.DISEASE_KEY].astype(str),
            sub.obs.get(C.TREATMENT_KEY, pd.Series(index=sub.obs.index)).astype(str),
            sub.obs.get(C.REMISSION_KEY, pd.Series(index=sub.obs.index)).astype(str),
        )
    ]

    # HVG + PCA on the group (X is already log-normalised).
    try:
        sc.pp.highly_variable_genes(sub, n_top_genes=min(N_HVG, sub.n_vars - 1))
        sub_hvg = sub[:, sub.var["highly_variable"]].copy()
    except Exception:
        sub_hvg = sub
    sc.pp.scale(sub_hvg, max_value=10)
    n_pcs = int(min(N_PCS, sub_hvg.n_obs - 1, sub_hvg.n_vars - 1))
    sc.tl.pca(sub_hvg, n_comps=n_pcs, svd_solver="arpack")
    sub.obsm["X_pca"] = sub_hvg.obsm["X_pca"]
    return sub


def sample_metadata(sub) -> pd.DataFrame:
    """One row per sample with covariates, disease state, severity, n_cells."""
    obs = sub.obs.copy()
    obs[C.SAMPLE_KEY] = obs[C.SAMPLE_KEY].astype(str)
    n_cells = obs.groupby(C.SAMPLE_KEY, observed=True).size().rename("n_cells")

    cols = [c for c in C.META_COLS if c in obs.columns]
    meta = obs.groupby(C.SAMPLE_KEY, observed=True)[cols].first()
    meta = meta.join(n_cells)

    meta["state"] = [
        C.assign_state(d, t, r)
        for d, t, r in zip(meta[C.DISEASE_KEY].astype(str),
                           meta.get(C.TREATMENT_KEY, pd.Series(index=meta.index)).astype(str),
                           meta.get(C.REMISSION_KEY, pd.Series(index=meta.index)).astype(str))
    ]
    meta["state_severity"] = meta["state"].map(C.SEVERITY)
    meta.index.name = C.SAMPLE_KEY
    return meta


# --------------------------------------------------------------------------
# Representation runners. Each returns a labelled distance DataFrame or raises.
# --------------------------------------------------------------------------
def run_composition(sub):
    # Native patpy CLR-transformed cell-type composition (Aitchison geometry),
    # matching the compositional PCA in fig_cd_ileum_compositional_pca. Cell
    # groups are the predicted HGCA v1 labels.
    from patpy.tl.sample_representation import CellGroupComposition
    m = CellGroupComposition(C.SAMPLE_KEY, C.CELLTYPE_KEY, apply_clr=True,
                             pseudocount=1)
    m.prepare_anndata(sub)
    d = m.calculate_distance_matrix(dist="euclidean")
    comp = m.sample_representation.copy()
    comp.index = [str(i) for i in comp.index]
    return labelled_distance_df(d, m.samples), comp


def run_pseudobulk(sub):
    from patpy.tl.sample_representation import Pseudobulk
    m = Pseudobulk(C.SAMPLE_KEY, C.CELLTYPE_KEY, layer="X_pca")
    m.prepare_anndata(sub)
    d = m.calculate_distance_matrix(aggregate="mean", dist="euclidean")
    return labelled_distance_df(d, m.samples)


def run_ct_pseudobulk(sub):
    from patpy.tl.sample_representation import GroupedPseudobulk
    m = GroupedPseudobulk(C.SAMPLE_KEY, C.CELLTYPE_KEY, layer="X_pca")
    m.prepare_anndata(sub)
    d = m.calculate_distance_matrix(aggregate="mean", dist="euclidean")
    return labelled_distance_df(d, m.samples)


def run_mofa(sub):
    from patpy.tl.sample_representation import MOFA
    # Cell types as views (the SPARE-paper MOFA setup); fall back to a single
    # aggregated view if the multi-view model fails to converge / errors.
    n_samples = sub.obs[C.SAMPLE_KEY].nunique()
    n_factors = int(min(MOFA_FACTORS, max(2, n_samples - 1)))
    for aggregate_ct in (True, False):
        try:
            m = MOFA(C.SAMPLE_KEY, C.CELLTYPE_KEY, layer="X_pca",
                     n_factors=n_factors, aggregate_cell_types=aggregate_ct,
                     quiet=True, verbose=False, seed=RANDOM_SEED)
            m.prepare_anndata(sub)
            d = m.calculate_distance_matrix(dist="euclidean")
            return labelled_distance_df(d, m.samples)
        except Exception as e:  # noqa: BLE001
            log(f"    MOFA aggregate_cell_types={aggregate_ct} failed: {e}")
    raise RuntimeError("MOFA failed in both view configurations")


def run_gloscope(sub):
    from patpy.tl.sample_representation import GloScope_py
    m = GloScope_py(C.SAMPLE_KEY, C.CELLTYPE_KEY, layer="X_pca",
                    k=GLOSCOPE_K, use_gpu=False, seed=RANDOM_SEED)
    m.prepare_anndata(sub)
    d = m.calculate_distance_matrix()
    return labelled_distance_df(d, m.samples)


def run_pilot(sub):
    from patpy.tl.sample_representation import PILOT
    # pilotpy expects adata.obsm[emb_matrix] to be a DataFrame (it reads
    # `.columns`), so expose the PCA embedding as a labelled frame.
    pcs = sub.obsm["X_pca"]
    sub.obsm["X_pca_df"] = pd.DataFrame(
        np.asarray(pcs), index=sub.obs_names,
        columns=[f"PC{i + 1}" for i in range(pcs.shape[1])],
    )
    m = PILOT(C.SAMPLE_KEY, C.CELLTYPE_KEY, sample_state_col="state",
              layer="X_pca_df", seed=RANDOM_SEED)
    m.prepare_anndata(sub)
    d = m.calculate_distance_matrix()
    return labelled_distance_df(d, m.samples)


def run_mixmil(sub):
    # Supervised attention-based multi-instance mixed model (Engelmann et al.
    # 2024). Each donor is a bag of per-cell X_pca features; the model learns
    # which cells drive a donor-level target. We supervise on Disease status
    # (CD/UC vs same-tissue Healthy) -- a target orthogonal to treatment
    # response, so the resulting donor embedding stays a fair representation
    # (not a response oracle) for the downstream trajectory / prediction stages.
    # The attention-weighted donor embedding (get_sample_representations) is
    # turned into a Euclidean donor x donor distance matrix, exactly like the
    # other latent-space methods.
    from patpy.tl.supervised import MixMIL
    disease = sub.obs[C.DISEASE_KEY].astype(str)
    if disease.nunique() < 2:
        raise RuntimeError(
            f"MixMIL needs >=2 disease classes, found {disease.unique().tolist()}")
    m = MixMIL(
        C.SAMPLE_KEY, label_keys=[C.DISEASE_KEY], tasks=["classification"],
        cell_group_key=C.CELLTYPE_KEY, layer="X_pca",
        n_epochs=MIXMIL_EPOCHS, seed=RANDOM_SEED,
    )
    m.prepare_anndata(sub)
    d = m.calculate_distance_matrix(dist="euclidean")
    return labelled_distance_df(d, m.samples)


def run_random(sub):
    from patpy.tl.sample_representation import RandomVector
    np.random.seed(RANDOM_SEED)
    m = RandomVector(C.SAMPLE_KEY, C.CELLTYPE_KEY, latent_dim=30, seed=RANDOM_SEED)
    m.prepare_anndata(sub)
    d = m.calculate_distance_matrix()
    return labelled_distance_df(d, m.samples)


RUNNERS = {
    "composition": run_composition,
    "pseudobulk": run_pseudobulk,
    "ct_pseudobulk": run_ct_pseudobulk,
    "mofa": run_mofa,
    "gloscope": run_gloscope,
    "pilot": run_pilot,
    "mixmil": run_mixmil,
    "random": run_random,
}


# --------------------------------------------------------------------------
def process_group(adata_norm, tissue: str, disease: str, label: str) -> dict:
    log(f"== group {label} ==")
    sub = build_group_adata(adata_norm, tissue, disease)
    status = {"group": label, "n_samples": 0, "n_cells": 0}
    if sub is None:
        log(f"  skipped (too few samples/cells)")
        status["skipped"] = True
        return status

    meta = sample_metadata(sub)
    meta.to_csv(C.group_meta_path(label))
    status["n_samples"] = int(meta.shape[0])
    status["n_cells"] = int(sub.n_obs)
    log(f"  {meta.shape[0]} samples, {sub.n_obs} cells; states: "
        f"{meta['state'].value_counts().to_dict()}")

    canonical = list(meta.index)

    only_methods = os.environ.get("STAGE1_METHODS")
    runners = RUNNERS
    if only_methods:
        wanted = {m.strip() for m in only_methods.split(",")}
        runners = {k: v for k, v in RUNNERS.items() if k in wanted}

    # Resume mode: skip (group, method) combos whose distance matrix already
    # exists, so a run that died (e.g. memory pressure) can simply be re-run.
    skip_existing = os.environ.get("STAGE1_SKIP_EXISTING")

    for key, runner in runners.items():
        if skip_existing and C.repr_distance_path(label, key).exists():
            status[key] = "skip(exists)"
            log(f"  [skip] {key:14s} (output exists)")
            continue
        t0 = time.time()
        try:
            out = runner(sub)
            if key == "composition":
                dist_df, comp = out
                comp = comp.reindex(canonical)
                comp.to_csv(C.group_composition_path(label))
            else:
                dist_df = out
            dist_df = dist_df.reindex(index=canonical, columns=canonical)
            dist_df.to_csv(C.repr_distance_path(label, key))
            emb = mds_embed(dist_df.values)
            pd.DataFrame(emb, index=canonical, columns=["MDS1", "MDS2"]).to_csv(
                C.repr_embedding_path(label, key, "mds")
            )
            status[key] = "ok"
            log(f"  [ok]   {key:14s} ({time.time() - t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            status[key] = f"FAIL: {e}"
            log(f"  [FAIL] {key:14s} {e}")
            log(traceback.format_exc().splitlines()[-1])
    return status


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None

    if not C.TAURUS_H5AD.is_file():
        raise SystemExit(
            "Set TAURUS_H5AD to the TAURUS h5ad with HGCA-transferred labels."
        )
    log(f"loading {C.TAURUS_H5AD.name} ...")
    t0 = time.time()
    adata = sc.read_h5ad(C.TAURUS_H5AD)
    if C.HGCA_V1_REMAP_SIDECAR.is_file():
        adata.obs = C.apply_remapped_hgca_v1_labels(adata.obs)
    log(f"  loaded {adata.shape} in {time.time() - t0:.0f}s")

    # Keep only cells with a usable tissue, disease and cell-type label.
    obs = adata.obs
    keep = (
        obs[C.TISSUE_KEY].astype(str).isin(C.TISSUES)
        & obs[C.CELLTYPE_KEY].notna()
        & (obs[C.CELLTYPE_KEY].astype(str) != "nan")
    )
    adata = adata[keep].copy()
    log(f"  kept {adata.n_obs} cells with usable tissue/celltype")

    # Normalise + log1p once (raw counts -> log-normalised in .X).
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    statuses = []
    for tissue, disease, label in C.GROUPS:
        if only is not None and label != only:
            continue
        statuses.append(process_group(adata, tissue, disease, label))

    if not os.environ.get("STAGE1_METHODS"):
        status_df = pd.DataFrame(statuses)
        status_path = C.DATA / "stage1_run_status.csv"
        status_df.to_csv(status_path, index=False)
        log(f"wrote run status -> {status_path}")
    log("done.")


if __name__ == "__main__":
    main()
