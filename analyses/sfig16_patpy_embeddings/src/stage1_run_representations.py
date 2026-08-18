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
SAMPLECLR_EPOCHS = int(os.environ.get("SAMPLECLR_EPOCHS", "40"))
SAMPLECLR_LATENT = int(os.environ.get("SAMPLECLR_LATENT", "32"))
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
def _finalize_group_adata(sub):
    """Drop tiny samples, annotate state, HVG + PCA."""
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


def build_group_adata(adata_norm, tissue: str, disease: str):
    """Subset to a tissue x disease group (+ same-tissue Healthy), HVG + PCA."""
    obs = adata_norm.obs
    keep = (obs[C.TISSUE_KEY].astype(str) == tissue) & (
        obs[C.DISEASE_KEY].astype(str).isin([disease, "Healthy"])
    )
    return _finalize_group_adata(adata_norm[keep].copy())


def build_special_group_adata(adata_norm, label: str):
    """Build a pooled multi-segment cohort from C.SPECIAL_GROUPS."""
    if label not in C.SPECIAL_GROUPS:
        raise KeyError(f"Unknown special group {label!r}")
    spec = C.SPECIAL_GROUPS[label]
    obs = adata_norm.obs
    disease = obs[C.DISEASE_KEY].astype(str)
    tissue = obs[C.TISSUE_KEY].astype(str)
    treatment = obs[C.TREATMENT_KEY].astype(str)
    keep = tissue.isin(spec["tissues"]) & disease.isin(spec["diseases"])
    if spec.get("treatments") is not None:
        # Keep Healthy regardless of treatment; restrict disease samples to
        # the requested treatment levels (e.g. Pre only).
        keep = keep & (
            (disease == "Healthy")
            | treatment.isin(spec["treatments"])
        )
    # Subsample on boolean mask indices *before* materializing a copy, so
    # large pooled cohorts (e.g. ~500k cells) do not OOM.
    max_cells = int(os.environ.get("STAGE1_MAX_CELLS_PER_SAMPLE", "1500"))
    positions = np.flatnonzero(np.asarray(keep))
    sample_ids = obs[C.SAMPLE_KEY].astype(str).to_numpy()[positions]
    rng = np.random.default_rng(RANDOM_SEED)
    chosen = []
    for sample in np.unique(sample_ids):
        idx = positions[sample_ids == sample]
        if len(idx) > max_cells:
            idx = rng.choice(idx, size=max_cells, replace=False)
        chosen.append(idx)
    chosen = np.concatenate(chosen)
    log(f"  special-group selecting {len(chosen)} cells "
        f"(max {max_cells}/sample; from {int(keep.sum())} eligible)")
    sub = adata_norm[chosen].copy()
    return _finalize_group_adata(sub)


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


def _subsample_cells_per_sample(sub, max_cells: int, seed: int = RANDOM_SEED):
    """Deterministic per-sample cell cap (shared by PILOT / MixMIL / SampleCLR)."""
    if max_cells <= 0:
        return sub
    vc = sub.obs[C.SAMPLE_KEY].astype(str).value_counts()
    if (vc <= max_cells).all():
        return sub
    rng = np.random.default_rng(seed)
    keep_idx = []
    for _, idx in sub.obs.groupby(C.SAMPLE_KEY, observed=True).indices.items():
        idx = np.asarray(idx, dtype=int)
        if len(idx) > max_cells:
            idx = rng.choice(idx, size=max_cells, replace=False)
        keep_idx.append(idx)
    out = sub[np.concatenate(keep_idx)].copy()
    log(f"    subsampled to {out.n_obs} cells (max {max_cells}/sample)")
    return out


def _pilot_wasserstein_distances(sub, *, regulizer: float = 0.2, metric: str = "cosine"):
    """PILOT-faithful Wasserstein distances without calling segfault-prone pilotpy.

    Reimplements pilotpy.tl.wasserstein_distance core steps:
    Dirichlet-smoothed cell-type proportions, cosine cost on cell-type median
    embeddings, and classical OT (ot.emd2) between samples.
    """
    import ot
    import scipy.spatial

    sample_col = C.SAMPLE_KEY
    ct_col = C.CELLTYPE_KEY
    emb = np.asarray(sub.obsm["X_pca"], dtype=float)
    samples = pd.Index(sub.obs[sample_col].astype(str).unique())
    cells = pd.Index(sub.obs[ct_col].astype(str).unique())
    if len(samples) < 2 or len(cells) < 2:
        raise RuntimeError(
            f"PILOT needs >=2 samples and cell types; got "
            f"{len(samples)} samples / {len(cells)} cell types"
        )

    ct = sub.obs[ct_col].astype(str).to_numpy()
    sid = sub.obs[sample_col].astype(str).to_numpy()
    n_cells_total = len(ct)
    prior = np.array(
        [(ct == c).sum() / max(n_cells_total - 1, 1) for c in cells], dtype=float
    )
    prior = prior * float(regulizer)

    proportions = {}
    for sample in samples:
        vc = pd.Series(ct[sid == sample]).value_counts()
        vector = np.zeros(len(cells), dtype=float)
        for cell_name, count in vc.items():
            vector[cells.get_loc(cell_name)] = float(count)
        proportions[sample] = (vector + prior) / (vector.sum() + prior.sum())

    centroids = np.vstack(
        [np.median(emb[ct == c], axis=0) for c in cells]
    )
    cost = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(centroids, metric=metric)
    )
    cost = cost / max(float(np.nanmax(cost)), 1e-12)

    n = len(samples)
    emd = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(
                ot.emd2(proportions[samples[i]], proportions[samples[j]], cost)
            )
            emd[i, j] = emd[j, i] = d
    return labelled_distance_df(emd, samples)


def run_pilot(sub):
    # pilotpy currently segfaults in this environment (even on tiny synthetic
    # data). Use a faithful OT reimplementation that matches PILOT's published
    # Wasserstein sample distances (proportions + cell-type cost + emd2).
    max_cells = int(os.environ.get("PILOT_MAX_CELLS_PER_SAMPLE", "800"))
    sub = _subsample_cells_per_sample(sub, max_cells)
    log(f"    PILOT via OT fallback ({sub.n_obs} cells, "
        f"{sub.obs[C.SAMPLE_KEY].nunique()} samples, "
        f"{sub.obs[C.CELLTYPE_KEY].nunique()} cell types)")
    return _pilot_wasserstein_distances(sub)


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
    max_cells = int(os.environ.get("MIXMIL_MAX_CELLS_PER_SAMPLE", "600"))
    sub = _subsample_cells_per_sample(sub, max_cells)
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


def run_sampleclr_remission(sub):
    """SampleCLR with Remission vs Non_Remission supervised stage-2 training.

    Healthy / unlabeled samples are held out of the supervised split but still
    receive projected embeddings for the distance matrix.
    """
    import torch
    from sampleclr.contrastive_model import ContrastiveModel
    from sampleclr.datasets import SamplesDataset
    from sampleclr.utils import get_sample_representations
    from sklearn.metrics.pairwise import euclidean_distances

    max_cells = int(os.environ.get("SAMPLECLR_MAX_CELLS_PER_SAMPLE", "256"))
    sub = _subsample_cells_per_sample(sub, max_cells)

    rem = sub.obs[C.REMISSION_KEY].astype(str).str.strip()
    rem = rem.where(rem.isin(["Remission", "Non_Remission"]))
    sub.obs["_sampleclr_remission"] = rem

    meta = (
        sub.obs.drop_duplicates(subset=[C.SAMPLE_KEY])
        .set_index(C.SAMPLE_KEY)
    )
    labeled = meta.index[meta["_sampleclr_remission"].notna()].astype(str).tolist()
    if len(labeled) < 6:
        raise RuntimeError(
            f"SampleCLR-remission needs >=6 labeled samples; found {len(labeled)}"
        )
    rng = np.random.default_rng(RANDOM_SEED)
    labeled = list(rng.permutation(labeled))
    n_val = max(2, int(round(0.2 * len(labeled))))
    val_ids = labeled[:n_val]
    train_ids = labeled[n_val:]
    if len(train_ids) < 4:
        train_ids = labeled[:-2]
        val_ids = labeled[-2:]

    device = torch.device("cpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Official SampleCLR can hit float64 empty-label issues on MPS; stay CPU.
        device = torch.device("cpu")

    model = ContrastiveModel(
        adata=sub,
        sample_key=C.SAMPLE_KEY,
        tasks={"classification": ["_sampleclr_remission"]},
        layer="X_pca",
        device=device,
        n_cells_per_sample=[32, 128],
        train_ids=train_ids,
        val_ids=val_ids,
        batch_size=int(min(16, max(4, len(train_ids)))),
        num_layers=3,
        hidden_size=32,
        learning_rate_feature=1e-3,
        learning_rate_discriminator=1e-5,
        weight_decay=1e-4,
        n_aggregator_heads=4,
        aggregator_num_layers=2,
        aggregator_hidden_size=32,
        aggregator_activation="relu",
        output_dim=SAMPLECLR_LATENT,
        classifier_num_layers=2,
        classifier_hidden_size=16,
        use_normalization=False,
        feature_normalization="BatchNorm",
        aggregator_normalization="LayerNorm",
        contrastive_loss="InfoNCECauchy",
        contrastive_loss_temperature=0.1,
        num_warmup_epochs_stage1=3,
        num_warmup_epochs_stage2=3,
        verbose=False,
        early_stopping_patience=8,
    )
    log(
        f"    SampleCLR-remission train={len(train_ids)} val={len(val_ids)} "
        f"epochs={SAMPLECLR_EPOCHS} latent={SAMPLECLR_LATENT}"
    )
    model.train(
        num_epochs_stage1=SAMPLECLR_EPOCHS,
        num_epochs_stage2=SAMPLECLR_EPOCHS,
        two_stages=True,
        stage_2="joint",
        verbose=False,
        stage1_val_metric="loss",
        stage2_val_metric="loss",
    )

    all_samples = meta.index.astype(str).tolist()
    ds = SamplesDataset(
        data=sub,
        unique_categories=all_samples,
        sample_col=C.SAMPLE_KEY,
        layer="X_pca",
    )
    reps = get_sample_representations(
        model.projector, model.aggregator, ds, subset_size=128, device=str(device)
    )
    dist = euclidean_distances(np.asarray(reps, dtype=float))
    return labelled_distance_df(dist, all_samples)


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
    "sampleclr_remission": run_sampleclr_remission,
    "random": run_random,
}


# --------------------------------------------------------------------------
def _run_methods_on_subset(sub, label: str) -> dict:
    """Fit all (or STAGE1_METHODS) representations on an already-built subset."""
    status = {"group": label, "n_samples": 0, "n_cells": 0}
    if sub is None:
        log("  skipped (too few samples/cells)")
        status["skipped"] = True
        return status

    meta = sample_metadata(sub)
    meta.to_csv(C.group_meta_path(label))
    status["n_samples"] = int(meta.shape[0])
    status["n_cells"] = int(sub.n_obs)
    log(
        f"  {meta.shape[0]} samples, {sub.n_obs} cells; states: "
        f"{meta['state'].value_counts().to_dict()}"
    )
    if C.TISSUE_KEY in meta.columns:
        log(f"  tissues: {meta[C.TISSUE_KEY].astype(str).value_counts().to_dict()}")

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


def process_group(adata_norm, tissue: str, disease: str, label: str) -> dict:
    log(f"== group {label} ==")
    sub = build_group_adata(adata_norm, tissue, disease)
    return _run_methods_on_subset(sub, label)


def process_special_group(adata_norm, label: str) -> dict:
    log(f"== special group {label} ==")
    log(f"  {C.SPECIAL_GROUPS[label]['description']}")
    sub = build_special_group_adata(adata_norm, label)
    return _run_methods_on_subset(sub, label)


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
    if C.LABELSET == "pangi":
        adata.obs = C.apply_pangi_labels(adata.obs)
    log(f"  loaded {adata.shape} in {time.time() - t0:.0f}s")

    # Keep only cells with a usable tissue, disease and cell-type label.
    obs = adata.obs
    keep = (
        obs[C.TISSUE_KEY].astype(str).isin(C.TISSUES)
        & obs[C.CELLTYPE_KEY].notna()
        & (obs[C.CELLTYPE_KEY].astype(str) != "nan")
    )
    # For pooled special cohorts, restrict to the target cells *before*
    # normalize/log1p so we do not materialize the full ~1M-cell matrix.
    if only is not None and only in C.SPECIAL_GROUPS:
        spec = C.SPECIAL_GROUPS[only]
        disease = obs[C.DISEASE_KEY].astype(str)
        tissue = obs[C.TISSUE_KEY].astype(str)
        treatment = obs[C.TREATMENT_KEY].astype(str)
        keep = keep & tissue.isin(spec["tissues"]) & disease.isin(spec["diseases"])
        if spec.get("treatments") is not None:
            keep = keep & (
                (disease == "Healthy") | treatment.isin(spec["treatments"])
            )
        log(f"  pre-filter special cohort {only}: {int(keep.sum())} cells")
    adata = adata[keep].copy()
    log(f"  kept {adata.n_obs} cells with usable tissue/celltype")

    # Normalise + log1p once (raw counts -> log-normalised in .X).
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    statuses = []
    if only is not None and only in C.SPECIAL_GROUPS:
        statuses.append(process_special_group(adata, only))
    elif only is not None:
        matched = [g for g in C.GROUPS if g[2] == only]
        if not matched:
            raise SystemExit(
                f"Unknown group {only!r}. "
                f"Choose from {[g[2] for g in C.GROUPS] + list(C.SPECIAL_GROUPS)}"
            )
        tissue, disease, label = matched[0]
        statuses.append(process_group(adata, tissue, disease, label))
    else:
        for tissue, disease, label in C.GROUPS:
            statuses.append(process_group(adata, tissue, disease, label))

    if not os.environ.get("STAGE1_METHODS"):
        status_df = pd.DataFrame(statuses)
        status_path = C.DATA / (
            f"stage1_run_status_{only}.csv" if only else "stage1_run_status.csv"
        )
        status_df.to_csv(status_path, index=False)
        log(f"wrote run status -> {status_path}")
    log("done.")


if __name__ == "__main__":
    main()
