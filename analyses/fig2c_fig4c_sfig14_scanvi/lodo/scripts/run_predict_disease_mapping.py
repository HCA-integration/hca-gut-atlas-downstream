#!/usr/bin/env python
"""
Reference-map the PREDICT (Marc Rose 2025) lineage query objects onto the HGCA
v1 SCANVI models (``results/<lineage>/models_full_v1/scanvi_full_hgca_celltype_v1``)
and render the downstream mapping-pipeline plots.

This mirrors the proven Taurus transfer pipeline
(``notebooks/05_transfer_labels_taurus.ipynb`` cell 17 + the
``src/visualization/umap_transfer`` helpers) but is driven from a script so the
four PREDICT lineages can be mapped headlessly.

For each lineage it:
  1. Loads the PREDICT lineage h5ad (raw counts, gene-symbol var).
  2. Renames query genes symbol -> Ensembl using the model ``genes.csv`` and
     aligns/pads the raw counts to the model's exact 4000-gene order.
  3. Loads the SCANVI ``hgca_celltype_v1`` model with its bundled reference
     ``adata.h5ad`` and transfers fields onto the aligned query.
  4. Predicts ``hgca_celltype_v1`` labels + per-cell prediction entropy
     (uncertainty) for every query cell.
  5. Writes an annotated h5ad (obs predictions + uncertainty + X_scanvi latent),
     a predictions CSV, and downstream plots:
       - reference (grey) vs query (blue) UMAP in SCANVI latent space
       - query UMAP colored by predicted label
       - per-cell prediction-entropy histogram
  6. Aggregates a ``label_usage_comparison.json`` and ``run_summary.json``.

Outputs go to ``results/disease_mapping_predict_v1/`` by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# NOTE: cap native thread pools BEFORE importing numpy/torch/scvi. On macOS the
# default multi-threaded PyTorch/OpenMP CPU path deadlocks inside SCANVI
# predict/get_latent_representation (process sits in uninterruptible state at
# 0% CPU). Single-threaded BLAS + torch avoids the deadlock; forward passes are
# cheap enough that throughput stays fine.
_THREADS = os.environ.get("REFMAP_NUM_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from scipy import sparse

torch.set_num_threads(int(_THREADS))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import scvi  # noqa: E402
from scvi.model import SCANVI  # noqa: E402

scvi.settings.dl_num_workers = 0

# Force CPU everywhere (MPS is unreliable for these models; CUDA absent on laptop).
ACCELERATOR = os.environ.get("REFMAP_ACCELERATOR", "cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("predict_mapping")

LABEL_TYPE = "hgca_celltype_v1"

DEFAULT_PREDICT_DIR = Path(
    os.environ.get("PREDICT_LINEAGES", "")
)
LINEAGE_FILES = {
    "myeloid": "myeloid.h5ad",
    "stroma": "stroma.h5ad",
    "lymphoid": "lymphoid.h5ad",
    "epithelial": "epithelial.h5ad",
}

REFERENCE_COLOR = "#d3d3d3"
QUERY_COLOR = "#2596be"


# ---------------------------------------------------------------------------
# Gene alignment (mirrors 05_transfer_labels_taurus.ipynb)
# ---------------------------------------------------------------------------

def map_query_symbols_to_ensembl(adata_q: AnnData, genes_csv: Path) -> tuple[AnnData, int]:
    """Rename query var_names (gene symbols) to model Ensembl ids using genes.csv.

    Only symbols that map to exactly one Ensembl id are renamed.
    """
    gdf = pd.read_csv(genes_csv).dropna(subset=["gene_symbol"])
    gdf["sym_up"] = gdf["gene_symbol"].astype(str).str.upper()
    dup = gdf["sym_up"].duplicated(keep=False)
    gmap = dict(zip(gdf.loc[~dup, "sym_up"], gdf.loc[~dup, "gene_id"].astype(str)))

    sym_up = pd.Index(adata_q.var_names.astype(str)).str.upper()
    new_names, mapped = [], 0
    for old, s in zip(adata_q.var_names.astype(str), sym_up):
        ens = gmap.get(s)
        if ens is not None:
            new_names.append(ens)
            mapped += 1
        else:
            new_names.append(old)
    ad = adata_q.copy()
    ad.var_names = pd.Index(new_names)
    return ad, mapped


def align_and_pad_to_model_genes(
    adata_q: AnnData, model_genes: list[str], counts_layer: str = "counts"
) -> AnnData:
    """Subset/reorder query to the model's gene order, padding missing genes with 0.

    Operates on raw counts (X, and counts_layer if present). Returns a new
    AnnData whose var is exactly ``model_genes``.
    """
    q_vars = pd.Index(adata_q.var_names.astype(str))
    idx_map = pd.Series(range(len(q_vars)), index=q_vars)
    # For duplicated query var names keep the first occurrence.
    idx_map = idx_map[~idx_map.index.duplicated(keep="first")]

    take = [idx_map.get(g, None) for g in model_genes]
    exist_idx = [int(j) for j in take if j is not None]
    n_cells = adata_q.n_obs
    n_missing = sum(1 for j in take if j is None)

    def _reassemble(mat):
        is_sp = sparse.issparse(mat)
        if is_sp:
            sub = mat[:, exist_idx] if exist_idx else sparse.csr_matrix((n_cells, 0), dtype=mat.dtype)
            sub = sub.tocsc()
            zero_col = sparse.csc_matrix((n_cells, 1), dtype=mat.dtype)
        else:
            sub = mat[:, exist_idx] if exist_idx else np.zeros((n_cells, 0), dtype=mat.dtype)
            zero_col = np.zeros((n_cells, 1), dtype=mat.dtype)
        parts, k = [], 0
        for j in take:
            if j is not None:
                parts.append(sub[:, k:k + 1] if is_sp else sub[:, [k]])
                k += 1
            else:
                parts.append(zero_col)
        if is_sp:
            return sparse.hstack(parts, format="csr")
        return np.concatenate(parts, axis=1) if parts else sub

    X_aligned = _reassemble(adata_q.X)
    ad = AnnData(
        X=X_aligned,
        obs=adata_q.obs.copy(),
        var=pd.DataFrame(index=pd.Index(model_genes, name="gene_id")),
    )
    if counts_layer in adata_q.layers:
        ad.layers[counts_layer] = _reassemble(adata_q.layers[counts_layer])
    else:
        ad.layers[counts_layer] = X_aligned.copy()

    logger.info(
        "Aligned to model genes: matched=%d missing=%d total=%d",
        len(model_genes) - n_missing,
        n_missing,
        len(model_genes),
    )
    return ad


# ---------------------------------------------------------------------------
# Core mapping
# ---------------------------------------------------------------------------

def prepare_query_counts(adata_raw: AnnData, model_dir: Path) -> AnnData:
    """Build a model-gene-aligned raw-count query AnnData for SCANVI prediction.

    We do NOT run HVG selection: SCANVI needs raw counts in the model's exact
    gene space, so we map symbols->Ensembl on the full gene set and pad to the
    model genes. This is the canonical scArches-style query alignment.
    """
    genes_csv = model_dir / "genes.csv"
    if not genes_csv.exists():
        raise FileNotFoundError(f"Missing gene table: {genes_csv}")

    ad = adata_raw.copy()
    if "counts" not in ad.layers:
        ad.layers["counts"] = ad.X.copy()

    ad, n_mapped = map_query_symbols_to_ensembl(ad, genes_csv)
    logger.info("symbol->ensembl mapped=%d / %d query genes", n_mapped, ad.n_vars)

    model_genes = pd.read_csv(genes_csv)["gene_id"].astype(str).tolist()
    ad = align_and_pad_to_model_genes(ad, model_genes, counts_layer="counts")
    return ad


def register_query_on_model(model, adata_q: AnnData) -> None:
    """Set the unlabeled label + sample batch columns and transfer the model's
    field registry onto the query (extending novel categories) so the query is
    ready for ``model.predict``."""
    labels_reg = model.adata_manager.get_state_registry("labels")
    labels_key = labels_reg.original_key
    unlabeled_cat = labels_reg.get("unlabeled_category", "Unknown")
    try:
        batch_reg = model.adata_manager.get_state_registry("batch")
        trained_batch = batch_reg.original_key
    except KeyError:
        trained_batch = None

    adata_q.obs[labels_key] = unlabeled_cat
    adata_q.obs[labels_key] = adata_q.obs[labels_key].astype("category")
    if unlabeled_cat not in adata_q.obs[labels_key].cat.categories:
        adata_q.obs[labels_key] = adata_q.obs[labels_key].cat.add_categories([unlabeled_cat])

    if trained_batch is not None:
        if trained_batch not in adata_q.obs:
            if "sample" in adata_q.obs:
                adata_q.obs[trained_batch] = adata_q.obs["sample"].astype(str).values
            else:
                adata_q.obs[trained_batch] = "predict_query"
        adata_q.obs[trained_batch] = adata_q.obs[trained_batch].astype("category")

    new_mgr = model.adata_manager.transfer_fields(adata_q, extend_categories=True)
    model._register_manager_for_instance(new_mgr)


def _setup_and_predict(model, adata_q: AnnData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Register the query on ``model`` and return (hard_predictions, entropy, confidence).

    Runs hard + soft predict and computes per-cell prediction entropy and
    max-softmax confidence. ``confidence`` (max class probability, in [0, 1]) is
    comparable across models with different label-space sizes; entropy is not.
    """
    register_query_on_model(model, adata_q)
    predictions = np.asarray(model.predict(adata_q))
    prob_arr = np.asarray(model.predict(adata_q, soft=True))
    eps = 1e-10
    entropy = -np.sum(prob_arr * np.log(prob_arr + eps), axis=1)
    confidence = prob_arr.max(axis=1)
    return predictions, np.asarray(entropy, dtype=float), np.asarray(confidence, dtype=float)


def export_soft_counts_from_annotated(
    lineage: str,
    annotated_path: Path,
    output_dir: Path,
    *,
    chunk_size: int = 50_000,
) -> dict:
    """Aggregate soft scANVI label probabilities by sample without storing cells × labels."""
    model_dir = (
        PROJECT_ROOT
        / "results"
        / lineage
        / "models_full_v1"
        / f"scanvi_full_{LABEL_TYPE}"
    )
    ref_path = model_dir / "adata.h5ad"
    logger.info("[%s] loading reference model for soft-count export", lineage)
    adata_ref = anndata.read_h5ad(ref_path)
    model = SCANVI.load(
        str(model_dir), adata=adata_ref, accelerator=ACCELERATOR
    )
    query = anndata.read_h5ad(annotated_path, backed="r")
    sample_counts: pd.DataFrame | None = None
    confidence_parts = []
    weighted_state: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    label_order: pd.Index | None = None
    n_latent: int | None = None
    for start in range(0, query.n_obs, chunk_size):
        stop = min(start + chunk_size, query.n_obs)
        chunk = query[start:stop].to_memory()
        register_query_on_model(model, chunk)
        soft = model.predict(chunk, soft=True)
        if isinstance(soft, pd.DataFrame):
            probabilities = soft.to_numpy(dtype=float)
            labels = soft.columns.astype(str)
        else:
            probabilities = np.asarray(soft, dtype=float)
            labels_registry = model.adata_manager.get_state_registry("labels")
            labels = pd.Index(
                labels_registry.categorical_mapping
            ).astype(str)
        labels = pd.Index(labels)
        if label_order is None:
            label_order = labels
        elif not label_order.equals(labels):
            raise ValueError(
                f"{lineage}: soft-label order changed between chunks"
            )
        sample_ids = chunk.obs["sample_id"].astype(str)
        chunk_counts = (
            pd.DataFrame(probabilities, columns=labels)
            .assign(sample_id=sample_ids.to_numpy())
            .groupby("sample_id", observed=True)
            .sum()
        )
        sample_counts = (
            chunk_counts
            if sample_counts is None
            else sample_counts.add(chunk_counts, fill_value=0)
        )
        confidence_parts.append(
            pd.DataFrame(
                {
                    "sample_id": sample_ids.to_numpy(),
                    "max_probability": probabilities.max(axis=1),
                }
            )
            .groupby("sample_id", observed=True)["max_probability"]
            .agg(["sum", "count"])
        )
        latent = np.asarray(chunk.obsm["X_scanvi"], dtype=np.float64)
        n_latent = latent.shape[1]
        sample_array = sample_ids.to_numpy()
        for sample_id in pd.unique(sample_array):
            sample_mask = sample_array == sample_id
            sample_probability = probabilities[sample_mask]
            weight_sum = sample_probability.sum(axis=0)
            weight_sq_sum = (sample_probability**2).sum(axis=0)
            latent_sum = sample_probability.T @ latent[sample_mask]
            if sample_id in weighted_state:
                old_weight, old_weight_sq, old_latent = weighted_state[
                    sample_id
                ]
                weighted_state[sample_id] = (
                    old_weight + weight_sum,
                    old_weight_sq + weight_sq_sum,
                    old_latent + latent_sum,
                )
            else:
                weighted_state[sample_id] = (
                    weight_sum,
                    weight_sq_sum,
                    latent_sum,
                )
        logger.info(
            "[%s] soft-count progress: %s / %s cells",
            lineage,
            f"{stop:,}",
            f"{query.n_obs:,}",
        )
    if sample_counts is None:
        raise ValueError(f"{lineage}: annotated query has no cells")
    sample_counts = sample_counts.sort_index()
    sample_counts.columns = [
        f"{lineage}::{label}" for label in sample_counts.columns
    ]
    confidence = pd.concat(confidence_parts).groupby(level=0).sum()
    confidence["mean_max_probability"] = (
        confidence["sum"] / confidence["count"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    counts_path = output_dir / f"{lineage}_soft_counts_{LABEL_TYPE}.csv"
    confidence_path = (
        output_dir / f"{lineage}_soft_confidence_{LABEL_TYPE}.csv"
    )
    sample_counts.to_csv(counts_path)
    confidence[["count", "mean_max_probability"]].to_csv(confidence_path)
    if label_order is None or n_latent is None:
        raise ValueError(f"{lineage}: no soft state statistics were accumulated")
    state_samples = sample_counts.index.astype(str)
    weight_sum = np.stack(
        [weighted_state[sample][0] for sample in state_samples]
    )
    weight_sq_sum = np.stack(
        [weighted_state[sample][1] for sample in state_samples]
    )
    latent_sum = np.stack(
        [weighted_state[sample][2] for sample in state_samples]
    )
    state_path = (
        output_dir
        / f"{lineage}_soft_state_sufficient_stats_{LABEL_TYPE}.npz"
    )
    np.savez_compressed(
        state_path,
        sample_ids=state_samples.to_numpy(dtype=str),
        labels=label_order.to_numpy(dtype=str),
        weight_sum=weight_sum,
        weight_sq_sum=weight_sq_sum,
        latent_sum=latent_sum,
        n_latent=np.asarray(n_latent),
    )
    return {
        "lineage": lineage,
        "n_cells": int(query.n_obs),
        "n_samples": int(len(sample_counts)),
        "n_labels": int(sample_counts.shape[1]),
        "soft_counts": str(counts_path),
        "soft_confidence": str(confidence_path),
        "soft_state_sufficient_stats": str(state_path),
    }


def predict_lineage(
    lineage: str,
    query_path: Path,
    output_dir: Path,
    *,
    max_umap_cells: int,
    seed: int = 42,
) -> dict:
    """Map one PREDICT lineage onto its HGCA v1 SCANVI model and render plots."""
    t0 = time.time()
    model_dir = PROJECT_ROOT / "results" / lineage / "models_full_v1" / f"scanvi_full_{LABEL_TYPE}"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found: {model_dir}")

    logger.info("=" * 70)
    logger.info("LINEAGE %s | query=%s", lineage.upper(), query_path.name)
    logger.info("=" * 70)

    logger.info("Loading query %s ...", query_path)
    adata_raw = anndata.read_h5ad(query_path)
    logger.info("Query: %s cells x %s genes", f"{adata_raw.n_obs:,}", f"{adata_raw.n_vars:,}")

    # Align query raw counts to model gene space.
    adata_q = prepare_query_counts(adata_raw, model_dir)

    # Load model with its bundled reference (already in 4000-gene space).
    ref_path = model_dir / "adata.h5ad"
    logger.info("Loading reference adata + model ...")
    adata_ref = anndata.read_h5ad(ref_path)
    model = SCANVI.load(str(model_dir), adata=adata_ref, accelerator=ACCELERATOR)

    logger.info("Predicting labels on %s cells ...", f"{adata_q.n_obs:,}")
    predictions, entropy, _conf = _setup_and_predict(model, adata_q)
    n_unique = int(pd.Series(predictions).nunique())
    logger.info("Prediction done: %d unique labels, mean entropy=%.3f", n_unique, float(np.mean(entropy)))

    logger.info("Computing query SCANVI latent ...")
    q_latent = model.get_latent_representation(adata_q)

    # Attach to a lightweight annotated object (obs + latent, keep aligned X).
    adata_q.obs[f"predicted_{LABEL_TYPE}"] = pd.Categorical(np.asarray(predictions))
    adata_q.obs[f"uncertainty_{LABEL_TYPE}"] = np.asarray(entropy, dtype=float)
    adata_q.obsm["X_scanvi"] = q_latent

    output_dir.mkdir(parents=True, exist_ok=True)

    # Predictions CSV.
    preds_df = pd.DataFrame(
        {
            "cell_id": adata_q.obs_names.astype(str),
            f"predicted_{LABEL_TYPE}": np.asarray(predictions),
            f"uncertainty_{LABEL_TYPE}": np.asarray(entropy, dtype=float),
        }
    )
    preds_csv = output_dir / f"{lineage}_predictions_{LABEL_TYPE}.csv"
    preds_df.to_csv(preds_csv, index=False)
    logger.info("Wrote predictions CSV: %s", preds_csv.name)

    # ---- Downstream plots ----
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Reference latent (reference is the model's own registered adata).
    logger.info("Computing reference SCANVI latent (%s cells) ...", f"{adata_ref.n_obs:,}")
    ref_latent = model.get_latent_representation(adata_ref)

    rng = np.random.default_rng(seed)
    q_idx = np.arange(adata_q.n_obs)
    if adata_q.n_obs > max_umap_cells:
        q_idx = np.sort(rng.choice(adata_q.n_obs, size=max_umap_cells, replace=False))
        logger.info("Subsampling query to %s cells for UMAP", f"{max_umap_cells:,}")
    r_idx = np.arange(adata_ref.n_obs)
    if adata_ref.n_obs > max_umap_cells:
        r_idx = np.sort(rng.choice(adata_ref.n_obs, size=max_umap_cells, replace=False))

    umap_info = _render_umaps(
        lineage,
        ref_latent[r_idx],
        q_latent[q_idx],
        np.asarray(predictions)[q_idx],
        np.asarray(entropy)[q_idx],
        plots_dir,
        seed=seed,
    )

    # Entropy histogram (all cells).
    _render_entropy_hist(lineage, np.asarray(entropy), plots_dir)

    # Annotated h5ad (drop the padded matrix layers to keep it lean: keep obs + latent + counts).
    ann_path = output_dir / f"{lineage}_annotated_v1.h5ad"
    logger.info("Saving annotated h5ad: %s", ann_path.name)
    adata_q.write_h5ad(ann_path)

    label_counts = pd.Series(np.asarray(predictions)).value_counts()
    summary = {
        "lineage": lineage,
        "query_path": str(query_path),
        "model_dir": str(model_dir),
        "n_query_cells": int(adata_q.n_obs),
        "n_reference_cells": int(adata_ref.n_obs),
        "n_predicted_labels": n_unique,
        "mean_uncertainty": float(np.mean(entropy)),
        "median_uncertainty": float(np.median(entropy)),
        "predicted_label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "annotated_h5ad": str(ann_path),
        "predictions_csv": str(preds_csv),
        "plots": umap_info,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    logger.info("Lineage %s complete in %.1fs", lineage, summary["elapsed_seconds"])
    return summary


def _render_umaps(
    lineage: str,
    ref_latent: np.ndarray,
    q_latent: np.ndarray,
    q_pred: np.ndarray,
    q_entropy: np.ndarray,
    plots_dir: Path,
    *,
    seed: int,
) -> dict:
    import umap

    combined = np.vstack([ref_latent, q_latent])
    logger.info("Fitting UMAP on %s points (ref=%s + query=%s) ...",
                f"{combined.shape[0]:,}", f"{ref_latent.shape[0]:,}", f"{q_latent.shape[0]:,}")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed, verbose=False)
    coords = reducer.fit_transform(combined)
    n_ref = ref_latent.shape[0]
    ref_xy, q_xy = coords[:n_ref], coords[n_ref:]

    # 1) Reference vs query overlay.
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(ref_xy[:, 0], ref_xy[:, 1], c=REFERENCE_COLOR, s=0.5, alpha=0.6,
               label=f"HGCA reference ({n_ref:,})")
    ax.scatter(q_xy[:, 0], q_xy[:, 1], c=QUERY_COLOR, s=0.5, alpha=0.8,
               label=f"PREDICT query ({q_xy.shape[0]:,})")
    ax.set_title(f"{lineage.title()} - HGCA v1 SCANVI\nReference + PREDICT query projection",
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", markerscale=10, framealpha=0.9)
    fig.tight_layout()
    overlay = plots_dir / f"reference_query_umap_{lineage}_{LABEL_TYPE}.png"
    fig.savefig(overlay, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2) Query UMAP colored by predicted label.
    fig, ax = plt.subplots(figsize=(13, 10))
    labels = pd.Series(q_pred)
    cats = list(labels.value_counts().index)
    cmap = plt.get_cmap("tab20")
    for i, ct in enumerate(cats):
        m = labels.values == ct
        ax.scatter(q_xy[m, 0], q_xy[m, 1], s=0.5, alpha=0.8,
                   color=cmap(i % 20), label=f"{ct} ({int(m.sum()):,})")
    ax.set_title(f"{lineage.title()} - PREDICT query predicted {LABEL_TYPE}",
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=10,
              fontsize=7, ncol=1)
    fig.tight_layout()
    by_label = plots_dir / f"query_umap_predicted_label_{lineage}_{LABEL_TYPE}.png"
    fig.savefig(by_label, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3) Query UMAP colored by uncertainty.
    fig, ax = plt.subplots(figsize=(12, 10))
    sctr = ax.scatter(q_xy[:, 0], q_xy[:, 1], c=q_entropy, s=0.5, alpha=0.8, cmap="viridis")
    ax.set_title(f"{lineage.title()} - PREDICT query prediction entropy",
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sctr, ax=ax, label="Prediction entropy")
    fig.tight_layout()
    by_unc = plots_dir / f"query_umap_uncertainty_{lineage}_{LABEL_TYPE}.png"
    fig.savefig(by_unc, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "reference_query_umap": str(overlay),
        "query_umap_predicted_label": str(by_label),
        "query_umap_uncertainty": str(by_unc),
        "n_ref_umap": int(ref_latent.shape[0]),
        "n_query_umap": int(q_latent.shape[0]),
    }


def _render_entropy_hist(lineage: str, entropy: np.ndarray, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(entropy, bins=60, color=QUERY_COLOR, alpha=0.85)
    ax.axvline(float(np.median(entropy)), color="black", linestyle="--", linewidth=1,
               label=f"median={np.median(entropy):.2f}")
    ax.set_title(f"{lineage.title()} - PREDICT prediction entropy ({LABEL_TYPE})")
    ax.set_xlabel("Prediction entropy"); ax.set_ylabel("cells")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / f"entropy_hist_{lineage}_{LABEL_TYPE}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lineages", nargs="+", default=list(LINEAGE_FILES),
                    choices=list(LINEAGE_FILES))
    ap.add_argument("--predict-dir", type=Path, default=DEFAULT_PREDICT_DIR)
    ap.add_argument("--output-dir", type=Path,
                    default=PROJECT_ROOT / "results" / "disease_mapping_predict_v1")
    ap.add_argument("--max-umap-cells", type=int, default=200_000,
                    help="Cap ref/query cells used in the UMAP fit (predictions always use all cells).")
    ap.add_argument("--subsample-query", type=int, default=0,
                    help="If >0, randomly subsample the query to this many cells (smoke tests).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--soft-counts-only",
        action="store_true",
        help="Aggregate soft probabilities from existing annotated h5ads.",
    )
    ap.add_argument("--soft-chunk-size", type=int, default=50_000)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.soft_counts_only:
        summary_path = args.output_dir / "soft_counts_summary.json"
        existing = (
            json.loads(summary_path.read_text())
            if summary_path.exists()
            else []
        )
        summaries_by_lineage = {
            item["lineage"]: item for item in existing
        }
        for lineage in args.lineages:
            annotated = args.output_dir / f"{lineage}_annotated_v1.h5ad"
            if not annotated.exists():
                logger.error("Annotated query missing: %s", annotated)
                continue
            summaries_by_lineage[lineage] = (
                export_soft_counts_from_annotated(
                    lineage,
                    annotated,
                    args.output_dir,
                    chunk_size=args.soft_chunk_size,
                )
            )
        summaries = [
            summaries_by_lineage[lineage]
            for lineage in LINEAGE_FILES
            if lineage in summaries_by_lineage
        ]
        summary_path.write_text(json.dumps(summaries, indent=2))
        return

    run_summaries = []
    for lineage in args.lineages:
        query_path = args.predict_dir / LINEAGE_FILES[lineage]
        if not query_path.exists():
            logger.error("Query file missing, skipping %s: %s", lineage, query_path)
            continue
        if args.subsample_query > 0:
            adata_raw = anndata.read_h5ad(query_path)
            if adata_raw.n_obs > args.subsample_query:
                rng = np.random.default_rng(args.seed)
                keep = np.sort(rng.choice(adata_raw.n_obs, size=args.subsample_query, replace=False))
                adata_raw = adata_raw[keep].copy()
            tmp = args.output_dir / f"_smoke_{lineage}.h5ad"
            adata_raw.write_h5ad(tmp)
            query_path = tmp
        try:
            summary = predict_lineage(
                lineage, query_path, args.output_dir,
                max_umap_cells=args.max_umap_cells, seed=args.seed,
            )
            run_summaries.append(summary)
        except Exception:  # keep going across lineages; surface the traceback
            logger.exception("Lineage %s FAILED", lineage)
            run_summaries.append({"lineage": lineage, "status": "failed"})

    # Aggregate label-usage + run summary.
    label_usage = {}
    for s in run_summaries:
        if s.get("predicted_label_counts") is None:
            continue
        key = f"{s['lineage']}_{LABEL_TYPE}"
        label_usage[key] = {
            "lineage": s["lineage"],
            "label_type": LABEL_TYPE,
            "n_cells": s["n_query_cells"],
            "n_predicted_types": s["n_predicted_labels"],
            "predicted_labels": s["predicted_label_counts"],
        }
    (args.output_dir / "label_usage_comparison.json").write_text(json.dumps(label_usage, indent=2))
    (args.output_dir / "run_summary.json").write_text(json.dumps(run_summaries, indent=2))
    logger.info("All done. Summary -> %s", args.output_dir / "run_summary.json")


if __name__ == "__main__":
    main()
