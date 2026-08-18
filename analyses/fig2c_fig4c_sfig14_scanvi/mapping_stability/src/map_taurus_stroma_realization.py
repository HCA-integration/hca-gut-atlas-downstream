#!/usr/bin/env python3
"""Map fixed TAURUS stroma query through one saved stroma SCANVI realization.

Saves hard labels + soft probabilities + entropy. Never discards outputs.
Skips if predictions_manifest.json already complete.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from scvi.model import SCANVI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paths import ANNOT_PARQUET, LOGS, MODELS, PREDICTIONS, RMB, TAURUS_H5AD  # noqa: E402

sys.path.insert(0, str(RMB / "scripts"))
from run_predict_disease_mapping import (  # noqa: E402
    prepare_query_counts,
    register_query_on_model,
)

LOGGER = logging.getLogger("map_taurus_stroma")


def _load_stroma_query(max_cells: int | None, seed: int) -> anndata.AnnData:
    ann = pd.read_parquet(ANNOT_PARQUET)
    stroma = ann[ann["assigned_lineage"].astype(str) == "stroma"].copy()
    barcodes = stroma["barcode"].astype(str).to_numpy()
    source = anndata.read_h5ad(TAURUS_H5AD, backed="r")
    obs_names = pd.Index(source.obs_names.astype(str))
    pos = obs_names.get_indexer(barcodes)
    ok = pos >= 0
    if ok.mean() < 0.99:
        raise RuntimeError(f"Stroma barcode match too low: {ok.mean():.3f}")
    pos = pos[ok]
    stroma = stroma.loc[ok].copy()
    query_ids = barcodes[ok]
    if max_cells is not None and len(pos) > max_cells:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(pos), max_cells, replace=False))
        pos = pos[keep]
        stroma = stroma.iloc[keep].copy()
        query_ids = query_ids[keep]
    X = source.X[pos]
    if not sparse.issparse(X):
        X = sparse.csr_matrix(np.asarray(X))
    else:
        X = X.tocsr()
    var = source.var.copy()
    if getattr(source, "file", None) is not None:
        source.file.close()
    obs = stroma.reset_index(drop=True)
    obs["query_cell_id"] = query_ids
    # batch keys expected by models
    obs["sample_id"] = obs["sample_id"].astype(str)
    obs["sampleID"] = obs["sample_id"]
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()
    return adata


def _label_order(model, soft_value) -> list[str]:
    if isinstance(soft_value, pd.DataFrame):
        return soft_value.columns.astype(str).tolist()
    registry = model.adata_manager.get_state_registry("labels")
    mapping = registry.get("categorical_mapping")
    return [str(v) for v in np.asarray(mapping)]


def map_one(model_dir: Path, out_dir: Path, max_cells: int | None, seed: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / "predictions_manifest.json"
    if done.exists() and (out_dir / "predictions.parquet").exists():
        LOGGER.info("Skip complete: %s", out_dir)
        return out_dir

    train_man = json.loads((model_dir / "training_manifest.json").read_text())
    atlas = train_man["atlas"]
    omit = train_man["omission"]["omitted_study"]
    model_seed = int(train_man["seed"])

    LOGGER.info("Loading query stroma cells")
    query_raw = _load_stroma_query(max_cells, seed)
    LOGGER.info("Aligning query to model genes: %s", model_dir)
    query = prepare_query_counts(query_raw, model_dir)

    LOGGER.info("Loading SCANVI model with reference adata")
    reference = anndata.read_h5ad(model_dir / "adata.h5ad")
    model = SCANVI.load(str(model_dir), adata=reference)
    register_query_on_model(model, query)
    del reference

    t0 = time.time()
    hard = np.asarray(model.predict(query)).astype(str)
    soft_value = model.predict(query, soft=True)
    soft = soft_value.to_numpy() if isinstance(soft_value, pd.DataFrame) else np.asarray(soft_value)
    labels = _label_order(model, soft_value)
    if soft.shape[1] != len(labels):
        raise ValueError(f"soft cols {soft.shape[1]} != labels {len(labels)}")
    eps = 1e-12
    entropy = -np.sum(soft * np.log(soft + eps), axis=1)
    max_n = np.log(soft.shape[1])
    norm_entropy = entropy / max_n if max_n > 0 else entropy
    max_post = soft.max(axis=1)
    elapsed = time.time() - t0

    pred = pd.DataFrame(
        {
            "query_cell_id": query.obs["query_cell_id"].astype(str).to_numpy(),
            "sample_id": query.obs["sample_id"].astype(str).to_numpy(),
            "Patient": query.obs["Patient"].astype(str).to_numpy()
            if "Patient" in query.obs
            else "",
            "Disease": query.obs["Disease"].astype(str).to_numpy()
            if "Disease" in query.obs
            else "",
            "leaf_prediction": hard,
            "max_posterior": max_post,
            "entropy": entropy,
            "normalized_entropy": norm_entropy,
            "atlas": atlas,
            "lineage": "stroma",
            "omitted_study": omit,
            "model_seed": model_seed,
        }
    )
    pred.to_parquet(out_dir / "predictions.parquet", index=False)
    # full posterior (cells x labels)
    soft_df = pd.DataFrame(soft, columns=labels)
    soft_df.insert(0, "query_cell_id", pred["query_cell_id"].to_numpy())
    soft_df.to_parquet(out_dir / "posterior.parquet", index=False)

    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir),
        "atlas": atlas,
        "omitted_study": omit,
        "model_seed": model_seed,
        "n_query_cells": int(len(pred)),
        "n_labels": int(len(labels)),
        "labels": labels,
        "predict_seconds": elapsed,
        "outputs": {
            "predictions": "predictions.parquet",
            "posterior": "posterior.parquet",
        },
    }
    tmp = done.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(done)
    LOGGER.info("Saved %s (%s cells, %.1fs)", out_dir, f"{len(pred):,}", elapsed)
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    model_dir = args.model_dir.expanduser()
    if not model_dir.is_absolute():
        model_dir = (ROOT / model_dir).resolve()
    else:
        model_dir = model_dir.resolve()
    models_root = MODELS.resolve()
    try:
        rel = model_dir.relative_to(models_root)
    except ValueError as exc:
        raise SystemExit(
            f"--model-dir must be under {models_root}, got {model_dir}"
        ) from exc
    out_dir = PREDICTIONS / rel
    if args.max_cells is not None:
        out_dir = PREDICTIONS / "_smoke" / rel
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS / f"map_{rel.as_posix().replace('/', '_')}.log"),
        ],
    )
    map_one(model_dir, out_dir, args.max_cells, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
