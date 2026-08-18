#!/usr/bin/env python3
"""Train full or leave-one-study SCANVI references for stroma or myeloid.

Never discards checkpoints. Smoke jobs write under _smoke/.

Usage:
  python src/train_lineage_reference.py --lineage myeloid --atlas HGCA --omit full --seed 0
  python src/train_lineage_reference.py --lineage myeloid --atlas PanGI --omit Lee2020 --seed 0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import torch
from scvi import settings as scvi_settings
from scvi.model import SCANVI, SCVI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paths import CONFIGS, HGCA_H5ADS, LOGS, MODELS, PANGI_H5AD, RMB  # noqa: E402

sys.path.insert(0, str(RMB))
from src.data.preparation import DataPreparation  # noqa: E402
from src.utils.config import load_config  # noqa: E402

LOGGER = logging.getLogger("train_lineage_reference")

LINEAGE_SPEC = {
    "stroma": {
        "recipe": CONFIGS / "stroma_scanvi_recipe_frozen.json",
        "hgca_h5ad": HGCA_H5ADS["stroma"],
        "hgca_config_name": "stroma",
        "pangi_config_name": "pangi_stroma",
        "pangi_lineage_values": ["Mesenchymal", "Endothelial"],
        "hgca_exclude_labels": ["Epithelial"],
    },
    "myeloid": {
        "recipe": CONFIGS / "myeloid_scanvi_recipe_frozen.json",
        "hgca_h5ad": HGCA_H5ADS["myeloid"],
        "hgca_config_name": "myeloid",
        "pangi_config_name": "pangi_myeloid",
        "pangi_lineage_values": ["Myeloid"],
        "hgca_exclude_labels": [],
    },
    "epithelial": {
        "recipe": CONFIGS / "epithelial_scanvi_recipe_frozen.json",
        "hgca_h5ad": HGCA_H5ADS["epithelial"],
        "hgca_config_name": "epithelial",
        "pangi_config_name": "pangi_epithelial",
        "pangi_lineage_values": ["Epithelial"],
        "hgca_exclude_labels": [],
    },
    "lymphoid": {
        "recipe": CONFIGS / "lymphoid_scanvi_recipe_frozen.json",
        "hgca_h5ad": HGCA_H5ADS["lymphoid"],
        "hgca_config_name": "lymphoid",
        "pangi_config_name": "pangi_lymphoid",
        "pangi_lineage_values": ["T and NK cells", "B and B plasma"],
        "hgca_exclude_labels": ["Epithelial"],
    },
}


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def _model_dir(atlas: str, lineage: str, omit: str, seed: int, smoke: bool = False) -> Path:
    omit_tag = "full" if omit in ("full", "", "none", "FULL") else f"omit_{omit}"
    base = MODELS / atlas / lineage
    if smoke:
        base = base / "_smoke"
    return base / omit_tag / f"seed{seed}"


def _gene_table(adata: anndata.AnnData) -> pd.DataFrame:
    symbol_column = next(
        (c for c in ("gene_symbol", "gene_symbols", "feature_name") if c in adata.var),
        None,
    )
    symbols = (
        adata.var[symbol_column].astype(str).to_numpy()
        if symbol_column
        else np.repeat("", adata.n_vars)
    )
    return pd.DataFrame(
        {"gene_id": adata.var_names.astype(str).to_numpy(), "gene_symbol": symbols}
    )


def _raw_count_pangi(source: anndata.AnnData, indices: np.ndarray) -> anndata.AnnData:
    if source.raw is None:
        raise ValueError("PanGI h5ad has no .raw matrix")
    counts = source.raw.X[indices, :]
    if not sparse.issparse(counts):
        counts = sparse.csr_matrix(np.asarray(counts))
    else:
        counts = counts.tocsr()
    if counts.data.size and not np.allclose(counts.data, np.rint(counts.data)):
        raise ValueError("PanGI .raw.X is not integer-valued")
    adata = anndata.AnnData(
        X=counts,
        obs=source.obs.iloc[indices].copy(),
        var=source.raw.var.copy(),
    )
    adata.layers["counts"] = counts.copy()
    return adata


def _omission_record(obs: pd.DataFrame, study_key: str, label_key: str, omit: str) -> dict:
    total = len(obs)
    if omit in ("full", "", "none", "FULL"):
        return {
            "omitted_study": "full",
            "n_cells_removed": 0,
            "frac_lineage_removed": 0.0,
            "n_donors_removed": 0,
            "remaining_reference_size": total,
            "labels_lost_entirely": [],
            "n_labels_gt10pct_removed": 0,
            "n_labels_gt25pct_removed": 0,
            "n_labels_gt50pct_removed": 0,
        }
    mask = obs[study_key].astype(str) == str(omit)
    removed = obs.loc[mask]
    kept = obs.loc[~mask]
    lab_tot = obs[label_key].astype(str).value_counts()
    lab_rem = removed[label_key].astype(str).value_counts()
    rem_frac = (lab_rem / lab_tot.reindex(lab_rem.index)).fillna(0)
    lost = [lab for lab, n in lab_rem.items() if n == lab_tot.get(lab, -1)]
    sample_col = "sampleID" if "sampleID" in removed.columns else "sample_id"
    return {
        "omitted_study": omit,
        "n_cells_removed": int(mask.sum()),
        "frac_lineage_removed": float(mask.sum() / max(total, 1)),
        "n_sample_batches_removed": int(removed[sample_col].nunique()) if sample_col in removed else None,
        "celltype_composition_removed": lab_rem.to_dict(),
        "remaining_reference_size": int(len(kept)),
        "labels_lost_entirely": lost,
        "n_labels_gt10pct_removed": int((rem_frac > 0.10).sum()),
        "n_labels_gt25pct_removed": int((rem_frac > 0.25).sum()),
        "n_labels_gt50pct_removed": int((rem_frac > 0.50).sum()),
        "max_label_frac_removed": float(rem_frac.max()) if len(rem_frac) else 0.0,
    }


def train_pangi(
    lineage: str,
    omit: str,
    seed: int,
    epochs: int | None,
    max_cells: int | None,
    smoke: bool = False,
) -> Path:
    spec = LINEAGE_SPEC[lineage]
    out = _model_dir("PanGI", lineage, omit, seed, smoke=smoke)
    done = out / "training_manifest.json"
    if done.exists() and (out / "model.pt").exists():
        LOGGER.info("Immutable checkpoint exists: %s", out)
        return out
    if out.exists() and any(out.iterdir()) and not done.exists():
        raise FileExistsError(f"Incomplete non-empty model dir (refuse to clobber): {out}")

    recipe = json.loads(spec["recipe"].read_text())
    pangi_config = load_config(spec["pangi_config_name"], config_dir=str(RMB / "configs"))
    hgca_config = load_config(spec["hgca_config_name"], config_dir=str(RMB / "configs"))

    np.random.seed(seed)
    torch.manual_seed(seed)
    scvi_settings.seed = seed

    lineage_vals = spec["pangi_lineage_values"]
    source = anndata.read_h5ad(PANGI_H5AD, backed="r")
    mask = source.obs["level_1_annot"].astype(str).isin(lineage_vals)
    lineage_obs = source.obs.loc[mask]
    omission = _omission_record(lineage_obs, "study", "level_3_annot", omit)
    if omit not in ("full", "", "none", "FULL"):
        mask = mask & (source.obs["study"].astype(str) != str(omit))
    indices = np.flatnonzero(mask.to_numpy())
    if max_cells is not None and len(indices) > max_cells:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, max_cells, replace=False))
    LOGGER.info("Materializing %s PanGI %s cells (omit=%s)", f"{len(indices):,}", lineage, omit)
    adata = _raw_count_pangi(source, indices)
    if getattr(source, "file", None) is not None:
        source.file.close()

    train_config = json.loads(json.dumps(pangi_config))
    train_config["scanvi"] = dict(hgca_config["scanvi"])
    train_config["scanvi"]["use_pretrained_embedding"] = False
    train_config["scanvi"]["use_scvi_pretrain"] = False
    train_config["labels"] = ["level_3_annot"]
    train_config["training"] = {"exclude_hgca_celltype_v1_labels": []}
    prep = DataPreparation(train_config)
    adata = prep.preprocess(adata, for_method="scanvi", copy_adata=False)
    adata = prep.subset_cells_for_supervised_training(adata, "level_3_annot", copy=True)

    batch_key = "sampleID"
    label_key = "level_3_annot"
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")
    adata.obs[label_key] = adata.obs[label_key].astype("category")

    params = dict(hgca_config["scanvi"])
    n_epochs = int(epochs if epochs is not None else params.get("n_epochs", 10))
    batch_size = int(params.get("batch_size", 256))

    tmp = out.with_name(out.name + f".tmp-{os.getpid()}")
    if tmp.exists():
        import shutil

        shutil.rmtree(tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)

    SCANVI.setup_anndata(
        adata,
        labels_key=label_key,
        batch_key=batch_key,
        layer="counts",
        unlabeled_category="Unknown",
    )
    model = SCANVI(
        adata,
        n_latent=30,
        n_layers=2,
        dropout_rate=0.1,
        gene_likelihood="nb",
    )
    t0 = time.time()
    model.train(max_epochs=n_epochs, batch_size=batch_size)
    training_seconds = time.time() - t0
    model.save(tmp, overwrite=True, save_anndata=True)
    _gene_table(adata).to_csv(tmp / "genes.csv", index=False)

    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "atlas": "PanGI",
        "lineage": lineage,
        "seed": seed,
        "recipe_path": str(spec["recipe"]),
        "recipe_notes": recipe["PanGI"],
        "source_h5ad": str(PANGI_H5AD.resolve()),
        "model_class": "direct SCANVI (no scVI pretraining)",
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_labels": int(adata.obs[label_key].nunique()),
        "n_batches": int(adata.obs[batch_key].nunique()),
        "label_counts": adata.obs[label_key].astype(str).value_counts().sort_index().to_dict(),
        "omission": omission,
        "params": {
            "n_latent": 30,
            "n_layers": 2,
            "dropout_rate": 0.1,
            "gene_likelihood": "nb",
            "epochs": n_epochs,
            "batch_size": batch_size,
            "seed": seed,
            "hvg_count": 4000,
            "use_scvi_pretrain": False,
        },
        "training_seconds": training_seconds,
        "hostname": os.uname().nodename,
    }
    _atomic_json(tmp / "training_manifest.json", manifest)
    tmp.replace(out)
    LOGGER.info("Saved %s", out)
    return out


def train_hgca(
    lineage: str,
    omit: str,
    seed: int,
    epochs: int | None,
    max_cells: int | None,
    smoke: bool = False,
) -> Path:
    spec = LINEAGE_SPEC[lineage]
    h5ad = spec["hgca_h5ad"]
    out = _model_dir("HGCA", lineage, omit, seed, smoke=smoke)
    done = out / "training_manifest.json"
    if done.exists() and (out / "model.pt").exists():
        LOGGER.info("Immutable checkpoint exists: %s", out)
        return out
    if out.exists() and any(out.iterdir()) and not done.exists():
        raise FileExistsError(f"Incomplete non-empty model dir (refuse to clobber): {out}")

    recipe = json.loads(spec["recipe"].read_text())
    hgca_config = load_config(spec["hgca_config_name"], config_dir=str(RMB / "configs"))
    np.random.seed(seed)
    torch.manual_seed(seed)
    scvi_settings.seed = seed

    exclude = set(spec["hgca_exclude_labels"])
    # Single full load (avoids reading large epithelial objects twice).
    adata = anndata.read_h5ad(h5ad)
    full_obs = adata.obs
    if exclude:
        full_obs = full_obs[~full_obs["hgca_celltype_v1"].astype(str).isin(exclude)]
    omission = _omission_record(full_obs, "dataset_id", "hgca_celltype_v1", omit)
    if exclude:
        adata = adata[~adata.obs["hgca_celltype_v1"].astype(str).isin(exclude)].copy()
    if omit not in ("full", "", "none", "FULL"):
        adata = adata[adata.obs["dataset_id"].astype(str) != str(omit)].copy()

    if max_cells is not None and adata.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(adata.n_obs, max_cells, replace=False))
        adata = adata[idx].copy()

    if "counts" not in adata.layers:
        # Prefer view-free assignment without duplicating X when already CSR counts.
        x = adata.X
        adata.layers["counts"] = x if sparse.issparse(x) else sparse.csr_matrix(x)

    train_config = json.loads(json.dumps(hgca_config))
    train_config["data"]["input_path"] = str(h5ad)
    prep = DataPreparation(train_config)
    adata = prep.preprocess(adata, for_method="scanvi", copy_adata=False)
    adata = prep.subset_cells_for_supervised_training(adata, "hgca_celltype_v1", copy=True)

    batch_key = "sample_id"
    label_key = "hgca_celltype_v1"
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")
    adata.obs[label_key] = adata.obs[label_key].astype("category")

    params = dict(hgca_config["scanvi"])
    n_epochs = int(epochs if epochs is not None else params.get("n_epochs", 10))
    scvi_epochs = int(n_epochs if epochs is not None else params.get("scvi_epochs", n_epochs))
    batch_size = int(params.get("batch_size", 256))

    tmp = out.with_name(out.name + f".tmp-{os.getpid()}")
    if tmp.exists():
        import shutil

        shutil.rmtree(tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)

    SCVI.setup_anndata(adata, batch_key=batch_key, layer="counts")
    scvi_model = SCVI(
        adata,
        n_latent=30,
        n_layers=2,
        dropout_rate=0.1,
        gene_likelihood="nb",
    )
    t0 = time.time()
    scvi_model.train(max_epochs=scvi_epochs, batch_size=batch_size)
    SCANVI.setup_anndata(
        adata,
        labels_key=label_key,
        batch_key=batch_key,
        layer="counts",
        unlabeled_category="Unknown",
    )
    model = SCANVI.from_scvi_model(
        scvi_model,
        labels_key=label_key,
        unlabeled_category="Unknown",
    )
    model.train(max_epochs=n_epochs, batch_size=batch_size)
    training_seconds = time.time() - t0
    model.save(tmp, overwrite=True, save_anndata=True)
    _gene_table(adata).to_csv(tmp / "genes.csv", index=False)

    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "atlas": "HGCA",
        "lineage": lineage,
        "seed": seed,
        "recipe_path": str(spec["recipe"]),
        "recipe_notes": recipe["HGCA"],
        "source_h5ad": str(h5ad.resolve()),
        "model_class": "SCVI pretrain -> SCANVI (notebook 04 parity)",
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_labels": int(adata.obs[label_key].nunique()),
        "n_batches": int(adata.obs[batch_key].nunique()),
        "label_counts": adata.obs[label_key].astype(str).value_counts().sort_index().to_dict(),
        "omission": omission,
        "params": {
            "n_latent": 30,
            "n_layers": 2,
            "dropout_rate": 0.1,
            "gene_likelihood": "nb",
            "scvi_epochs": scvi_epochs,
            "scanvi_epochs": n_epochs,
            "batch_size": batch_size,
            "seed": seed,
            "hvg_count": 4000,
            "use_scvi_pretrain": True,
        },
        "training_seconds": training_seconds,
        "hostname": os.uname().nodename,
    }
    _atomic_json(tmp / "training_manifest.json", manifest)
    tmp.replace(out)
    LOGGER.info("Saved %s", out)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--lineage",
        choices=["stroma", "myeloid", "epithelial", "lymphoid"],
        required=True,
    )
    p.add_argument("--atlas", choices=["HGCA", "PanGI"], required=True)
    p.add_argument("--omit", default="full")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-cells", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                LOGS
                / f"train_{args.atlas}_{args.lineage}_{args.omit}_seed{args.seed}.log"
            ),
        ],
    )
    if args.atlas == "PanGI":
        train_pangi(args.lineage, args.omit, args.seed, args.epochs, args.max_cells, smoke=args.smoke)
    else:
        train_hgca(args.lineage, args.omit, args.seed, args.epochs, args.max_cells, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
