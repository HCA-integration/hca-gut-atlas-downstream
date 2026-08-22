#!/usr/bin/env python3
"""
Reference-mapping benchmark for one lineage — **SCANVI only** (leave-one-dataset-out).

This script lives in ``reference_mapping_benchmark``; use any conda env that has
``scvi-tools``, ``scanpy``, and optionally ``mapqc`` (``pip install mapqc``), not necessarily a
scimilarity-specific environment.

Local single fold::
    python workflows/slurm/run_lineage_benchmark.py --lineage myeloid

Full LODO CV (all datasets)::
    python workflows/slurm/run_lineage_benchmark.py --lineage myeloid --lodo-cv-all

Pilot: N random LODO folds (``benchmark.random_state`` fixes which holdouts)::
    python workflows/slurm/run_lineage_benchmark.py --lineage myeloid --lodo-cv-all --max-lodo-folds 5 --run-name pilot5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preparation import DataPreparation
from src.evaluation.mapqc_scanvi import run_mapqc_after_scanvi
from src.evaluation.metrics import BenchmarkEvaluator
from src.models.scanvi_benchmark import SCANVITrainer
from src.utils.config import load_config
from src.utils.hgca_obs import normalize_hgca_obs_columns


def _ensure_benchmark_block(config: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "split_mode": "leave_one_dataset_out",
        "dataset_id_column": None,
        "holdout_dataset_id": None,
        "lodo_fold_index": 0,
        "test_size": 0.2,
        "random_state": 42,
        "max_lodo_folds": None,
    }
    bm = dict(config.get("benchmark") or {})
    for k, v in defaults.items():
        bm.setdefault(k, v)
    config["benchmark"] = bm
    return config


def save_scanvi_results(results: dict, output_dir: Path, label_type: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    safe_label = label_type.replace(" ", "_")
    results["predictions"].to_csv(
        predictions_dir / f"predictions_scanvi_{safe_label}.csv", index=False
    )
    summary = {
        "method": results["method"],
        "label_type": results["label_type"],
        "accuracy": results["accuracy"],
        "n_train": results["n_train"],
        "n_test": results["n_test"],
        "n_labels": results["n_labels"],
        "training_time": results.get("training_time"),
    }
    with open(output_dir / f"scanvi_{safe_label}_results.json", "w") as f:
        json.dump([summary], f, indent=2)


def _run_evaluation_plots(output_dir: Path, label_key: str) -> None:
    evaluator = BenchmarkEvaluator(str(output_dir))
    predictions = evaluator.load_all_predictions(
        methods=["scanvi"],
        label_types=[label_key.replace(" ", "_")],
    )
    if not predictions:
        return
    metrics = evaluator.calculate_all_metrics(predictions)
    metrics.to_csv(output_dir / "per_class_metrics.csv", index=False)
    for label_type in metrics["label_type"].unique():
        evaluator.create_f1_comparison_plot(metrics, label_type)
        evaluator.create_accuracy_heatmap(metrics, label_type)


def run_scanvi_split(
    config: Dict[str, Any],
    adata_train_scanvi,
    adata_test_scanvi,
    label_key: str,
    output_dir: Path,
    split_meta: Dict[str, Any],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "split_meta.json", "w") as f:
        json.dump(split_meta, f, indent=2, default=str)

    depth_dir = output_dir / "annotation_depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    sm = split_meta

    scanvi_trainer = SCANVITrainer(config["scanvi"])
    scanvi_results = scanvi_trainer.train_and_evaluate(
        adata_train_scanvi,
        adata_test_scanvi,
        label_key,
        batch_key=config.get("batch_key"),
        split_meta=sm,
        annotation_depth_output_dir=str(depth_dir),
    )
    save_scanvi_results(scanvi_results, output_dir, label_key)

    out: Dict[str, Any] = {
        "split_meta": sm,
        "accuracy": {"scanvi": float(scanvi_results["accuracy"])},
    }

    model = scanvi_results.get("model")
    if model is not None:
        try:
            mq = run_mapqc_after_scanvi(
                model,
                adata_train_scanvi,
                adata_test_scanvi,
                config,
                output_dir,
                label_key=label_key,
                scanvi_accuracy=float(scanvi_results["accuracy"]),
            )
            if mq:
                out["mapqc"] = mq
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning("mapQC step failed: %s", e, exc_info=True)

    combined = _write_annotation_depth_comparison_if_present(depth_dir)
    if combined:
        out["annotation_depth_comparison_figure"] = str(combined)

    return out


def _write_annotation_depth_comparison_if_present(depth_dir: Path) -> Optional[Path]:
    from src.evaluation.annotation_depth import write_annotation_depth_multi_method_figure

    depth_dir = Path(depth_dir)
    if not depth_dir.is_dir():
        return None
    summaries = sorted(depth_dir.glob("annotation_depth_summary_by_method_*.csv"))
    if len(summaries) < 1:
        return None
    out = depth_dir / "annotation_depth_comparison_all_methods.png"
    write_annotation_depth_multi_method_figure(
        summaries,
        out,
        title="Annotation depth distribution (levels 1–5) — SCANVI benchmark",
    )
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="SCANVI LODO benchmark (one lineage).")
    _HGCA_LINEAGES = ("myeloid", "lymphoid", "epithelial", "stroma")
    _PANGI_LINEAGES = (
        "pangi_myeloid",
        "pangi_lymphoid",
        "pangi_epithelial",
        "pangi_stroma",
        "pangi_neural",
    )
    p.add_argument(
        "--lineage",
        required=True,
        choices=list(_HGCA_LINEAGES + _PANGI_LINEAGES),
    )
    p.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    p.add_argument(
        "--label-key",
        default=None,
        help="obs column for labels (default: first config labels entry, else hgca_celltype_v1).",
    )
    p.add_argument("--lodo-fold-index", type=int, default=None)
    p.add_argument("--lodo-cv-all", action="store_true")
    p.add_argument("--max-lodo-folds", type=int, default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--no-normalize-hgca-obs", action="store_true")
    p.add_argument("--skip-plots", action="store_true")
    args = p.parse_args(argv)

    project_root: Path = args.project_root.resolve()
    sys.path.insert(0, str(project_root))

    config = load_config(args.lineage, config_dir=str(project_root / "configs"))
    config = _ensure_benchmark_block(config)

    if args.lodo_fold_index is not None:
        config["benchmark"]["lodo_fold_index"] = int(args.lodo_fold_index)
    if args.max_lodo_folds is not None:
        config["benchmark"]["max_lodo_folds"] = int(args.max_lodo_folds)

    label_key = args.label_key
    if label_key is None:
        lc = config.get("labels") or []
        label_key = lc[0] if lc else "hgca_celltype_v1"

    data_prep = DataPreparation(config)
    adata_raw = data_prep.load_lineage_data()
    if not args.no_normalize_hgca_obs and not str(args.lineage).startswith("pangi"):
        adata_raw = normalize_hgca_obs_columns(adata_raw)

    adata_scanvi = data_prep.preprocess(adata_raw, for_method="scanvi")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or ts
    base_out = Path(config["output"]["results_dir"])
    if not base_out.is_absolute():
        base_out = project_root / base_out
    run_root = base_out / "slurm_benchmarks" / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    bm_cfg = config.get("benchmark") or {}
    manifest: Dict[str, Any] = {
        "lineage": args.lineage,
        "label_key": label_key,
        "lodo_cv_all": bool(args.lodo_cv_all),
        "run_name": run_name,
        "started_utc": ts,
        "pipeline": "scanvi_only",
        "reference_name": bm_cfg.get("reference_name")
        or ("HGCA" if not str(args.lineage).startswith("pangi") else "PanGI Healthy"),
        "mapqc_match_scanvi_accuracy": bool(
            (config.get("mapqc") or {}).get("match_scanvi_accuracy", False)
        ),
    }

    if args.lodo_cv_all:
        fold_results = []
        for ad_tr_sv, ad_te_sv, sm in data_prep.iter_split_for_benchmark_lodo(
            adata_scanvi, label_key
        ):
            hold = str(sm.get("holdout_dataset_id", "fold"))
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in hold)[:80]
            sub = run_root / f"fold_{safe}"
            r = run_scanvi_split(
                config,
                ad_tr_sv,
                ad_te_sv,
                label_key,
                sub,
                sm,
            )
            fold_results.append(r)
            if not args.skip_plots:
                _run_evaluation_plots(sub, label_key)
        manifest["n_folds"] = len(fold_results)
        manifest["max_lodo_folds"] = config["benchmark"].get("max_lodo_folds")
        manifest["fold_results"] = fold_results
    else:
        ad_tr_sv, ad_te_sv, sm = data_prep.split_for_benchmark(
            adata_scanvi, label_key, copy=True
        )
        sub = run_root / "single_lodo"
        manifest["single_lodo"] = run_scanvi_split(
            config,
            ad_tr_sv,
            ad_te_sv,
            label_key,
            sub,
            sm,
        )
        if not args.skip_plots:
            _run_evaluation_plots(sub, label_key)

    manifest_path = run_root / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
